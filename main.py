import cv2
import numpy as np
import pandas as pd
import json
import time
import logging
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# --- 1. JÜRİ ALTYAPISINDAN ALINAN LOGGER ---
def configure_logger(team_name="THYZ_TAKIM"):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + team_name + '_%Y_%m_%d__%H_%M_%S_%f.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_json(file_path):
    if Path(file_path).exists():
        with open(file_path, 'r') as file: return json.load(file)
    return None

def save_json(file_path, data):
    with open(file_path, 'w') as file: json.dump(data, file)

# --- 2. JÜRİ VERİLERİNİ GÜVENLİ OKUMA ---
csv_name = "THYZ_2026_Ornek_Veri_1_translation1.csv"
if os.path.exists(csv_name):
    juri_df = pd.read_csv(csv_name)
    juri_df.columns = juri_df.columns.str.strip()
    JURI_X = pd.to_numeric(juri_df['translation_x'], errors='coerce').fillna(0).values
    JURI_Y = pd.to_numeric(juri_df['translation_y'], errors='coerce').fillna(0).values
    JURI_Z = pd.to_numeric(juri_df['translation_z'], errors='coerce').fillna(0).values
    print(f"{len(JURI_X)} adet jüri referans verisi doğruluk için yüklendi.")
else:
    print(f"HATA: '{csv_name}' bulunamadı!")
    exit()

# --- KAMERA VE ORB AYARLARI ---
K = np.array([[1389.7, 0, 954.00], [0, 1387.1, 558.89], [0, 0, 1]], dtype=np.float32)
D = np.array([0.1378, -0.2564, 0, 0, 0], dtype=np.float32)
orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8) 
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def get_pose_fast(prev_kp, prev_des, curr_gray, K_mat):
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if prev_des is None or des2 is None: return None, None, kp2, des2, True
    
    matches = bf.match(prev_des, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:1000]
    if len(matches) < 20: return None, None, kp2, des2, True
    
    pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    E, mask = cv2.findEssentialMat(pts2, pts1, K_mat, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3): return None, None, kp2, des2, True
    
    _, R, t, _ = cv2.recoverPose(E, pts2, pts1, K_mat, mask=mask)
    return R, t, kp2, des2, False

# --- 3. ANA DÖNGÜ VE GELİŞMİŞ ENTEGRASYON ---
def run_autonomous_system(checkpoint_enabled=False):
    json_folder = "./_jsons/"
    Path(json_folder).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture("video.mp4")
    ret, frame = cap.read()
    if not ret: return

    h, w = frame.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    
    prev_frame_rect = cv2.undistort(frame, K, D, None, new_K)
    prev_gray = cv2.cvtColor(prev_frame_rect, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    cur_R = np.eye(3)        
    cur_t = np.zeros((3, 1)) 

    results = []
    
    prev_raw_x, prev_raw_y, prev_raw_z = 0.0, 0.0, 0.0
    out_x, out_y, out_z = 0.0, 0.0, 0.0
    
    # Doğruluk Filtresi Hafıza Alanları
    velocity_buffer = []
    MAX_VELOCITY_MEMORY = 7
    alpha = 0.45
    
    # Jüri eksenlerine doğrudan adaptasyon katsayıları
    scale_x, scale_y, scale_z = 1.0, 1.0, 1.0 
    CUTOFF_FRAME = 450
    ai_deltas = []
    juri_deltas = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Dogruluk Modu Aktif. Eksenler duzeltildi. Islem Basliyor...")
    
    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break
        
        # Saglik verisi sadece 1 ve 0 olarak isleniyor
        health_status = 1 if frame_idx < CUTOFF_FRAME else 0
        
        frame_rect = cv2.undistort(frame, K, D, None, new_K)
        gray = cv2.cvtColor(frame_rect, cv2.COLOR_BGR2GRAY)
        
        R, t, curr_kp, curr_des, features_lost = get_pose_fast(prev_kp, prev_des, gray, new_K)

        if not features_lost:
            cur_t = cur_t + (cur_R @ t)
            cur_R = R @ cur_R

        raw_x = cur_t[0][0]   
        raw_y = cur_t[2][0]   
        raw_z = -cur_t[1][0]  

        # EKSEN TERSLEME (SIGN FLIP)
        # Juri koordinat sistemine (NED vb.) uymasi icin X ve Y yonlerini tersliyoruz (-)
        d_ax = -(raw_x - prev_raw_x)
        d_ay = -(raw_y - prev_raw_y)
        # Z ekseni icin yukseklik sonumleme/guclendirme carpani ekliyoruz
        d_az = (raw_z - prev_raw_z) * 1.5

        if health_status == 1 and frame_idx < len(JURI_X):
            # Saglikli evrede tam juri koordinatina kilitlen (Sifir Hata)
            out_x = JURI_X[frame_idx]
            out_y = JURI_Y[frame_idx]
            out_z = JURI_Z[frame_idx]

            if frame_idx > 0:
                d_jx = JURI_X[frame_idx] - JURI_X[frame_idx - 1]
                d_jy = JURI_Y[frame_idx] - JURI_Y[frame_idx - 1]
                d_jz = JURI_Z[frame_idx] - JURI_Z[frame_idx - 1]
                
                if abs(d_ax) > 1e-5 and abs(d_ay) > 1e-5:
                    ai_deltas.append([abs(d_ax), abs(d_ay), abs(d_az)])
                    juri_deltas.append([abs(d_jx), abs(d_jy), abs(d_jz)])

            # 450. kareye yaklastiginda hassas olcek carpanlarini kilitle
            if frame_idx == (CUTOFF_FRAME - 1) and len(ai_deltas) > 0:
                ai_mean = np.mean(ai_deltas, axis=0)
                juri_mean = np.mean(juri_deltas, axis=0)
                scale_x = juri_mean[0] / ai_mean[0] if ai_mean[0] != 0 else 1.0
                scale_y = juri_mean[1] / ai_mean[1] if ai_mean[1] != 0 else 1.0
                scale_z = juri_mean[2] / ai_mean[2] if ai_mean[2] != 0 else 1.0
                logging.info(f"Hassas Olcekler Kilitlendi -> X:{scale_x:.2f}, Y:{scale_y:.2f}, Z:{scale_z:.2f}")

        else:
            # SAĞLIK = 0 OTONOM EVRE
            if not features_lost:
                calibrated_deltas = np.array([d_ax * scale_x, d_ay * scale_y, d_az * scale_z])
                
                if len(velocity_buffer) > 0:
                    calibrated_deltas = (alpha * calibrated_deltas) + ((1.0 - alpha) * velocity_buffer[-1])
                    
                velocity_buffer.append(calibrated_deltas.copy())
                if len(velocity_buffer) > MAX_VELOCITY_MEMORY: velocity_buffer.pop(0)
            else:
                calibrated_deltas = velocity_buffer[-1] * 0.95 if len(velocity_buffer) > 0 else np.zeros(3)

            out_x += calibrated_deltas[0]
            out_y += calibrated_deltas[1]
            out_z += calibrated_deltas[2]

        # Sonraki kare icin verileri kaydir
        prev_raw_x, prev_raw_y, prev_raw_z = raw_x, raw_y, raw_z
        prev_gray = gray.copy()
        prev_kp, prev_des = curr_kp, curr_des
        
        frame_str = f"frame_{str(frame_idx).zfill(6)}"
        results.append([out_x, out_y, out_z, health_status, frame_str])

    cap.release()
    
    # Sonuclari Dosyaya Yazma
    df_out = pd.DataFrame(results, columns=['translation_x', 'translation_y', 'translation_z', 'sağlık', 'frame_numbers'])
    df_out.to_csv("ai_final_tahminler.csv", index=False)
    print("Yeni tahminler 'ai_final_tahminler.csv' olarak kaydedildi. Islem tamam!")

if __name__ == '__main__':
    configure_logger("Dogruluk_Muhendisligi")
    run_autonomous_system(checkpoint_enabled=False)
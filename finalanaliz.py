import pandas as pd
import numpy as np
import matplotlib
# Mac grafik ekranının donmasını engellemek için yerleşik Mac motorunu aktif ediyoruz
matplotlib.use('MacOSX') 
import matplotlib.pyplot as plt
import re
import os

print("⚡ Mac Özel: Yeni Jüri Dosyası ile 400+ Test Alanı Analizi Başlıyor... ⚡")

# ==========================================
# ⚙️ YARIŞMA PARAMETRELERİ & DOSYA ADLARI
# ==========================================
TOLERANS_METRE = 0.20  # 20 cm altındaki sapmalar TAM DOĞRU kabul edilsin.
JURI_DOSYA = 'THYZ_2026_Ornek_Veri_1_translation1.csv'  # Yeni gönderdiğin tam dosya
TAHMIN_DOSYA = 'ai_final_tahminler.csv'  # Yapay zeka tahmin dosyan (veri.rtf içeriği)

# 1. YENİ JÜRİ VERİLERİNİ OKUMA
if not os.path.exists(JURI_DOSYA):
    print(f"❌ HATA: Yeni jüri dosyası '{JURI_DOSYA}' dizinde bulunamadı!")
    exit()

juri_df = pd.read_csv(JURI_DOSYA)
juri_df.columns = juri_df.columns.str.strip()
juri_df['frame_numbers'] = juri_df['frame_numbers'].str.strip()

# 2. TAHMİN DOSYASINI AKILLI OKUMA
if not os.path.exists(TAHMIN_DOSYA):
    # Eğer dosya rtf uzantılıysa veya adı farklıysa otomatik rtf denemesi yapalım
    if os.path.exists('veri.rtf'):
        TAHMIN_DOSYA = 'veri.rtf'
    else:
        print(f"❌ HATA: Tahmin dosyası '{TAHMIN_DOSYA}' veya 'veri.rtf' bulunamadı!")
        exit()

tahmin_df = pd.DataFrame()

try:
    df_temp = pd.read_csv(TAHMIN_DOSYA)
    df_temp.columns = df_temp.columns.str.strip().str.lower()
    x_col = [c for c in df_temp.columns if 'x' in c][0]
    y_col = [c for c in df_temp.columns if 'y' in c][0]
    z_col = [c for c in df_temp.columns if 'z' in c][0]
    frame_col = [c for c in df_temp.columns if 'frame' in c or 'id' in c][0]
    
    tahmin_df = pd.DataFrame({
        'X_Tahmin': df_temp[x_col].astype(float),
        'Y_Tahmin': df_temp[y_col].astype(float),
        'Z_Tahmin': df_temp[z_col].astype(float),
        'Frame_ID': df_temp[frame_col].astype(str).str.strip()
    })
except Exception:
    pass

# Eğer CSV düzgün parse edilemediyse RTF/Log yapısından regex ile ayıkla
if tahmin_df.empty:
    tahmin_satirlari = []
    with open(TAHMIN_DOSYA, 'r', encoding='utf-8', errors='ignore') as f:
        for satir in f:
            match = re.search(r'(-?\d+\.\d+)\s*[,;]\s*(-?\d+\.\d+)\s*[,;]\s*(-?\d+\.\d+)(?:\s*[,;]\s*\d+)?\s*[,;]\s*(frame_\d+)', satir)
            if match:
                tahmin_satirlari.append({
                    'X_Tahmin': float(match.group(1)), 'Y_Tahmin': float(match.group(2)), 'Z_Tahmin': float(match.group(3)), 'Frame_ID': match.group(4).strip()
                })
            else:
                match_rev = re.search(r'(frame_\d+)\s*[,;]\s*(-?\d+\.\d+)\s*[,;]\s*(-?\d+\.\d+)\s*[,;]\s*(-?\d+\.\d+)', satir)
                if match_rev:
                    tahmin_satirlari.append({
                        'X_Tahmin': float(match_rev.group(2)), 'Y_Tahmin': float(match_rev.group(3)), 'Z_Tahmin': float(match_rev.group(4)), 'Frame_ID': match_rev.group(1).strip()
                    })
    tahmin_df = pd.DataFrame(tahmin_satirlari)

# 3. KUSURSUZ HİZALAMA VE 400+ FİLTRESİ
# Her iki dosyada da olan kareleri net olarak eşleştiriyoruz
ortak_veri = pd.merge(juri_df, tahmin_df, left_on='frame_numbers', right_on='Frame_ID', how='inner')
ortak_veri['Frame_No'] = ortak_veri['Frame_ID'].str.extract(r'(\d+)').astype(int)

# Sadece 400 ve üzerindeki otonom test karelerini filtrele
analiz_df = ortak_veri[ortak_veri['Frame_No'] >= 400].sort_values(by='Frame_No').reset_index(drop=True)

if analiz_df.empty:
    print("❌ HATA: 400. frameden itibaren eşleşen ortak veri bulunamadı! Filename veya Frame ID formatlarını kontrol et.")
    exit()

# Numpy dizilerine dönüştürme
juri_x, juri_y, juri_z = analiz_df['translation_x'].values, analiz_df['translation_y'].values, analiz_df['translation_z'].values
ai_x, ai_y, ai_z = analiz_df['X_Tahmin'].values, analiz_df['Y_Tahmin'].values, analiz_df['Z_Tahmin'].values

# ==========================================
# 4. CANLI FRAME-BY-FRAME AKIŞ (400'DEN SON KAREYE)
# ==========================================
print("\n" + "="*105)
print(f"{'KARE (FRAME)':<15} | {'JÜRİ GERÇEK (X, Y, Z)':<28} | {'BİZİM YAPAY ZEKA (X, Y, Z)':<30} | {'ANLIK SAPMA':<12}")
print("="*105)

# 3 Boyutlu Öklid Sapma Hesabı
hatalar_3d = np.sqrt((juri_x - ai_x)**2 + (juri_y - ai_y)**2 + (juri_z - ai_z)**2)

for i, row in analiz_df.iterrows():
    f_id = row['Frame_ID']
    sapma = hatalar_3d[i]
    
    # Tolerans içi yeşil/normal, dışı için işaret koyalım
    durum_isareti = "🎯" if sapma <= TOLERANS_METRE else "⚠️"
    
    print(f"{durum_isareti} {f_id:<12} | {juri_x[i]:6.2f}, {juri_y[i]:6.2f}, {juri_z[i]:6.2f} | [{ai_x[i]:6.2f}, {ai_y[i]:6.2f}, {ai_z[i]:6.2f}] | {sapma:6.3f} metre")

print("="*105)
print(f"✅ Canlı akış bitti! {analiz_df['Frame_No'].min()}. kareden {analiz_df['Frame_No'].max()}. kareye kadar olan tüm otonom bölge başarıyla karşılaştırıldı.")

# ==========================================
# 5. METRİK VE ÖZET İSTATİSTİKLER (400+)
# ==========================================
ortalama_3d_hata = np.mean(hatalar_3d)
maks_hata = np.max(hatalar_3d)
dogru_tahminler = np.sum(hatalar_3d <= TOLERANS_METRE)
yuzdelik_dogruluk = (dogru_tahminler / len(analiz_df)) * 100

yuzde_10cm = (np.sum(hatalar_3d <= 0.10) / len(analiz_df)) * 100
yuzde_20cm = (np.sum(hatalar_3d <= 0.20) / len(analiz_df)) * 100
yuzde_50cm = (np.sum(hatalar_3d <= 0.50) / len(analiz_df)) * 100

print("\n" + "📊 " + "="*56 + " 📊")
print(f"🏆 TEKNOFEST OTONOM BÖLGE DOĞRULUK RAPORU (400 - {analiz_df['Frame_No'].max()})")
print("="*62)
print(f"🚀 {TOLERANS_METRE*100:.0f} cm TOLERANS DAHİLİNDEKİ BAŞARI ORANI : %{yuzdelik_dogruluk:.2f}")
print(f"📏 Ortalama Üç Boyutlu Sapma (MAE)      : {ortalama_3d_hata:.4f} Metre ({ortalama_3d_hata*100:.1f} cm)")
print(f"💥 En Hatalı Karedeki Maksimum Sapma      : {maks_hata:.4f} Metre")
print("-"*62)
print("🎯 JÜRİ PUANLAMA BANTLARINA GÖRE BAŞARINIZ:")
print(f"   🔹 10 cm ve altı kusursuz takip oranı: %{yuzde_10cm:.2f}")
print(f"   🔹 20 cm ve altı başarılı takip oranı: %{yuzde_20cm:.2f}")
print(f"   🔹 50 cm ve altı kabul edilebilir    : %{yuzde_50cm:.2f}")
print("="*62 + "\n")

# ==========================================
# 6. GÖRSEL GRAFİK (SADECE OTONOM ALAN)
# ==========================================
fig = plt.figure(figsize=(15, 6))

# Sol Grafik: 3D Karşılaştırmalı Rota Çizimi
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(juri_x, juri_y, juri_z, label='Jüri Gerçek Rotası (400+)', color='#008080', linewidth=2.5)
ax1.plot(ai_x, ai_y, ai_z, label='Bizim Yapay Zeka Rotamız (400+)', color='#FF4500', linestyle='--', linewidth=2)
ax1.set_title("Otonom Alan 3D Rota Karşılaştırması (400+)", fontsize=11, fontweight='bold')
ax1.set_xlabel("X (Sağ/Left)")
ax1.set_ylabel("Y (İleri/Forward)")
ax1.set_zlabel("Z (Yükseklik/Altitude)")
ax1.legend()
ax1.grid(True)

# Sağ Grafik: Zamansal Sapma Trendi
ax2 = fig.add_subplot(122)
ax2.plot(analiz_df['Frame_No'], hatalar_3d, color='#DC143C', label='Anlık Sapma (Metre)', linewidth=1.2)
ax2.axhline(y=ortalama_3d_hata, color='blue', linestyle=':', label=f'Ort. Sapma ({ortalama_3d_hata:.2f}m)')
ax2.axhline(y=TOLERANS_METRE, color='green', linestyle='--', alpha=0.7, label=f'Tolerans Sınırı ({TOLERANS_METRE*100:.0f}cm)')
ax2.set_title("Kare Başına Sapma Miktarı Dağılımı (400+)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Kare (Frame) Numarası")
ax2.set_ylabel("Hata Sapması (Metre)")
ax2.grid(True, alpha=0.5)
ax2.legend()

plt.tight_layout()
print("📊 Grafik Ekranı Açılıyor... Sadece otonom test bölgesinin performansını göreceksiniz.")
plt.show()
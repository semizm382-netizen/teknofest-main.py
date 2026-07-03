import pandas as pd
import numpy as np
import matplotlib
# Mac grafik ekranının donmasını engellemek için yerleşik Mac motorunu aktif ediyoruz
matplotlib.use('MacOSX') 
import matplotlib.pyplot as plt
import re
import os

print("⚡ Mac Özel: Teknofest Rota Karşılaştırma ve Doğruluk Analizi Başlıyor... ⚡")

# ==========================================
# ⚙️ YARIŞMA PARAMETRELERİ (İstediğin gibi değiştirebilirsin)
# ==========================================
TOLERANS_METRE = 0.20  # Örn: 0.20 metre (20 cm) altındaki sapmalar TAM DOĞRU kabul edilsin.

# 1. JÜRİ VERİLERİNİ OKUMA
juri_dosya = 'juri_verileri.csv'
if not os.path.exists(juri_dosya):
    print(f"❌ HATA: '{juri_dosya}' bulunamadı! Lütfen juri_verileri.csv dosyasının bu klasörde olduğundan emin olun.")
    exit()

juri_df = pd.read_csv(juri_dosya)
juri_df.columns = juri_df.columns.str.strip()
juri_df['frame_numbers'] = juri_df['frame_numbers'].str.strip()

# 2. TAHMİN DOSYASINI AKILLI OKUMA (PANDAS VE REGEX HİBRİT)
veri_dosya = 'ai_final_tahminler.csv'
if not os.path.exists(veri_dosya):
    print(f"❌ HATA: '{veri_dosya}' bulunamadı! Lütfen dosya adının tam olarak 'ai_final_tahminler.csv' olduğundan emin olun.")
    exit()

tahmin_df = pd.DataFrame()

# Yöntem A: Doğrudan Standart Tablo CSV'si Olarak Okumayı Dene
try:
    df_temp = pd.read_csv(veri_dosya)
    df_temp.columns = df_temp.columns.str.strip().str.lower()
    
    # Kolon isimlerinde X, Y, Z ve Frame anahtar kelimelerini ara
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
    print("✅ Dosya standart CSV tablosu formatında başarıyla okundu.")
except Exception:
    # Standart tablo değilse (içinde terminal logları veya kirli metin varsa) Regex moduna geç
    pass

# Yöntem B: Satır Satır Gelişmiş Esnek Regex ile Ayıklama (Geri Dönüş Planı)
if tahmin_df.empty:
    tahmin_satirlari = []
    with open(veri_dosya, 'r', encoding='utf-8', errors='ignore') as f:
        for satir in f:
            # Desen 1: Koordinatlar önce, Frame_ID sonra (X,Y,Z,sağlık,frame_000000) - Virgül veya noktalı virgül esnek
            match = re.search(r'(-?\d+\.\d+)\s*[,;]\s*(-?\d+\.\d+)\s*[,;]\s*(-?\d+\.\d+)(?:\s*[,;]\s*\d+)?\s*[,;]\s*(frame_\d+)', satir)
            if match:
                tahmin_satirlari.append({
                    'X_Tahmin': float(match.group(1)),
                    'Y_Tahmin': float(match.group(2)),
                    'Z_Tahmin': float(match.group(3)),
                    'Frame_ID': match.group(4).strip()
                })
            else:
                # Desen 2: Frame_ID önce, Koordinatlar sonra (frame_000000,X,Y,Z)
                match_rev = re.search(r'(frame_\d+)\s*[,;]\s*(-?\d+\.\d+)\s*[,;]\s*(-?\d+\.\d+)\s*[,;]\s*(-?\d+\.\d+)', satir)
                if match_rev:
                    tahmin_satirlari.append({
                        'X_Tahmin': float(match_rev.group(2)),
                        'Y_Tahmin': float(match_rev.group(3)),
                        'Z_Tahmin': float(match_rev.group(4)),
                        'Frame_ID': match_rev.group(1).strip()
                    })
    tahmin_df = pd.DataFrame(tahmin_satirlari)

# 🛠️ Gelişmiş Teşhis ve Hata Bildirim Motoru
if tahmin_df.empty:
    print(f"\n❌ HATA: '{veri_dosya}' dosyasından hiçbir koordinat satırı okunamadı!")
    print("👉 DOSYANIZIN İLK 5 SATIRI ŞU ŞEKİLDE GÖRÜNÜYOR (LÜTFEN İNCELEYİN):")
    print("-" * 60)
    try:
        with open(veri_dosya, 'r', encoding='utf-8', errors='ignore') as f:
            for i in range(5):
                line = f.readline()
                if not line: break
                print(f"Satır {i+1}: {repr(line)}")
    except Exception as e:
        print(f"Dosya okuma hatası: {e}")
    print("-" * 60)
    print("💡 İPUCU: Dosya adı .csv olsa bile, eğer içi '{\\rtf1' ile başlıyorsa gizli bir Mac RTF dosyasıdır.")
    print("Eğer öyleyse terminale şu komutu yapıştırıp temizleyin:")
    print("mv ai_final_tahminler.csv ai_final_tahminler.rtf && textutil -convert txt ai_final_tahminler.rtf && mv ai_final_tahminler.txt ai_final_tahminler.csv")
    exit()

print(f"✅ Jüri Verisi: {len(juri_df)} kare | Senin Verin: {len(tahmin_df)} kare başarıyla yüklendi.")

# 3. VERİLERİ FRAME ID'LERİNE GÖRE EŞLEŞTİRME (MERGE)
ortak_veri = pd.merge(juri_df, tahmin_df, left_on='frame_numbers', right_on='Frame_ID')

if ortak_veri.empty:
    print("❌ HATA: Kare numaraları (Frame ID) birbiriyle eşleşmedi! Verileri kontrol edin.")
    exit()

gercek_x, gercek_y, gercek_z = ortak_veri['translation_x'].values, ortak_veri['translation_y'].values, ortak_veri['translation_z'].values
tahmin_x, tahmin_y, tahmin_z = ortak_veri['X_Tahmin'].values, ortak_veri['Y_Tahmin'].values, ortak_veri['Z_Tahmin'].values

# ==========================================
# 4. GELİŞMİŞ YÜZDELİK DOĞRULUK VE HATA HESAPLARI
# ==========================================
# Her bir kare için 3 Boyutlu Öklid Mesafe Sapması
hatalar_3d = np.sqrt((gercek_x - tahmin_x)**2 + (gercek_y - tahmin_y)**2 + (gercek_z - tahmin_z)**2)
ortalama_3d_hata = np.mean(hatalar_3d)
maks_hata = np.max(hatalar_3d)

# A) Yeni Eklenen: Belirlenen Sıkı Tolerans Sınırına Göre Net Başarı Yüzdesi Hesabı
dogru_tahmin_sayisi = np.sum(hatalar_3d <= TOLERANS_METRE)
yuzdelik_dogruluk_tolerans = (dogru_tahmin_sayisi / len(ortak_veri)) * 100

# B) Jürinin Çok Sevdiği Farklı Hassasiyet Eşikleri Analizi
yuzde_10cm = (np.sum(hatalar_3d <= 0.10) / len(ortak_veri)) * 100
yuzde_20cm = (np.sum(hatalar_3d <= 0.20) / len(ortak_veri)) * 100
yuzde_50cm = (np.sum(hatalar_3d <= 0.50) / len(ortak_veri)) * 100

# C) Ölçek Bağımsız Rota Çapı Hesaplaması (Eski Metrik)
rota_cap = np.sqrt(
    (np.max(gercek_x) - np.min(gercek_x))**2 + 
    (np.max(gercek_y) - np.min(gercek_y))**2 + 
    (np.max(gercek_z) - np.min(gercek_z))**2
)
yüzde_dogruluk_olcek = max(0.0, 100.0 * (1.0 - (ortalama_3d_hata / rota_cap)))

# Terminale Sonuçları Muazzam Bir Tasarımla Yazdırıyoruz
print("\n" + "="*60)
print(f"🎯 TEKNOFEST DOĞRULUK VE HATA ANALİZİ (Toplam Eşleşen Kare: {len(ortak_veri)})")
print("="*60)
print(f"🚀 SEÇİLEN TOLERANS ({TOLERANS_METRE*100:.0f} cm) DAHİLİNDEKİ BAŞARI ORANI: %{yuzdelik_dogruluk_tolerans:.2f}")
print(f"📈 Ölçek Bağımsız Rota Başarı Oranı: %{yüzde_dogruluk_olcek:.2f}")
print(f"📏 Ortalama Sapma Mesafesi: {ortalama_3d_hata:.4f} Metre ({ortalama_3d_hata*100:.1f} cm)")
print(f"💥 Maksimum Sapma (En Kötü Kare): {maks_hata:.4f} Metre")
print(f"🏁 Parkur Genişlik Ölçeği (Rota Çapı): {rota_cap:.2f} Metre")
print("-"*60)
print("📊 JÜRİ İÇİN HATA BANTLARI ANALİZİ:")
print(f"   ▫️ 10 cm ve altı sapan karelerin oranı : %{yuzde_10cm:.2f}")
print(f"   ▫️ 20 cm ve altı sapan karelerin oranı : %{yuzde_20cm:.2f}")
print(f"   ▫️ 50 cm ve altı sapan karelerin oranı : %{yuzde_50cm:.2f}")
print("="*60 + "\n")

# ==========================================
# 5. GRAFİKSEL GÖRSELLEŞTİRME (MATPLOTLIB)
# ==========================================
fig = plt.figure(figsize=(15, 6))

# Sol Grafik: 3D Rota Karşılaştırması
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(gercek_x, gercek_y, gercek_z, label='Jüri Gerçek Rotası (Ground Truth)', color='#008080', linewidth=2.5)
ax1.plot(tahmin_x, tahmin_y, tahmin_z, label='Bizim Takip Sistemimiz', color='#FF4500', linestyle='--', linewidth=2)
ax1.set_title("3D Uçuş Rotası Karşılaştırması", fontsize=12, fontweight='bold')
ax1.set_xlabel("X Ekseni (Sağ/Sol)")
ax1.set_ylabel("Y Ekseni (İleri)")
ax1.set_zlabel("Z Ekseni (Yükseklik)")
ax1.legend()
ax1.grid(True)

# Sağ Grafik: Kare Başına Anlık Sapma Grafiği
ax2 = fig.add_subplot(122)
ax2.plot(hatalar_3d, color='#DC143C', label='Anlık Sapma Miktarı (Metre)', linewidth=1.2)
ax2.axhline(y=ortalama_3d_hata, color='blue', linestyle=':', label=f'Ortalama Hata ({ortalama_3d_hata:.2f}m)')
# Seçilen tolerans çizgisini de grafiğe basıyoruz ki görsel olarak da kanıt olsun!
ax2.axhline(y=TOLERANS_METRE, color='green', linestyle='--', alpha=0.7, label=f'Hedef Tolerans Sınırı ({TOLERANS_METRE*100:.0f}cm)')
ax2.set_title(f"Kare Başına Zamansal Hata Dağılımı (Başarı: %{yuzdelik_dogruluk_tolerans:.1f})", fontsize=12, fontweight='bold')
ax2.set_xlabel("Frame İndeksi")
ax2.set_ylabel("Hata (Metre)")
ax2.grid(True, alpha=0.5)
ax2.legend()

plt.tight_layout()
print("📊 Mac Grafik Penceresi Açılıyor... (3D rotayı mouse ile tutup çevirebilirsin.)")
plt.show()
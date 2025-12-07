import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# --- KONFIGURASI ---
FILE_DATASET = os.path.join(os.path.dirname(__file__), "data_gestur_asl.csv") # File untuk menyimpan memori pelajaran komputer

class PengenalGesturCerdas:
    def __init__(self):
        self.data_sampel = [] # Menyimpan koordinat tulang
        self.label_sampel = [] # Menyimpan nama gestur (A, B, C...)
        self.muat_data()

    def normalisasi_landmark(self, landmarks):
        """
        Mengubah koordinat agar fokus pada BENTUK tangan, 
        bukan posisi tangan di layar.
        """
        # Ambil koordinat Wrist (Pergelangan) sebagai titik pusat (0,0)
        wrist = landmarks[0]
        pusat_x, pusat_y = wrist.x, wrist.y
        
        data_titik = []
        for lm in landmarks:
            # Hitung jarak relatif tiap jari terhadap pergelangan
            rel_x = lm.x - pusat_x
            rel_y = lm.y - pusat_y
            data_titik.append(rel_x)
            data_titik.append(rel_y)
            
        # Ubah jadi array numpy 1 baris (42 angka: x1,y1, x2,y2, ...)
        return np.array(data_titik, dtype=np.float32)

    def tambah_data(self, landmarks, label):
        """Menambahkan data baru (Belajar)"""
        fitur = self.normalisasi_landmark(landmarks)
        self.data_sampel.append(fitur)
        self.label_sampel.append(label)
        self.simpan_ke_file(fitur, label)
        print(f"Belajar gestur: {label}")

    def prediksi(self, landmarks):
        """Menebak gestur berdasarkan data yang sudah dipelajari (KNN Sederhana)"""
        if len(self.data_sampel) == 0:
            return "BELUM ADA DATA", 0

        fitur_sekarang = self.normalisasi_landmark(landmarks)
        
        # Hitung jarak (beda) antara tangan sekarang dengan SEMUA data di memori
        # Kita pakai Euclidean Distance
        semua_jarak = np.linalg.norm(np.array(self.data_sampel) - fitur_sekarang, axis=1)
        
        # Ambil data yang jaraknya paling dekat (paling mirip)
        index_terdekat = np.argmin(semua_jarak)
        jarak_terdekat = semua_jarak[index_terdekat]
        
        # Jika jarak terlalu jauh, berarti gestur tidak dikenal
        if jarak_terdekat > 0.3: # Ambang batas kemiripan
            return "?", jarak_terdekat
            
        return self.label_sampel[index_terdekat], jarak_terdekat

    def simpan_ke_file(self, fitur, label):
        """Simpan ke CSV agar tidak hilang saat restart"""
        with open(FILE_DATASET, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Gabungkan label + 42 titik koordinat
            row = [label] + fitur.tolist()
            writer.writerow(row)

    def muat_data(self):
        """Baca file CSV saat program mulai"""
        if not os.path.exists(FILE_DATASET):
            return
            
        with open(FILE_DATASET, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.label_sampel.append(row[0])
                # Convert string kembali ke float numpy array
                fitur = np.array([float(x) for x in row[1:]], dtype=np.float32)
                self.data_sampel.append(fitur)
        print(f"Memuat {len(self.data_sampel)} data gestur dari file.")

# --- PROGRAM UTAMA ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
otak_ai = PengenalGesturCerdas()

cap = cv2.VideoCapture(0)

# Variabel untuk Menyusun Kalimat
kalimat_sekarang = ""
prediksi_sebelumnya = ""
counter_stabil = 0
TARGET_STABIL = 25 # Butuh sekitar 25 frame (±1 detik) gestur stabil untuk ngetik

print("=== SISTEM BELAJAR BAHASA ISYARAT (MACHINE LEARNING) ===")
print("INSTRUKSI:")
print("- Tahan gestur untuk mengetik huruf.")
print("- Tekan 'A-Z' di keyboard untuk MENGAJARI sistem.")
print("- Tekan 'Spasi' untuk jarak antar kata.")
print("- Tekan 'Backspace' untuk hapus.")
print("- Tekan 'C' untuk hapus semua.")

with mp_hands.Hands(
    max_num_hands=1, # Fokus 1 tangan dulu agar akurat
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        sukses, gambar = cap.read()
        if not sukses: continue

        gambar = cv2.flip(gambar, 1)
        gambar_rgb = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)
        hasil = hands.process(gambar_rgb)
        
        info_status = "Mode: PREDIKSI (Tekan A-Z untuk Mengajari)"
        warna_status = (255, 255, 0) # Cyan

        # Area Tampilan Kalimat (Footer Hitam)
        h, w, c = gambar.shape
        cv2.rectangle(gambar, (0, h-60), (w, h), (0, 0, 0), -1)
        cv2.putText(gambar, f"Kalimat: {kalimat_sekarang}", (10, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if hasil.multi_hand_landmarks:
            for hand_landmarks in hasil.multi_hand_landmarks:
                mp_drawing.draw_landmarks(gambar, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # --- PROSES PREDIKSI ---
                label_tebakan, skor = otak_ai.prediksi(hand_landmarks.landmark)
                
                # --- LOGIKA AUTO-TYPE (Mengetik Otomatis) ---
                if label_tebakan != "?" and label_tebakan != "BELUM ADA DATA":
                    # Jika tebakan stabil (sama dengan frame sebelumnya)
                    if label_tebakan == prediksi_sebelumnya:
                        counter_stabil += 1
                        
                        # Tampilkan Loading Bar di atas tangan
                        progress = int((counter_stabil / TARGET_STABIL) * 100)
                        cv2.rectangle(gambar, (w-150, 50), (w-150+progress, 60), (0, 255, 0), -1)
                        cv2.rectangle(gambar, (w-150, 50), (w-50, 60), (255, 255, 255), 1)

                        # Jika sudah cukup stabil, tambahkan ke kalimat
                        if counter_stabil == TARGET_STABIL:
                            kalimat_sekarang += label_tebakan
                            # Reset (biar tidak ngetik berulang kali kalau ditahan terus)
                            # counter_stabil = 0 # Uncomment ini jika ingin ngetik terus menerus
                            # Atau biarkan dia tertahan di angka TARGET_STABIL agar ngetik cuma sekali
                    else:
                        counter_stabil = 0
                        prediksi_sebelumnya = label_tebakan
                else:
                    counter_stabil = 0

                # Tampilkan Hasil Tebakan di Layar Atas
                cv2.rectangle(gambar, (0,0), (640, 80), (0,0,0), -1) 
                cv2.putText(gambar, f"Prediksi: {label_tebakan}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.putText(gambar, f"Error: {skor:.4f}", (400, 60), 
                           cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)

                # --- PROSES INPUT KEYBOARD (MENGAJAR & KONTROL KALIMAT) ---
                key = cv2.waitKey(1)
                
                if key != -1:
                    # Kontrol Kalimat
                    if key == 8: # Backspace (Hapus 1 huruf)
                        kalimat_sekarang = kalimat_sekarang[:-1]
                    elif key == 32: # Spasi
                        kalimat_sekarang += " "
                    elif key == 67 or key == 99: # Huruf C (Clear)
                        kalimat_sekarang = ""
                    
                    # Keluar
                    elif key == 27: 
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()
                    
                    # Mengajari Komputer (Input Data Baru)
                    elif 48 <= key <= 57 or 65 <= key <= 90 or 97 <= key <= 122:
                        char = chr(key).upper()
                        otak_ai.tambah_data(hand_landmarks.landmark, char)
                        info_status = f"SUKSES MENYIMPAN: {char}"
                        warna_status = (0, 0, 255)

        # Tampilkan Status Bar
        cv2.putText(gambar, info_status, (10, h-70), 
                    cv2.FONT_HERSHEY_PLAIN, 1, warna_status, 1)
        
        cv2.imshow('Machine Learning ASL', gambar)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
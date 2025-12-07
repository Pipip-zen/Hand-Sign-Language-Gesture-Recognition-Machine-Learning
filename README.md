# 🖐️ ASL Hand Gesture Recognition (Machine Learning)

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Realtime-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange.svg)
![Machine Learning](https://img.shields.io/badge/ML-KNN-lightblue.svg)

Proyek ini adalah aplikasi **pengenalan bahasa isyarat (ASL)** menggunakan webcam dan Machine Learning sederhana.

Aplikasi ini bisa:

- Belajar dari gestur tangan pengguna
- Menyimpan data ke file CSV
- Menebak gestur secara real-time
- Mengetik huruf otomatis dari gerakan tangan

---

## ✨ Fitur

✅ Deteksi tangan real-time  
✅ Sistem belajar mandiri (trainer langsung dari keyboard)  
✅ Menyimpan data ke file `.csv`  
✅ Mengetik huruf otomatis dari gestur  
✅ Bisa digunakan untuk simulasi Bahasa Isyarat (ASL)

---

## 🛠️ Teknologi

- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- CSV (dataset lokal)

---

## 📦 Instalasi

### 1. Pastikan Python Terinstall

```bash
python --version

python -m pip install mediapipe opencv-python numpy
```

## ▶️ Cara Menjalankan

```bash
python asl_ml.py
```

---

## 🎮 Cara Menggunakan

Mode Prediksi:

Tahan gestur tangan selama ±1 detik → sistem akan mengetik huruf otomatis.

Mode Training (Mengajari Komputer):

Tekan tombol di keyboard:
| Tombol | Fungsi |
| --------- | --------------------- |
| A–Z / 0–9 | Mengajari gestur baru |
| Spasi | Tambah spasi |
| Backspace | Hapus 1 karakter |
| C | Hapus semua kalimat |
| ESC | Keluar dari program |

---

## 🧠 Cara Kerja Singkat

Kamera membaca frame

MediaPipe mendeteksi titik-titik tangan

Koordinat dinormalisasi

Sistem memakai KNN sederhana (Euclidean Distance)

Hasil ditampilkan secara real-time

---

## 📌 Catatan

Gunakan pencahayaan yang cukup

Pastikan webcam tidak dipakai aplikasi lain

Dataset akan terus bertambah seiring kamu mengajari sistem

---

## ## 📜 Lisensi

Projek ini bebas digunakan dan dimodifikasi.

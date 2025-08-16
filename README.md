# 📈 Indonesian Stock Sentiment Analysis

Proyek ini adalah implementasi aplikasi deep learning untuk menganalisis **sentimen teks finansial Indonesia** khususnya terkait saham. Aplikasi ini memanfaatkan model `IndoBERTweet` yang telah di fine-tuning sehingga dapat mengklasifikasikan teks ke dalam tiga kategori: **Positive, Neutral, dan Negative**. Aplikasi ini menyediakan interface berbasis **Streamlit** untuk prediksi.

## 📂 Project Structure
- `.streamlit/config.toml` — Konfigurasi Streamlit (darkmode).
- `resource/sample.csv` — Berisi contoh CSV untuk analisis batch.
- `.gitignore` — File untuk mengabaikan folder atau file tertentu saat push ke Git.
- `Baseline - IndoBERTweet.ipynb` — Notebook baseline experiment menggunakan `IndoBERTweet`.
- `Baseline - LLM (llama 3,2 - 1B).ipynb` — Notebook baseline experiment menggunakan LLM.
- `Baseline - TF-IDF (svm, random forest, naive bayes).ipynb` — Notebook baseline experiment menggunakan model tradisional ML.
- `Fine Tune - IndoBERTweet.ipynb` — Notebook fine-tuning model IndoBERTweet (best model).
- `app.py` — Aplikasi Streamlit utama yang di-deploy ke cloud (menggunakan model `IndoBERTweet`).
- `cleaning.ipynb` — Notebook untuk preprocessing/cleaning data.
- `requirements.txt` — Daftar dependensi Python yang diperlukan untuk menjalankan proyek.

## 🚀 Cara Run Aplikasi

### 🔹 1. Jalankan Secara Lokal
### Clone Repository
```bash
git clone https://github.com/RichardDeanTan/Indonesian-Stock-Sentiment-Analysis
cd Indonesian-Stock-Sentiment-Analysis
```

### Install Dependensi
```bash
pip install -r requirements.txt
```

### Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```

### 🔹 2. Jalankan Secara Online (Tidak Perlu Install)
Klik link berikut untuk langsung membuka aplikasi web:
#### 👉 [Streamlit - Indonesian Stock Sentiment Analysis](https://indonesian-stock-sentiment-analysis-richardtanjaya.streamlit.app/)

## 💡 Fitur
- ✅ **Sentiment Classification** — Mengklasifikasikan teks finansial Indonesia menjadi tiga kategori: Positive, Neutral, Negative.
- ✅ **Single Text Analysis** — Analisis cepat untuk satu teks dengan visualisasi confidence score (bar chart).
- ✅ **Batch Analysis** — Upload file CSV untuk menganalisis ratusan teks sekaligus, hasil ditampilkan dalam tabel, grafik distribusi, serta dapat diunduh kembali dalam format CSV.
- ✅ **Confidence Score Visualization** — Menampilkan grafik batang (bar chart) menggunakan Plotly.
- ✅ **Sentiment Distribution** — Visualisasi pie chart untuk distribusi hasil sentimen dalam analisis batch.
- ✅ **Interactive Sidebar** — Menampilkan detail model, baseline vs fine-tuned performance, pengaturan parameter, dan contoh teks.

## ⚙️ Tech Stack
- **Deep Learning Models** ~ PyTorch, Hugging Face Transformers
- **Arsitektur Model** ~ IndoBERTweet (Fine-Tuned)
- **Web Framework** ~ Streamlit
- **Manipulasi & Visualisasi Data** ~ Pandas, NumPy, Plotly
- **Deployment** ~ Streamlit Cloud, HuggingFace

## 🧠 Model Details
- **Nama Model:** [`RichTan/IndoStockSentiment`](https://huggingface.co/RichTan/IndoStockSentiment)
- **Bahasa:** Indonesia
- **Domain:** Finance & Stock Market
- **Kelas:** Positive, Neutral, Negative
- **Baseline Test Results:**
  - TF-IDF SVM: 73.89%
  - IndoBERTweet: 86.85%
  - Llama 3.2 - 1B: 80.22%
- **Fine-Tuned Test Result:** IndoBERTweet mencapai **87.53%** metric F1-Macro.

## ⭐ Deployment
Aplikasi ini di-deploy menggunakan:
- Streamlit Cloud
- HuggingFace
- GitHub

## 👨‍💻 Pembuat
Richard Dean Tanjaya

## 📝 License
Proyek ini bersifat open-source dan bebas digunakan untuk keperluan edukasi dan penelitian.


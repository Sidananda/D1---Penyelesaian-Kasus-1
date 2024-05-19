import streamlit as st
import joblib
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler



def load_model_and_vectorizer():
    # Memuat model SVM
    svm_model = joblib.load('svm_model.pkl')
    
    # Memuat vektor TF-IDF
    vectorizer = joblib.load('vectorizer.pkl')
    
    return svm_model, vectorizer

def preprocess_text(text):
    # Mengonversi teks menjadi huruf kecil
    text = text.lower()
    
    # Menghapus karakter non-alfanumerik dan angka
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenisasi teks
    tokens = word_tokenize(text)
    
    # Penghapusan stop words
    stop_words = set(stopwords.words('indonesian'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming teks
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Menggabungkan kembali token menjadi kalimat
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def classify_text(text, svm_model, vectorizer):
    # Pra-pemrosesan teks
    preprocessed_text = preprocess_text(text)

    # Mengonversi teks input menjadi vektor fitur menggunakan TF-IDF
    input_vector = vectorizer.transform([preprocessed_text])
    
    # Normalisasi fitur
    scaler = StandardScaler(with_mean=False)
    input_vector_scaled = scaler.fit_transform(input_vector)

    # Melakukan prediksi menggunakan model SVM
    prediction = svm_model.predict(input_vector_scaled)

    # Menampilkan hasil prediksi
    if prediction[0] == 1:
        return "POSITIF"
    else:
        return "NEGATIF"

def main():
    st.title("Analisis Sentimen Pada Teks")
    st.subheader("Kelompok D1")
    st.caption("Anggota Kelompok: \n"
               "1. I Made Prenawa Sida Nanda (2208561017)\n"
               "2. Kadek Yuni Suratri (2208561055)\n"
               "3. Pande Komang Bhargo Anantha Yogiswara (2208561067)\n"
               "4. I Gusti Agung Ayu Gita Pradnyaswari Mantara (2208561105)\n"
               "5. I Kadek Agus Candra Widnyana (2208561129)")
    st.divider()

    # Memuat model dan vektor
    svm_model, vectorizer = load_model_and_vectorizer()

    # Input teks
    st.subheader("Masukkan Kata atau Kalimat")
    text_input = st.text_area("")

    # Upload file
    st.subheader("Atau dengan file")
    uploaded_file = st.file_uploader("Unggah file", type="txt")

    if text_input or uploaded_file is not None:
        if text_input:
            text = text_input
        else:
            text = uploaded_file.read().decode("utf-8")
            st.write("Anda telah berhasil mengunggah file")

        # Memproses teks dan mengklasifikasikannya
        classifications = [classify_text(sentence, svm_model, vectorizer) for sentence in sent_tokenize(text)]

        # Menampilkan hasil klasifikasi
        st.subheader("Hasil Klasifikasi:")
        df_result = pd.DataFrame({"Kalimat": sent_tokenize(text), "Sentimen": classifications})
        st.write(df_result)

        # Menghitung jumlah kalimat positif dan negatif
        num_positif = (df_result["Sentimen"] == "POSITIF").sum()
        num_negatif = (df_result["Sentimen"] == "NEGATIF").sum()

        # Membuat grafik perbandingan akurasi
        data = pd.DataFrame({"Sentimen": ["Positif", "Negatif"], "Jumlah": [num_positif, num_negatif]})
        
        # Membuat grafik perbandingan jumlah kalimat positif dan negatif
        colors = ['blue', 'red']
        fig, ax = plt.subplots()
        ax.bar(data["Sentimen"], data["Jumlah"], color=colors)
        ax.set_ylabel("Jumlah")
        ax.set_title("Perbandingan Jumlah Kalimat Positif dan Negatif")
        st.pyplot(fig)

        # Menyimpan hasil dalam bentuk file
        st.download_button(
            label="Download Hasil Klasifikasi",
            data=df_result.to_csv(index=False).encode(),
            file_name="hasil_klasifikasi.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
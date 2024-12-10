import re
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load Corpus
with open('corpus.txt', 'r') as file:
    loaded_corpus = [line.strip() for line in file]

word_to_index = {word: i for i, word in enumerate(loaded_corpus)}

def pre_process_text(text, word_to_index, max_length=75, padding_type='post', trunc_type='pre'):
    """
    Preprocesses the input text by tokenizing it based on the provided word-to-index dictionary
    and padding/truncating the result to a fixed length.

    Returns:
    - padded_data (numpy.ndarray): The padded sequence.
    """
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    tokenized = [word_to_index.get(word, 0) for word in tokens]
    padded_data = pad_sequences(
        [tokenized], maxlen=max_length, padding=padding_type, truncating=trunc_type
    )
    return padded_data

# Load Models
model_flatten = load_model('model_flatten.h5')
model_conv = load_model('model_convolutional.h5')
model_lstm = load_model('model_lstm.h5')
model_gru = load_model('model_gru.h5')

# Streamlit UI
st.title("Phishing URL Detection")
st.write("Masukkan URL untuk mendeteksi apakah berpotensi phising atau tidak.")

# Input form
new_text = st.text_input("Masukkan URL:")

if st.button("Prediksi"):
    if new_text.strip():
        # Preprocess input
        processed_text = pre_process_text(text=new_text, word_to_index=word_to_index)

        # Predictions
        flatten_pred = model_flatten.predict(processed_text)[0][0]
        conv_pred = model_conv.predict(processed_text)[0][0]
        lstm_pred = model_lstm.predict(processed_text)[0][0]
        gru_pred = model_gru.predict(processed_text)[0][0]

        # Display predictions
        st.write(f"### Hasil Prediksi:")
        st.write(f"- Model **Flatten**: {flatten_pred:.4f}")
        st.write(f"- Model **Convolutional**: {conv_pred:.4f}")
        st.write(f"- Model **LSTM**: {lstm_pred:.4f}")
        st.write(f"- Model **GRU**: {gru_pred:.4f}")

        # Threshold
        threshold = 0.5

        # Display results
        st.subheader("Kesimpulan:")
        st.write("- **Model Flatten**:")
        if flatten_pred > threshold:
            st.error(f"URL terdeteksi sebagai **Phishing** dengan Model Flatten.")
        else:
            st.success(f"URL terdeteksi sebagai **Non-Malicious** dengan Model Flatten.")

        st.write("- **Model Convolutional**:")
        if conv_pred > threshold:
            st.error(f"URL terdeteksi sebagai **Phishing** dengan Model Convolutional.")
        else:
            st.success(f"URL terdeteksi sebagai **Non-Malicious** dengan Model Convolutional.")

        st.write("- **Model LSTM**:")
        if lstm_pred > threshold:
            st.error(f"URL terdeteksi sebagai **Phishing** dengan Model LSTM.")
        else:
            st.success(f"URL terdeteksi sebagai **Non-Malicious** dengan Model LSTM.")

        st.write("- **Model GRU**:")
        if gru_pred > threshold:
            st.error(f"URL terdeteksi sebagai **Phishing** dengan Model GRU.")
        else:
            st.success(f"URL terdeteksi sebagai **Non-Malicious** dengan Model GRU.")
    else:
        st.warning("Mohon masukkan URL terlebih dahulu.")

import re
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

    Parameters:
    - text (str): The input text to be preprocessed.
    - word_to_index (dict): Dictionary mapping words to their respective indices.
    - max_length (int): Maximum length of the padded sequence.
    - padding_type (str): Padding type.
    - trunc_type (str): Truncation type.

    Returns:
    - padded_data (numpy.ndarray): The padded sequence.
    """
    # Lowercase and split text into words
    text = text.lower()
    tokens = re.findall(r'\w+', text)  # Split teks menjadi token

    # Tokenize using word_to_index
    tokenized = [word_to_index.get(word, 0) for word in tokens]  # Gunakan indeks 0 untuk kata yang tidak dikenal

    # Pad the sequence to ensure consistent length
    padded_data = pad_sequences(
        [tokenized], maxlen=max_length, padding=padding_type, truncating=trunc_type
    )

    return padded_data

# Load Model

model_flatten_loaded = load_model('model_flatten.h5')
model_conv_loaded = load_model('model_convolutional.h5')
model_lstm_loaded = load_model('model_lstm.h5')
model_gru_loaded = load_model('model_gru.h5')

new_text = input("Masukkan teks yang ingin diprediksi: ")
processed_text = pre_process_text(text=new_text, word_to_index=word_to_index)

# Melakukan prediksi 
flatten_model_prediction = model_flatten_loaded.predict(processed_text)
convlotuional_model_prediction = model_conv_loaded.predict(processed_text)
lstm_model_prediction = model_lstm_loaded.predict(processed_text)
gru_model_prediction = model_gru_loaded.predict(processed_text)


# Menampilkan hasil prediksi
print(f"Prediksi dengan model Flatten \t\t: {flatten_model_prediction[0][0]:.4f} (probabilitas)")
print(f"Prediksi dengan model Convlotuional \t: {convlotuional_model_prediction[0][0]:.4f} (probabilitas)")
print(f"Prediksi dengan model LSTM \t\t: {lstm_model_prediction[0][0]:.4f} (probabilitas)")
print(f"Prediksi dengan model GRU \t\t: {gru_model_prediction[0][0]:.4f} (probabilitas)")

# Menentukan hasil berdasarkan threshold
threshold = 0.5  


# flattenn
if flatten_model_prediction[0][0] > threshold:
    print(f"Web dengan alamat url {new_text} ini terdeteksi Phising dengan model Flatten")
else:
    print("Hasil: Non-Malicious dengan model Flatten")

# convolutional
if convlotuional_model_prediction[0][0] > threshold:
    print(f"Web dengan alamat url {new_text} ini terdeteksi Phising dengan model COnvolutional")
else:
    print("Hasil: Non-Malicious dengan model Convolutional")

# lstm
if lstm_model_prediction[0][0] > threshold:
    print(f"Web dengan alamat url {new_text} ini terdeteksi Phising dengan model LSTM")
else:
    print("Hasil: Non-Malicious dengan model LSTM")

# gru
if gru_model_prediction[0][0] > threshold:
    print(f"Web dengan alamat url {new_text} ini terdeteksi Phising dengan model GRU")
else:
    print("Hasil: Non-Malicious dengan model GRU")
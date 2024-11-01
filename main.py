# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# CSS for background and font styling
st.markdown(
    """
    <style>
    /* Add a background gradient */
    .main {
        background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%);
        color: white;
        padding: 20px;
    }

    /* Center and style the title */
    .title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #ffffff;
        text-shadow: 2px 2px 8px #000000;
    }

    /* Style the text input area */
    .stTextArea {
        border: 2px solid #feb47b;
        background-color: #fffbf1;
        color: #333;
        font-size: 1.1em;
        padding: 10px;
        border-radius: 8px;
    }

    /* Style the classify button */
    .stButton>button {
        color: white;
        background-color: #ff7e5f;
        border: none;
        border-radius: 12px;
        font-size: 1.2em;
        padding: 8px 20px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #feb47b;
        color: #ffffff;
    }

    /* Style the output result */
    .result {
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        color: #fffbf1;
        margin-top: 20px;
        text-shadow: 1px 1px 6px #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display TensorFlow version
st.write(f"TensorFlow version: {tf.__version__}")

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Function to load the model with error handling
def load_sentiment_model():
    try:
        model = load_model('simple_rnn_imdb (1).h5')
        return model
    except Exception as e:
        st.error("Error loading the model. Please check the file format and compatibility.")
        st.write("Detailed error:", e)
        return None

# Load the model only once and store it in session state
if 'model' not in st.session_state:
    st.session_state.model = load_sentiment_model()

# Helper function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Helper function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app interface
st.markdown('<div class="title">IMDB Movie Review Sentiment Analysis</div>', unsafe_allow_html=True)
st.write("Enter a movie review below to classify it as positive or negative.")

# User input
user_input = st.text_area('Movie Review', placeholder="Type your review here...")

# Classification and Prediction
if st.button('Classify'):
    if st.session_state.model:  # Check if the model loaded successfully
        preprocessed_input = preprocess_text(user_input)
        prediction = st.session_state.model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Display the result
        st.markdown(f'<div class="result">Sentiment: {sentiment}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result">Prediction Score: {prediction[0][0]:.2f}</div>', unsafe_allow_html=True)
    else:
        st.error("Model could not be loaded. Please check for issues with the model file.")
else:
    st.write('Please enter a movie review.')

# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Check TensorFlow version
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

# Load the model only once and store in session state
if 'model' not in st.session_state:
    st.session_state.model = load_sentiment_model()

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app interface
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

# Classification and Prediction
if st.button('Classify'):
    if st.session_state.model:  # Check if the model loaded successfully
        preprocessed_input = preprocess_text(user_input)
        prediction = st.session_state.model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Display the result
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {prediction[0][0]}')
    else:
        st.error("Model could not be loaded. Please check for issues with the model file.")
else:
    st.write('Please enter a movie review.')

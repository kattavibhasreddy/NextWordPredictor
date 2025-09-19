# =================================================================
# Phase 4.1: Predictor Class
# =================================================================
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TextPredictor:
    """A class to encapsulate the trained model and tokenizer for prediction."""

    def __init__(self, model_path, tokenizer_path):
        """
        Loads the trained Keras model and the tokenizer.
        Args:
            model_path (str): Path to the saved .h5 model file.
            tokenizer_path (str): Path to the saved .pkl tokenizer file.
        """
        print("Loading model and tokenizer...")
        try:
            self.model = tf.keras.models.load_model(model_path)
            with open(tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            
            # Infer max_sequence_len from the model's input layer
            self.max_sequence_len = self.model.input_shape[1] + 1
            print("Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            print("Please ensure 'general_model.h5' and 'tokenizer.pkl' exist.")
            exit()

    def predict_next_words(self, text, n_words=3):
        """
        Predicts the top N most likely next words.
        Args:
            text (str): The input text (sequence of words).
            n_words (int): The number of top words to suggest.
        Returns:
            list: A list of the top N predicted words.
        """
        if not text:
            return []
            
        # 1. Preprocess the input text using the loaded tokenizer
        token_list = self.tokenizer.texts_to_sequences([text])[0]
        
        # 2. Pad the sequence to match the model's input length
        padded_sequence = pad_sequences([token_list], maxlen=self.max_sequence_len - 1, padding='pre')
        
        # 3. Predict the probabilities for the next word
        predicted_probabilities = self.model.predict(padded_sequence, verbose=0)[0]
        
        # 4. Get the indices of the top N words using argsort
        # np.argsort returns indices from smallest to largest, so we take the last 'n_words'.
        top_indices = np.argsort(predicted_probabilities)[-n_words:][::-1] # Reverse for descending order
        
        # 5. Convert the indices back to words
        predicted_words = []
        for index in top_indices:
            # The tokenizer's word_index is 1-based, 0 is reserved.
            if index == 0:
                continue
            word = self.tokenizer.index_word.get(index, '')
            if word:
                predicted_words.append(word)
                
        return predicted_words
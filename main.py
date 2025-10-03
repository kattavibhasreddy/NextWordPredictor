# -*- coding: utf-8 -*-
"""
Enhanced Next-Word Prediction Model with Dynamic Vocabulary Adaptation.

This script refactors the original proof-of-concept model into a robust,
reusable WordPredictor class. It addresses the core limitation of the
original model by allowing for the dynamic addition of new words (e.g., slang)
to the vocabulary, followed by a controlled retraining process to adapt the
model to the new lexicon.

This version enhances the neural network architecture by using a stacked
LSTM model (two LSTM layers) to potentially capture more complex patterns
in the text data.

Key Features:
- Object-Oriented Design: Encapsulates all functionality within a single class
  for easy state management, persistence, and usability.
- Dynamic Vocabulary: Implements an `add_slang` method to augment the corpus,
  rebuild the vocabulary, and retrain the model.
- Stacked LSTM Architecture: Utilizes two LSTM layers for improved learning capacity.
- Corrected Inference: Provides a fully functional and logically sound
  `predict_next_words` method that operates correctly at the word level.
- Persistence: Includes `save` and `load` methods to store and retrieve the
  entire predictor state (model, vocabulary, configuration).
- Clear Workflow: A demonstration `if __name__ == "__main__":` block shows the
  complete lifecycle: initial training, prediction, adding new words, and
  re-prediction.
"""

import numpy as np
import pickle
import heapq
import os
from difflib import SequenceMatcher
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

class WordPredictor:
    """
    An LSTM-based next-word prediction model encapsulated in a class.

    This class handles text preprocessing, model building, training,
    prediction, and dynamic vocabulary updates.
    """

    def __init__(self, word_length=5, corpus_path='corpus_file.txt'):
        """
        Initializes the WordPredictor instance.

        Args:
            word_length (int): The number of words in an input sequence.
            corpus_path (str): The path to the initial training text file.
        """
        self.word_length = word_length
        self.corpus_path = corpus_path
        self.model = None
        self.word_to_index = {}
        self.index_to_word = {}
        self.unique_words_count = 0
        self.text_corpus = ""

    def _load_corpus(self):
        """Loads and normalizes the text corpus from the file."""
        print(f"Loading corpus from {self.corpus_path}...")
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            self.text_corpus = f.read().lower()
        print(f"Corpus length: {len(self.text_corpus)} characters.")

    def _build_vocabulary(self):
        """Builds the word-to-index and index-to-word mappings."""
        print("Building vocabulary...")
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(self.text_corpus)
        unique_words = sorted(list(set(words)))
        
        self.unique_words_count = len(unique_words)
        self.word_to_index = {word: i for i, word in enumerate(unique_words)}
        self.index_to_word = {i: word for i, word in enumerate(unique_words)}
        print(f"Vocabulary size: {self.unique_words_count} unique words.")
        return words

    def _prepare_dataset(self, words):
        """
        Prepares the dataset for training by creating sequences and
        one-hot encoding them.
        """
        print("Preparing dataset for training...")
        prev_words_list = []
        next_words_list = []
        for i in range(len(words) - self.word_length):
            prev_words_list.append(words[i:i + self.word_length])
            next_words_list.append(words[i + self.word_length])

        # One-hot encoding
        X = np.zeros((len(prev_words_list), self.word_length, self.unique_words_count), dtype=bool)
        Y = np.zeros((len(next_words_list), self.unique_words_count), dtype=bool)

        for i, each_words in enumerate(prev_words_list):
            for j, each_word in enumerate(each_words):
                if each_word in self.word_to_index:
                    X[i, j, self.word_to_index[each_word]] = 1
            if next_words_list[i] in self.word_to_index:
                Y[i, self.word_to_index[next_words_list[i]]] = 1
        
        print(f"Dataset prepared with X shape: {X.shape} and Y shape: {Y.shape}")
        return X, Y

    def _build_model(self):
        """Builds the Keras stacked LSTM model."""
        print("Building stacked LSTM model...")
        model = Sequential()
        # First LSTM layer. `return_sequences=True` is essential to pass the
        # full sequence output to the next LSTM layer.
        model.add(LSTM(128, input_shape=(self.word_length, self.unique_words_count), return_sequences=True))
        # Second LSTM layer.
        model.add(LSTM(128))
        model.add(Dense(self.unique_words_count))
        model.add(Activation('softmax'))
        self.model = model

    def train(self, epochs=20, batch_size=128):
        """
        Trains the LSTM model. This method orchestrates the entire
        preprocessing and training pipeline.
        """
        self._load_corpus()
        words = self._build_vocabulary()
        X, Y = self._prepare_dataset(words)
        self._build_model()

        optimizer = RMSprop(learning_rate=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        print(f"Starting training for {epochs} epochs...")
        history = self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, shuffle=True)
        print("Training complete.")
        return history

    def add_slang(self, new_text_with_slang, retrain_epochs=20):
        """
        Adds new text (containing slang or new words) to the corpus and
        retrains the model.

        Args:
            new_text_with_slang (str): A string containing sentences with the new words.
            retrain_epochs (int): The number of epochs for the retraining session.
        """
        print("\n--- Dynamic Vocabulary Update Initiated ---")
        print("Adding new text to corpus...")
        # Append new text to the file and reload
        with open(self.corpus_path, 'a', encoding='utf-8') as f:
            f.write("\n" + new_text_with_slang.lower())
        
        print("Retraining model with updated corpus...")
        # The train method will handle reloading the augmented corpus,
        # rebuilding the vocabulary, and training a new model.
        self.train(epochs=retrain_epochs)
        print("--- Dynamic Vocabulary Update Complete ---")

    def _find_closest_word(self, word, threshold=0.6):
        """
        Finds the closest word in the vocabulary using string similarity.
        
        Args:
            word (str): The word to find a match for
            threshold (float): Minimum similarity threshold (0-1)
            
        Returns:
            str or None: The closest matching word, or None if no good match found
        """
        best_match = None
        best_ratio = 0
        
        for vocab_word in self.word_to_index.keys():
            ratio = SequenceMatcher(None, word.lower(), vocab_word.lower()).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = vocab_word
                
        return best_match

    def _prepare_input_for_prediction(self, text, auto_correct=True):
        """
        Prepares a single text sequence for prediction by tokenizing and
        one-hot encoding it.
        
        Args:
            text (str): Input text sequence
            auto_correct (bool): Whether to auto-correct unknown words
        """
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text.lower())
        
        if len(words) != self.word_length:
            raise ValueError(f"Input text must contain exactly {self.word_length} words.")

        x_pred = np.zeros((1, self.word_length, self.unique_words_count))
        corrected_words = []
        
        for t, word in enumerate(words):
            if word in self.word_to_index:
                x_pred[0, t, self.word_to_index[word]] = 1.
                corrected_words.append(word)
            else:
                if auto_correct:
                    # Try to find a close match
                    closest_word = self._find_closest_word(word)
                    if closest_word:
                        x_pred[0, t, self.word_to_index[closest_word]] = 1.
                        corrected_words.append(closest_word)
                        print(f"Auto-corrected '{word}' to '{closest_word}'")
                    else:
                        print(f"Warning: Word '{word}' not in vocabulary and no close match found. Ignoring.")
                        corrected_words.append(word)
                else:
                    print(f"Warning: Word '{word}' not in vocabulary. Ignoring.")
                    corrected_words.append(word)
        
        return x_pred, corrected_words

    @staticmethod
    def _sample(preds, top_n=3):
        """Helper function to sample top N indices from a probability array."""
        preds = np.asarray(preds).astype('float64')
        # Add a small epsilon to prevent log(0)
        preds += 1e-7
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        # Get the indices of the top N predictions
        return heapq.nlargest(top_n, range(len(preds)), key=preds.take)

    def predict_next_words(self, text, top_n=3, auto_correct=True):
        """
        Predicts the top N most likely next words for a given text sequence.

        Args:
            text (str): The input text sequence. Must contain `word_length` words.
            top_n (int): The number of top predictions to return.
            auto_correct (bool): Whether to auto-correct unknown words.

        Returns:
            tuple: (predictions, corrected_input) where predictions is a list of 
                   the top N predicted next words and corrected_input is the 
                   auto-corrected version of the input text.
        """
        if not self.model:
            raise RuntimeError("Model is not trained or loaded. Please train or load a model first.")
        
        try:
            x_pred, corrected_words = self._prepare_input_for_prediction(text, auto_correct)
            preds = self.model.predict(x_pred, verbose=0)
            next_indices = self._sample(preds, top_n)
            predictions = [self.index_to_word[i] for i in next_indices]
            corrected_input = " ".join(corrected_words)
            return predictions, corrected_input
        except ValueError as e:
            print(f"Error during prediction: {e}")
            return None, text

    def save(self, directory='word_predictor_model'):
        """
        Saves the entire WordPredictor state to a directory.
        
        Args:
            directory (str): The directory where the model and state will be saved.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the Keras model
        self.model.save(os.path.join(directory, 'keras_model.h5'))
        
        # Save the rest of the object's state
        state = {
            'word_length': self.word_length,
            'corpus_path': self.corpus_path,
            'word_to_index': self.word_to_index,
            'index_to_word': self.index_to_word,
            'unique_words_count': self.unique_words_count,
            'text_corpus': self.text_corpus
        }
        with open(os.path.join(directory, 'predictor_state.pkl'), 'wb') as f:
            pickle.dump(state, f)
        print(f"Predictor saved to '{directory}'.")

    @classmethod
    def load(cls, directory='word_predictor_model'):
        """
        Loads a WordPredictor instance from a directory.

        Args:
            directory (str): The directory from which to load the model.

        Returns:
            WordPredictor: A loaded instance of the class.
        """
        state_path = os.path.join(directory, 'predictor_state.pkl')
        model_path = os.path.join(directory, 'keras_model.h5')

        if not os.path.exists(state_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find model files in '{directory}'.")

        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        
        # Create a new instance and populate it with the loaded state
        predictor = cls(word_length=state['word_length'], corpus_path=state['corpus_path'])
        predictor.word_to_index = state['word_to_index']
        predictor.index_to_word = state['index_to_word']
        predictor.unique_words_count = state['unique_words_count']
        predictor.text_corpus = state['text_corpus']
        
        # Load the Keras model
        predictor.model = load_model(model_path)
        print(f"Predictor loaded from '{directory}'.")
        return predictor


if __name__ == '__main__':
    # --- Step 1: Create a dummy corpus file ---
    corpus_text = """
    the project gutenberg ebook of the adventures of sherlock holmes
    by sir arthur conan doyle. this ebook is for the use of anyone
    anywhere at no cost and with almost no restrictions whatsoever.
    you may copy it, give it away or re-use it under the terms of
    the project gutenberg license included with this ebook or online
    at www.gutenberg.org.
    """
    corpus_file = 'corpus.txt'
    with open(corpus_file, 'w', encoding='utf-8') as f:
        f.write(corpus_text)

    # --- Step 2: Initial Training ---
    # Instantiate the predictor. Using the larger corpus_file.txt for better training.
    predictor = WordPredictor(word_length=5, corpus_path='corpus_file.txt')
    # Train the model. Use fewer epochs for a quick demonstration.
    predictor.train(epochs=30)
    
    # --- Step 3: Save the trained model ---
    predictor.save('my_word_model')

    # --- Step 4: Test Prediction with the initial model ---
    print("\n--- Testing initial model ---")
    input_text = "the project gutenberg ebook"
    predictions, corrected_input = predictor.predict_next_words(input_text, top_n=3)
    print(f"Input: '{input_text}'")
    print(f"Corrected input: '{corrected_input}'")
    print(f"Predicted next words: {predictions}")
    
    # --- Test Auto-correct functionality ---
    print("\n--- Testing auto-correct functionality ---")
    typo_text = "the projct gutenbrg ebook"  # Contains typos
    predictions_typo, corrected_typo = predictor.predict_next_words(typo_text, top_n=3)
    print(f"Input with typos: '{typo_text}'")
    print(f"Auto-corrected: '{corrected_typo}'")
    print(f"Predicted next words: {predictions_typo}")

    # --- Step 5: Add a new slang word and retrain ---
    # Let's imagine we want to add the slang 'yeet' and some context.
    slang_context = "one does not simply yeet the ebook away."
    predictor.add_slang(slang_context, retrain_epochs=30)

    # --- Step 6: Test Prediction with the updated model ---
    print("\n--- Testing updated model with new vocabulary ---")
    # A new prediction that might now involve the word 'yeet'
    input_text_new = "one does not simply"
    predictions_new, corrected_new = predictor.predict_next_words(input_text_new, top_n=3)
    print(f"Input: '{input_text_new}'")
    print(f"Corrected input: '{corrected_new}'")
    print(f"Predicted next words: {predictions_new}")

    # --- Step 7: Demonstrate loading the model from disk ---
    print("\n--- Demonstrating loading model from disk ---")
    del predictor  # Remove the current instance
    loaded_predictor = WordPredictor.load('my_word_model') # Load the first saved model
    
    # Predictions from the loaded model should match the initial model
    loaded_predictions, loaded_corrected = loaded_predictor.predict_next_words(input_text, top_n=3)
    print(f"Input (loaded model): '{input_text}'")
    print(f"Corrected input (loaded model): '{loaded_corrected}'")
    print(f"Predicted next words (loaded model): {loaded_predictions}")
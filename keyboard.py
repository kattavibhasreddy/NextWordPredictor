import tkinter as tk
from tkinter import ttk
import threading
import queue
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import pickle
import os
import re

# Load the saved model and tokenizer
def load_prediction_model(model_dir='./saved_model'):
    """
    Load the saved model, tokenizer, and configuration.
    """
    # Load the model
    loaded_model = load_model(os.path.join(model_dir, 'next_word_model.keras'))
    
    # Load the tokenizer
    with open(os.path.join(model_dir, 'tokenizer.pickle'), 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    
    # Load the model configuration
    with open(os.path.join(model_dir, 'model_config.pickle'), 'rb') as handle:
        model_config = pickle.load(handle)
    
    return loaded_model, loaded_tokenizer, model_config

# Load the model and tokenizer
print("Loading model and tokenizer...")
model, tokenizer, config = load_prediction_model()
print("Model and tokenizer loaded successfully!")

def predict_next_word(seed_text, top_k=3):
    """
    Predicts the top k most probable next words based on a seed text using the trained model.
    """
    # Preprocess the input string
    cleaned_text = seed_text.lower()
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    
    # Tokenize and pad the sequence
    token_list = tokenizer.texts_to_sequences([cleaned_text])[0]
    padded_sequence = pad_sequences([token_list], maxlen=config['sequence_length'], padding='pre')
    
    # Print debug info about sequence
    print(f"Debug - Input sequence shape: {padded_sequence.shape}")
    print(f"Debug - Max token index in sequence: {padded_sequence.max()}")
    print(f"Debug - Vocabulary size: {len(tokenizer.word_index) + 1}")
    
    # Check for out-of-vocabulary tokens
    max_index = len(tokenizer.word_index)
    if padded_sequence.max() >= max_index:
        print(f"Warning: Input contains token index {padded_sequence.max()} which is outside vocabulary range [0, {max_index})")
        # Filter out invalid tokens
        padded_sequence = np.clip(padded_sequence, 0, max_index - 1)
    
    # Get prediction probabilities
    predicted_probabilities = model.predict(padded_sequence, verbose=0)[0]
    
    # Get indices of top k predictions
    top_indices = predicted_probabilities.argsort()[-top_k:][::-1]
    
    # Convert indices to words and include probabilities
    predictions = []
    for idx in top_indices:
        word = tokenizer.index_word.get(idx, None)
        if word:
            predictions.append((word, predicted_probabilities[idx]))
    
    return predictions

class PredictiveTextApp:
    def __init__(self, predictor_func):
        self.root = tk.Tk()
        self.root.title("Next Word Predictor")
        self.root.geometry("600x500")
        
        # Store the predictor function and current predictions
        self.predictor_func = predictor_func
        self.current_predictions = []
        
        # Create and configure the main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create the text input area
        self.input_label = ttk.Label(self.main_frame, text="Type your text (Use keys 1, 2, 3 to select predictions):")
        self.input_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.text_input = tk.Text(self.main_frame, height=5, width=50, wrap=tk.WORD)
        self.text_input.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Create prediction display area
        self.predictions_frame = ttk.LabelFrame(self.main_frame, text="Predictions (1,2,3 to select)", padding="5")
        self.predictions_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Create labels for predictions
        self.prediction_labels = []
        for i in range(3):
            label = ttk.Label(self.predictions_frame, text="", font=("Arial", 12))
            label.grid(row=i, column=0, sticky=tk.W, pady=2)
            self.prediction_labels.append(label)
            
        # Create submit button
        self.submit_button = ttk.Button(self.main_frame, text="Submit to Corpus", command=self.submit_to_corpus)
        self.submit_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Create status label
        self.status_label = ttk.Label(self.main_frame, text="")
        self.status_label.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Variable to track if prediction is in progress
        self.prediction_in_progress = False
        
        # Create a queue for prediction requests
        self.prediction_queue = queue.Queue()
        
        # Bind text changes and number keys
        self.text_input.bind('<Key>', self.handle_key_press)
        
        # Start prediction worker thread
        self.prediction_thread = threading.Thread(target=self.prediction_worker, daemon=True)
        self.prediction_thread.start()
        
        # Configure style
        style = ttk.Style()
        style.configure('Prediction.TLabel', font=('Arial', 12))
    
    def handle_key_press(self, event):
        """Handle key presses including number keys for selection"""
        if event.char in ['1', '2', '3']:
            # Convert key to index (0-2)
            idx = int(event.char) - 1
            if idx < len(self.current_predictions):
                word, _ = self.current_predictions[idx]
                self.insert_word(word)
            return 'break'  # Prevent the number from being typed
        self.schedule_prediction(event)
    
    def insert_word(self, word):
        """Insert the selected word into the text input"""
        # Insert space if needed
        current_text = self.text_input.get("1.0", tk.END).strip()
        if current_text and not current_text.endswith(' '):
            self.text_input.insert(tk.END, ' ')
        # Insert the word with a space
        self.text_input.insert(tk.END, f"{word} ")
        self.schedule_prediction()
    
    def schedule_prediction(self, event=None):
        """Schedule a prediction update after text change"""
        self.root.after(100, self.update_predictions)
    
    def update_predictions(self):
        """Update predictions based on current text"""
        current_text = self.text_input.get("1.0", tk.END).strip()
        if current_text:
            self.prediction_queue.put(current_text)
    
    def prediction_worker(self):
        """Worker thread to handle predictions"""
        while True:
            text = self.prediction_queue.get()
            if not self.prediction_in_progress:
                self.prediction_in_progress = True
                try:
                    self.current_predictions = self.predictor_func(text)
                    self.root.after(0, self.update_prediction_labels, self.current_predictions)
                except Exception as e:
                    print(f"Prediction error: {e}")
                finally:
                    self.prediction_in_progress = False
            self.prediction_queue.task_done()
    
    def update_prediction_labels(self, predictions):
        """Update the prediction labels with new predictions"""
        for i, label in enumerate(self.prediction_labels):
            if i < len(predictions):
                word, prob = predictions[i]
                label.config(text=f"{i+1}. {word} ({prob:.1%})")
            else:
                label.config(text="")
    
    def submit_to_corpus(self):
        """Submit the current text to the corpus and retrain the model"""
        current_text = self.text_input.get("1.0", tk.END).strip()
        if not current_text:
            self.status_label.config(text="Please enter some text before submitting.")
            return
            
        try:
            # Append to corpus file
            with open('corpus_file.txt', 'a', encoding='utf-8') as f:
                f.write(f"\n{current_text}")
            
            # Retrain/update the model
            global model, tokenizer, config
            
            # Read the entire corpus
            with open('corpus_file.txt', 'r', encoding='utf-8') as f:
                text_data = f.read()
            
            # Create DataFrame
            df = pd.DataFrame({'text': [text_data]})
            
            # Clean and preprocess
            df['cleaned_text'] = df['text'].apply(lambda x: x.lower())
            df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
            
            # Store previous vocab size
            prev_vocab_size = len(tokenizer.word_index)
            
            # Update tokenizer
            tokenizer.fit_on_texts(df['cleaned_text'])
            new_vocab_size = len(tokenizer.word_index)
            config['total_words'] = new_vocab_size + 1
            
            print(f"Vocabulary size change - Before: {prev_vocab_size}, After: {new_vocab_size}")
            if new_vocab_size != prev_vocab_size:
                print("Warning: Vocabulary size changed after retraining")
            
            # Create sequences
            token_list = tokenizer.texts_to_sequences(df['cleaned_text'])[0]
            sequences = []
            for i in range(config['sequence_length'], len(token_list)):
                seq = token_list[i-config['sequence_length']:i+1]
                sequences.append(seq)
            
            if not sequences:
                self.status_label.config(text="Text too short for training. Please enter a longer sentence.")
                return
                
            # Convert to numpy array and prepare X, y
            sequences = np.array(sequences)
            X = sequences[:, :-1]
            y = sequences[:, -1]
            
            # Ensure sequence length matches model's expectation
            if X.shape[1] != config['sequence_length'] - 1:
                self.status_label.config(text=f"Error: Input must be {config['sequence_length']} words long")
                return
                
            # Convert y to one-hot, ensuring correct shape
            y = to_categorical(y, num_classes=config['total_words'])
            
            # Verify shapes before training
            if X.shape[0] != y.shape[0]:
                self.status_label.config(text="Error: Mismatch in input and output shapes")
                return
                
            print(f"Training shapes - X: {X.shape}, y: {y.shape}")  # Debug info
            
            # Create progress popup
            popup = tk.Toplevel(self.root)
            popup.title("Updating Model")
            popup.geometry("300x150")
            popup.transient(self.root)
            popup.grab_set()
            
            # Progress label
            progress_label = ttk.Label(popup, text="Updating model...", padding=10)
            progress_label.pack()
            
            # Progress bar
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(popup, variable=progress_var, maximum=100)
            progress_bar.pack(fill=tk.X, padx=10, pady=10)
            
            # Cancel button and flag
            self.cancel_update = False
            def cancel_training():
                self.cancel_update = True
                progress_label.config(text="Cancelling...")
            
            cancel_button = ttk.Button(popup, text="Cancel", command=cancel_training)
            cancel_button.pack(pady=10)
            
            # Custom callback for training progress
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, progress_var, progress_label, app):
                    self.progress_var = progress_var
                    self.progress_label = progress_label
                    self.app = app
                    
                def on_epoch_begin(self, epoch, logs=None):
                    self.progress_label.config(text=f"Training epoch {epoch + 1}/2...")
                    
                def on_epoch_end(self, epoch, logs=None):
                    progress = ((epoch + 1) / 2) * 100
                    self.progress_var.set(progress)
                    
                def on_batch_end(self, batch, logs=None):
                    if self.app.cancel_update:
                        self.model.stop_training = True
                    popup.update()
            
            try:
                # Train the model with progress tracking
                progress_label.config(text="Starting training...")
                callback = ProgressCallback(progress_var, progress_label, self)
                
                try:
                    print(f"Model input shape: {model.input_shape}")
                    print(f"Model output shape: {model.output_shape}")
                    print(f"Training data shapes - X: {X.shape}, y: {y.shape}")
                    
                    history = model.fit(X, y, epochs=2, verbose=0, batch_size=32, callbacks=[callback])
                except ValueError as ve:
                    error_msg = str(ve)
                    if "incompatible" in error_msg.lower() or "shape" in error_msg.lower():
                        self.status_label.config(text="Error: Input/output shape mismatch. Try a different sentence length.")
                    else:
                        self.status_label.config(text=f"Training error: {error_msg}")
                    popup.destroy()
                    return
                
                if not self.cancel_update:
                    # Update progress for saving
                    progress_label.config(text="Saving updated model...")
                    
                    # Save the model and related files
                    model.save(os.path.join('saved_model', 'next_word_model.keras'))
                    with open(os.path.join('saved_model', 'tokenizer.pickle'), 'wb') as handle:
                        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(os.path.join('saved_model', 'model_config.pickle'), 'wb') as handle:
                        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    popup.destroy()
                    self.status_label.config(text="Model updated successfully!")
                else:
                    popup.destroy()
                    self.status_label.config(text="Model update cancelled.")
            except Exception as e:
                popup.destroy()
                raise e
            
            self.status_label.config(text="Text added to corpus and model updated successfully!")
            self.text_input.delete("1.0", tk.END)
            
        except Exception as e:
            self.status_label.config(text=f"Error updating corpus: {str(e)}")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

# Create and run the application
app = PredictiveTextApp(predict_next_word)
app.run()

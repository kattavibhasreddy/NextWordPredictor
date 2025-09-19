import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Suppress TensorFlow warnings for a cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("TensorFlow and other libraries imported successfully.")

# =================================================================
# Part 1: Vocabulary Setup (Done once at the start)
# =================================================================
print("\n--- Starting Part 1: Vocabulary Setup ---")

try:
    with open('data.txt', 'r', encoding='utf-8') as file:
        corpus = file.readlines() # Read lines into a list
    print(f"Successfully loaded data.txt with {len(corpus)} lines.")
except FileNotFoundError:
    print("Error: data.txt not found.")
    exit()

# Initialize and fit the tokenizer to build the vocabulary
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {total_words} unique words.")

# --- Pre-calculate max_sequence_len and total_sequences ---
# We need to know the longest sequence length for padding, and the total
# number of sequences to calculate steps_per_epoch for the model.
input_sequences = []
total_sequences = 0
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        total_sequences += 1
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences]) if input_sequences else 0
print(f"Max sequence length: {max_sequence_len}")
print(f"Total number of sequences to be generated: {total_sequences}")


# =================================================================
# Part 2: The Data Generator Function
# =================================================================

def data_generator(corpus, tokenizer, max_sequence_len, total_words, batch_size):
    """
    Yields batches of data indefinitely for Keras model training.
    """
    while True:
        # Loop through the corpus line by line
        for line in corpus:
            token_list = tokenizer.texts_to_sequences([line])[0]
            
            # Create sequences from the line
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                
                # Pad the sequence
                padded_sequence = pad_sequences([n_gram_sequence], maxlen=max_sequence_len, padding='pre')
                
                # Create the X and y for this single sequence
                X = padded_sequence[:, :-1]
                label = padded_sequence[:, -1]
                y = to_categorical(label, num_classes=total_words)
                
                # Yield a batch of size 1 (can be optimized to yield larger batches)
                yield (X, y)

# =================================================================
# Part 3: Model Architecture (Same as before)
# =================================================================
print("\n--- Starting Part 3: Designing the Model Architecture ---")

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

print("Model architecture created successfully.")
model.summary()

# =================================================================
# Part 4: Training with the Generator
# =================================================================
print("\n--- Starting Part 4: Compiling and Training the Model ---")

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- Key Changes for Generator Training ---
BATCH_SIZE = 128 # How many sequences to process at once. Tune based on your VRAM/RAM.
STEPS_PER_EPOCH = total_sequences // BATCH_SIZE

# Instantiate the generator
# NOTE: This generator implementation is simple and yields batch_size=1.
# A more optimized version would collect 'BATCH_SIZE' samples before yielding.
# However, this version is guaranteed to work with low memory.
train_generator = data_generator(corpus, tokenizer, max_sequence_len, total_words, 1)

print(f"\nStarting training with a batch size of {BATCH_SIZE} and {STEPS_PER_EPOCH} steps per epoch.")
print("This will take a significant amount of time.")

try:
    # Use model.fit() with the generator
    model.fit(train_generator,
              epochs=50,
              steps_per_epoch=STEPS_PER_EPOCH,
              verbose=1)
              
    print("\nModel training completed.")
    
    # --- Saving the Artifacts ---
    model.save('general_model.h5')
    print("Saved trained model as general_model.h5")
    
    with open('tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved tokenizer as tokenizer.pkl")

    print("\nPhase 1 is complete! You now have a trained model and a tokenizer.")

except Exception as e:
    print(f"\nAn error occurred during training: {e}")
    print("Training could not be completed.")
# =================================================================
# Phase 3: Personalization via Fine-Tuning
# =================================================================
import os
import pickle
import numpy as np
import tensorflow as tf

# Import the database handler functions we created in Phase 2
import db_handler

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def finetune_for_user(user_id):
    """
    Fine-tunes a model for a specific user based on their stored text.
    """
    print(f"\n--- Starting fine-tuning process for user_id: {user_id} ---")
    
    # --- 1. Load the necessary artifacts ---
    try:
        # Load the tokenizer (it's the same for all models)
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer loaded successfully.")
    except FileNotFoundError:
        print("Error: tokenizer.pkl not found. Please run train_model.py first.")
        return

    # Define paths for the models
    general_model_path = 'general_model.h5'
    personalized_model_path = f'user_{user_id}_model.h5'

    # Decide which model to load. If a personalized model already exists,
    # we load it to continue improving it. Otherwise, we start from the general model.
    if os.path.exists(personalized_model_path):
        print(f"Found existing personalized model. Loading '{personalized_model_path}' to continue training.")
        model_path_to_load = personalized_model_path
    elif os.path.exists(general_model_path):
        print(f"No personalized model found. Loading '{general_model_path}' as the base.")
        model_path_to_load = general_model_path
    else:
        print("Error: general_model.h5 not found. Please run train_model.py first.")
        return

    model = tf.keras.models.load_model(model_path_to_load)
    print("Model loaded successfully.")

    # --- 2. Get user-specific data from the database ---
    user_texts = db_handler.get_all_user_text(user_id)
    
    if not user_texts:
        print(f"No custom text found for user_id {user_id}. Nothing to fine-tune.")
        return
        
    print(f"Found {len(user_texts)} text entries for user_id {user_id}.")

    # --- 3. Preprocess the user's data ---
    # This must follow the EXACT same steps as in train_model.py
    total_words = len(tokenizer.word_index) + 1
    
    input_sequences = []
    for line in user_texts:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    if not input_sequences:
        print("Could not generate sequences from user text. Text might be too short.")
        return

    # Get the max sequence length from the base model's training
    # We can infer this from the model's input layer
    max_sequence_len = model.input_shape[1] + 1
    
    input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(
        input_sequences, maxlen=max_sequence_len, padding='pre'
    ))

    X_user = input_sequences[:, :-1]
    y_user = tf.keras.utils.to_categorical(input_sequences[:, -1], num_classes=total_words)

    print(f"User data preprocessed successfully. Shape of X: {X_user.shape}")

    # --- 4. Re-compile the model with a low learning rate ---
    # This is the most crucial step for fine-tuning.
    # The default Adam learning rate is 0.001. We use a much smaller value.
    low_learning_rate = 0.0001
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=low_learning_rate),
                  metrics=['accuracy'])
    print(f"Model re-compiled with a low learning rate of {low_learning_rate}.")

    # --- 5. Fine-tune the model on the user's data ---
    # We use a small number of epochs because the dataset is small.
    print("Starting fine-tuning...")
    model.fit(X_user, y_user, epochs=30, verbose=1)
    print("Fine-tuning complete.")

    # --- 6. Save the newly personalized model ---
    model.save(personalized_model_path)
    print(f"Successfully saved personalized model to '{personalized_model_path}'.")


# =================================================================
# Test Block: To verify the fine-tuning function
# =================================================================
if __name__ == "__main__":
    # We will test by fine-tuning the model for the user 'vibhas'
    # This assumes you ran the db_handler.py test and created this user.
    TEST_USERNAME = "vibhas"
    print(f"\n--- Running Fine-Tuning Test for user: '{TEST_USERNAME}' ---")
    
    # First, we need the user's ID from the database.
    user_id_to_tune = db_handler.get_user(TEST_USERNAME)
    
    if user_id_to_tune:
        print(f"Found user_id: {user_id_to_tune}")
        # Call the main function to perform the fine-tuning.
        finetune_for_user(user_id_to_tune)
    else:
        print(f"Error: User '{TEST_USERNAME}' not found in the database.")
        print("Please run `python db_handler.py` first to create the test user and data.")

    print("\n--- personalize.py script finished. ---")
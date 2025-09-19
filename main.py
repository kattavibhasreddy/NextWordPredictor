# =================================================================
# Phase 4 (Advanced): Real-Time, Event-Driven CLI
# =================================================================
import os
import sys
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML

# Import our custom modules
import db_handler
from predictor import TextPredictor
from personalize import finetune_for_user

# --- Application State ---
# These variables will hold the state of our application and will be
# modified by the key binding handlers.
generated_sentence = []
current_word = ""
suggestions = []
user_info = {"id": None, "name": None}
predictor_instance = None

# --- UI Layout Functions ---
# These functions define what text appears in different parts of our UI.

def get_top_toolbar_text():
    """ Renders the text for the top part of the UI (the completed sentence). """
    sentence_so_far = ' '.join(generated_sentence)
    return HTML(f"<grey>Completed Sentence:</grey> <green>{sentence_so_far}</green> ")

def get_bottom_toolbar_text():
    """ Renders the text for the bottom part of the UI (live typing and suggestions). """
    suggestion_html = ""
    for i, word in enumerate(suggestions):
        suggestion_html += f"<b><skyblue>[{i+1}: {word}]</skyblue></b> "
    
    # The '>' prompt, the word being typed, and the suggestions
    return HTML(f"<yellow>></yellow> {current_word}  {suggestion_html}")


# --- Key Binding Handlers ---
# This is the core logic of our application. We define what happens on each keypress.
kb = KeyBindings()

@kb.add('<any>')
def _(event):
    """ Handles any regular character keypress. """
    global current_word
    current_word += event.data
    event.app.invalidate() # Redraw the screen

@kb.add('c-h') # Backspace
def _(event):
    """ Handles the backspace key. """
    global current_word
    current_word = current_word[:-1]
    event.app.invalidate()

@kb.add(' ') # Space bar
def _(event):
    """ Handles the space bar, which is our primary prediction trigger. """
    global generated_sentence, current_word, suggestions

    word_to_process = current_word.strip()
    if not word_to_process: # Ignore empty spaces
        current_word = ""
        event.app.invalidate()
        return

    # --- IMPLICIT SLANG DETECTION ---
    # If suggestions were active and the user typed a new word instead of choosing,
    # it means they are using their own custom word.
    if suggestions:
        # The word they ignored was the last one in the sentence
        full_context = ' '.join(generated_sentence)
        db_handler.add_user_text(user_info["id"], full_context)
        print(f"\n[Learning Mode] Saved your custom phrase: '{full_context}'") # A little feedback

    generated_sentence.append(word_to_process)
    context_for_prediction = ' '.join(generated_sentence)
    suggestions = predictor_instance.predict_next_words(context_for_prediction, 3)
    current_word = ""
    event.app.invalidate()

# Handlers for selecting a suggestion
@kb.add('1')
@kb.add('2')
@kb.add('3')
def _(event):
    """ Handles the user selecting a suggestion with keys 1, 2, or 3. """
    global generated_sentence, current_word, suggestions
    
    if suggestions:
        choice_index = int(event.data) - 1
        if choice_index < len(suggestions):
            generated_sentence.append(suggestions[choice_index])
            # Reset for the next word
            current_word = ""
            suggestions = []
            event.app.invalidate()

@kb.add('c-c')
@kb.add('c-d')
def _(event):
    """ Handles exiting the application with Ctrl+C or Ctrl+D. """
    event.app.exit()

# --- Main Application Setup ---
def run_interactive_session():
    """ Sets up the layout and runs the main application loop. """
    
    # Define the layout using our functions
    layout = Layout(
        container=HSplit([
            Window(content=FormattedTextControl(get_top_toolbar_text), height=1, style='class:top-toolbar'),
            Window(char='-'), # A separator line
            Window(content=FormattedTextControl(get_bottom_toolbar_text), height=1, style='class:bottom-toolbar')
        ])
    )
    
    # Create and run the Application object
    app = Application(layout=layout, key_bindings=kb, full_screen=True)
    print(f"--- Welcome, {user_info['name']}! Starting interactive session. ---")
    print("Start typing. Press Space to predict, 1/2/3 to select, Ctrl+C to exit.")
    app.run()

# --- Login and Initialization ---
def main():
    global user_info, predictor_instance
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("--- Welcome to the Real-Time Next-Word Predictor ---")
    
    db_handler.setup_database()

    # Simple login/register (could be replaced by the prompt_toolkit version from main.py)
    while True:
        choice = input("Enter '1' to Login, '2' to Register: ").strip()
        if choice in ['1', '2']:
            username = input("Username: ").strip()
            if choice == '1':
                user_id = db_handler.get_user(username)
                if user_id:
                    user_info = {"id": user_id, "name": username}
                    break
                else: print("User not found.")
            elif choice == '2':
                if db_handler.get_user(username):
                    print("Username already exists.")
                else:
                    db_handler.add_user(username)
                    user_id = db_handler.get_user(username)
                    user_info = {"id": user_id, "name": username}
                    break
        else: print("Invalid choice.")
        
    # Load the appropriate model
    personalized_model_path = f'user_{user_info["id"]}_model.h5'
    model_to_load = personalized_model_path if os.path.exists(personalized_model_path) else 'general_model.h5'
    print(f"Loading model: {model_to_load}")
    
    predictor_instance = TextPredictor(model_path=model_to_load, tokenizer_path='tokenizer.pkl')
    
    # Start the main application
    run_interactive_session()


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Gradio Interface for Next-Word Prediction with Auto-correct and Dynamic Vocabulary Learning.

This interface provides a mobile keyboard-like experience with:
- Real-time next word prediction
- Auto-correct functionality
- Dynamic vocabulary learning from user corrections
- Context-aware learning when users override auto-corrections
"""

import gradio as gr
import re
from main import WordPredictor
import os

class SmartKeyboard:
    def __init__(self, model_path='my_word_model'):
        """
        Initialize the Smart Keyboard with a trained WordPredictor model.
        
        Args:
            model_path (str): Path to the saved model directory
        """
        self.predictor = None
        self.last_input = ""
        self.last_predictions = []
        self.user_corrections = {}  # Track user corrections for learning
        self.current_word = ""
        self.word_context = []  # Store context for vocabulary learning
        
        # Try to load existing model, otherwise create a new one
        try:
            self.predictor = WordPredictor.load(model_path)
            print(f"Loaded existing model from {model_path}")
        except:
            print("No existing model found. Please train a model first by running main.py")
            self.predictor = None
    
    def get_current_word_and_context(self, text):
        """
        Extract the current word being typed and its context.
        
        Args:
            text (str): Full input text
            
        Returns:
            tuple: (current_word, context_words)
        """
        # Split text into words
        words = text.split()
        
        if not words:
            return "", []
        
        # Get the last word (might be incomplete)
        current_word = words[-1]
        
        # Get context (previous words for prediction)
        context_words = words[:-1] if len(words) > 1 else []
        
        return current_word, context_words
    
    def predict_next_words(self, text):
        """
        Predict next words based on the current context.
        
        Args:
            text (str): Input text
            
        Returns:
            list: Top 3 predicted next words
        """
        if not self.predictor:
            return ["Model not loaded"]
        
        current_word, context_words = self.get_current_word_and_context(text)
        
        # Need at least word_length words for prediction
        if len(context_words) < self.predictor.word_length:
            return ["Keep typing..."]
        
        # Get the last word_length words for prediction
        context_text = " ".join(context_words[-self.predictor.word_length:])
        
        try:
            predictions, corrected = self.predictor.predict_next_words(context_text, top_n=3)
            return predictions if predictions else ["No predictions"]
        except Exception as e:
            return [f"Error: {str(e)}"]
    
    def auto_correct_word(self, text):
        """
        Auto-correct the current word being typed.
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (corrected_text, was_corrected)
        """
        if not self.predictor:
            return text, False
        
        current_word, context_words = self.get_current_word_and_context(text)
        
        if not current_word:
            return text, False
        
        # Check if word needs correction
        if current_word.lower() in self.predictor.word_to_index:
            return text, False  # Word is already correct
        
        # Find closest match
        closest_word = self.predictor._find_closest_word(current_word, threshold=0.6)
        
        if closest_word:
            # Replace the last word with the corrected version
            corrected_text = " ".join(context_words + [closest_word])
            return corrected_text, True
        
        return text, False
    
    def learn_from_correction(self, original_text, corrected_text):
        """
        Learn from user corrections by adding new words to vocabulary.
        
        Args:
            original_text (str): Original text with user's word
            corrected_text (str): Auto-corrected text
        """
        if not self.predictor:
            return
        
        original_words = original_text.split()
        corrected_words = corrected_text.split()
        
        if len(original_words) != len(corrected_words):
            return
        
        # Find words that were changed
        for orig, corr in zip(original_words, corrected_words):
            if orig.lower() != corr.lower():
                # User corrected this word - add original to vocabulary
                context_sentence = " ".join(original_words)
                print(f"Learning new word: '{orig}' from context: '{context_sentence}'")
                
                # Add to corpus and retrain
                self.predictor.add_slang(context_sentence, retrain_epochs=5)
                break
    
    def process_input(self, text):
        """
        Process user input and return predictions with auto-correction.
        
        Args:
            text (str): Current input text
            
        Returns:
            tuple: (predictions, corrected_text, show_corrections)
        """
        if not self.predictor:
            return ["Model not loaded", "", ""], text, False
        
        current_word, context_words = self.get_current_word_and_context(text)
        
        # Auto-correct current word if it's complete (ends with space)
        if text.endswith(' ') and current_word:
            corrected_text, was_corrected = self.auto_correct_word(text)
            if was_corrected:
                # Store correction for potential learning
                self.user_corrections[text] = corrected_text
                predictions = self.predict_next_words(corrected_text)
                return predictions, corrected_text, True
            else:
                predictions = self.predict_next_words(text)
                return predictions, text, False
        
        # Regular typing - just predict
        predictions = self.predict_next_words(text)
        return predictions, text, False
    
    def save_model(self, path="word_predictor_model"):
        """Save the current model state."""
        if self.predictor:
            self.predictor.save(path)
            return f"Model saved to {path}"
        return "No model to save"

# Global instance
smart_keyboard = SmartKeyboard()

def create_interface():
    """Create and configure the Gradio interface."""
    
    def update_predictions(text):
        """Update predictions as user types."""
        predictions, corrected_text, show_correction = smart_keyboard.process_input(text)
        
        # Ensure we always have 3 predictions
        while len(predictions) < 3:
            predictions.append("")
        
        return (
            gr.update(value=corrected_text if show_correction else text),  # Text input
            gr.update(value=predictions[0]),  # Prediction 1
            gr.update(value=predictions[1]),  # Prediction 2
            gr.update(value=predictions[2]),  # Prediction 3
            f"Auto-corrected!" if show_correction else f"Predictions: {', '.join([p for p in predictions if p])}"
        )
    
    def select_prediction_1(text):
        """Handle prediction 1 selection (key '1')."""
        predictions, corrected_text, show_correction = smart_keyboard.process_input(text)
        if predictions and len(predictions) > 0 and predictions[0]:
            new_text = text + " " + predictions[0]
            return update_predictions(new_text)
        return update_predictions(text)
    
    def select_prediction_2(text):
        """Handle prediction 2 selection (key '2')."""
        predictions, corrected_text, show_correction = smart_keyboard.process_input(text)
        if predictions and len(predictions) > 1 and predictions[1]:
            new_text = text + " " + predictions[1]
            return update_predictions(new_text)
        return update_predictions(text)
    
    def select_prediction_3(text):
        """Handle prediction 3 selection (key '3')."""
        predictions, corrected_text, show_correction = smart_keyboard.process_input(text)
        if predictions and len(predictions) > 2 and predictions[2]:
            new_text = text + " " + predictions[2]
            return update_predictions(new_text)
        return update_predictions(text)
    
    def save_model():
        """Save the current model."""
        result = smart_keyboard.save_model()
        return result
    
    # Create the interface
    with gr.Blocks(title="Mobile Keyboard - AI Prediction", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 📱 Mobile Keyboard with AI Prediction")
        gr.Markdown("Type naturally - predictions update with every keystroke! Use **1**, **2**, **3** keys to select predictions.")
        
        with gr.Column():
            # Main text input
            text_input = gr.Textbox(
                label="Type your message",
                placeholder="Start typing...",
                lines=4,
                max_lines=15,
                interactive=True,
                show_label=False
            )
            
            # Status display
            status_text = gr.Textbox(
                label="Status", 
                interactive=False,
                show_label=False,
                lines=1
            )
            
            # Prediction buttons in a row
            with gr.Row():
                pred_1 = gr.Button(
                    "1. Keep typing...", 
                    variant="secondary",
                    size="lg"
                )
                pred_2 = gr.Button(
                    "2. Keep typing...", 
                    variant="secondary",
                    size="lg"
                )
                pred_3 = gr.Button(
                    "3. Keep typing...", 
                    variant="secondary",
                    size="lg"
                )
            
            # Save model button
            with gr.Row():
                save_btn = gr.Button("💾 Save Model", variant="primary")
                save_status = gr.Textbox(label="Save Status", interactive=False, show_label=False)
        
        # Event handlers
        text_input.change(
            fn=update_predictions,
            inputs=[text_input],
            outputs=[text_input, pred_1, pred_2, pred_3, status_text]
        )
        
        pred_1.click(
            fn=select_prediction_1,
            inputs=[text_input],
            outputs=[text_input, pred_1, pred_2, pred_3, status_text]
        )
        
        pred_2.click(
            fn=select_prediction_2,
            inputs=[text_input],
            outputs=[text_input, pred_1, pred_2, pred_3, status_text]
        )
        
        pred_3.click(
            fn=select_prediction_3,
            inputs=[text_input],
            outputs=[text_input, pred_1, pred_2, pred_3, status_text]
        )
        
        save_btn.click(
            fn=save_model,
            outputs=[save_status]
        )
        
        # Instructions
        gr.Markdown("""
        ## 📱 How to Use:
        1. **Type naturally** - Predictions appear as you type
        2. **Press 1, 2, or 3** - Select from top 3 predictions
        3. **Auto-correct** - Happens automatically on space
        4. **Learn slang** - Type unrecognized words, they get added to vocabulary
        
        ## 🧠 Features:
        - **Real-time predictions** with every keystroke
        - **Auto-correct** for spelling mistakes
        - **Dynamic learning** from your typing patterns
        - **Context-aware** vocabulary expansion
        - **Mobile-like experience** with number key selection
        """)
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        show_error=True
    )

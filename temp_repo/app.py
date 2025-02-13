import gradio as gr
import torch
import logging
import sys
from pathlib import Path
import os

# Add the current directory to Python path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from training.transformer import CustomTransformer
from training.tokenizer import SimpleTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ChatBot:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            try:
                logger.info("Initializing ChatBot...")
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Using device: {self.device}")
                
                # Load model
                model_path = Path("model")  # Path relative to app.py in Space
                logger.info(f"Loading model from {model_path}")
                self.model = CustomTransformer.from_pretrained(str(model_path))
                self.model.to(self.device)
                self.model.eval()
                logger.info("Model loaded successfully")
                
                # Load tokenizer
                logger.info("Loading tokenizer...")
                self.tokenizer = SimpleTokenizer.from_pretrained(str(model_path))
                logger.info("Tokenizer loaded successfully")
                
                self._initialized = True
                logger.info("ChatBot initialization complete")
                
            except Exception as e:
                logger.error(f"Error initializing ChatBot: {str(e)}")
                raise

    def generate_response(self, message: str) -> str:
        try:
            # Tokenize input
            input_ids = self.tokenizer.encode(message)
            input_tensor = torch.tensor([input_ids]).to(self.device)
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_tensor,
                    max_length=100,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9
                )
            
            # Decode response
            response = self.tokenizer.decode(output_ids[0].tolist())
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

# Initialize chatbot once
chatbot = ChatBot()

def predict(message: str) -> str:
    return chatbot.generate_response(message)

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Message", placeholder="Type your message here..."),
    outputs=gr.Textbox(label="Response"),
    title="Changelog LLM Chatbot",
    description="A custom transformer model trained on Wikipedia data",
    examples=[
        ["Tell me about basic physics concepts"],
        ["Explain how simple machines work"],
        ["What are some common English words?"],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=False)

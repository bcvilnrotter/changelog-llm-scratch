import os
import sys
import gradio as gr
import torch
import logging
from pathlib import Path
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add the parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    logger.info(f"Added {parent_dir} to Python path")

# Import custom modules
try:
    from training.transformer import CustomTransformer
    from training.tokenizer import SimpleTokenizer
    logger.info("Successfully imported custom modules")
except ImportError as e:
    logger.error(f"Failed to import custom modules: {str(e)}")
    logger.error(f"Python path: {sys.path}")
    raise

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
                
                # List all files in current directory for debugging
                cwd = os.getcwd()
                logger.info(f"Current working directory: {cwd}")
                logger.info(f"Directory contents: {os.listdir(cwd)}")
                
                # Try different model paths
                possible_paths = [
                    Path("model"),  # Hugging Face Space path
                    Path(cwd) / "model",  # Absolute Space path
                    Path("models/final"),  # Local development path
                    Path(__file__).parent.parent / "models/final"  # Relative to script
                ]
                
                model_path = None
                for path in possible_paths:
                    logger.info(f"Trying model path: {path}")
                    try:
                        if path.exists() and (path / "config.json").exists():
                            model_path = path
                            logger.info(f"Found model at: {path}")
                            break
                    except Exception as e:
                        logger.warning(f"Error checking path {path}: {str(e)}")
                
                if model_path is None:
                    raise FileNotFoundError(
                        f"Could not find model files in any of the expected locations: {[str(p) for p in possible_paths]}"
                    )
                
                # Load model with detailed error handling
                try:
                    logger.info(f"Loading model from {model_path}")
                    logger.info(f"Model directory contents: {list(model_path.glob('*'))}")
                    self.model = CustomTransformer.from_pretrained(str(model_path))
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info("Model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise
                
                # Load tokenizer with detailed error handling
                try:
                    logger.info("Loading tokenizer...")
                    self.tokenizer = SimpleTokenizer.from_pretrained(str(model_path))
                    logger.info("Tokenizer loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading tokenizer: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise
                
                self._initialized = True
                logger.info("ChatBot initialization complete")
                
            except Exception as e:
                logger.error(f"Error initializing ChatBot: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise

    def generate_response(self, message: str) -> str:
        try:
            logger.info(f"Generating response for message: {message}")
            
            # Tokenize input (no dropout during inference)
            try:
                input_ids = self.tokenizer.encode(
                    message,
                    dropout_prob=0.0  # Ensure no dropout during inference
                )
                input_tensor = torch.tensor([input_ids]).to(self.device)
                logger.info(f"Input tensor shape: {input_tensor.shape}")
            except Exception as e:
                logger.error(f"Error during tokenization: {str(e)}")
                return f"Error during tokenization: {str(e)}"
            
            # Generate response
            try:
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_tensor,
                        max_length=100,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.9
                    )
                    logger.info(f"Output tensor shape: {output_ids.shape}")
            except Exception as e:
                logger.error(f"Error during generation: {str(e)}")
                return f"Error during generation: {str(e)}"
            
            # Decode response
            try:
                response = self.tokenizer.decode(output_ids[0].tolist())
                logger.info(f"Generated response: {response}")
                return response.strip()
            except Exception as e:
                logger.error(f"Error during decoding: {str(e)}")
                return f"Error during decoding: {str(e)}"
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return f"Error: {str(e)}"

# Initialize chatbot with error handling
try:
    logger.info("Creating ChatBot instance...")
    chatbot = ChatBot()
    logger.info("ChatBot instance created successfully")
except Exception as e:
    logger.error(f"Failed to create ChatBot instance: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

def predict(message: str) -> str:
    try:
        return chatbot.generate_response(message)
    except Exception as e:
        error_msg = f"Error in predict function: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return f"Error: {str(e)}"

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

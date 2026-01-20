import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Create Gemini client
client = genai.Client(api_key=gemini_api_key)

# Read prompt from ./prompts folder
def read_prompt(filename: str) -> str:
    """Read prompt from the prompts folder."""
    prompts_dir = Path(__file__).parent / "prompts"
    prompt_path = prompts_dir / filename
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

# Custom Gemini model wrapper for deepeval
class GeminiModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        self.model_name = model_name
        self.client = client
        super().__init__(model_name)
    
    def load_model(self):
        return self.model_name
    
    def generate(self, prompt: str) -> str:
        """Generate response using Gemini API."""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Error generating content with Gemini: {str(e)}")
    
    async def a_generate(self, prompt: str) -> str:
        """Async generate response using Gemini API."""
        # For simplicity, using sync version since google-genai doesn't require true async
        return self.generate(prompt)
    
    def get_model_name(self) -> str:
        return self.model_name

# Load the criteria from the prompt file
criteria_prompt = read_prompt("test_generation_prompt.md")

# Create Gemini model instance
gemini_model = GeminiModel(model_name="gemini-3-flash-preview")

# Configure GEval to use Gemini model
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5,
    model=gemini_model
)

test_case = LLMTestCase(
    input="I have a persistent cough and fever. Should I be worried?",
    # Replace this with the actual output from your LLM application
    actual_output="A persistent cough and fever could signal various illnesses, from minor infections to more serious conditions like pneumonia or COVID-19. It's advisable to seek medical attention if symptoms worsen, persist beyond a few days, or if you experience difficulty breathing, chest pain, or other concerning signs.",
    expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
)

evaluate([test_case], [correctness_metric])
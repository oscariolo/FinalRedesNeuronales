from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import uuid
import sys
import os
from DeepLearning.models import get_model_and_tokenizer, load_fine_tuned_model

# Add DeepLearning folder to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepLearning'))

from DeepLearning.predictor import Predictor

# Global dictionary to store all predictors
predictors = {}

TUNNED_MODELS = {
    "Tabularisai": "tabularisai/multilingual-sentiment-analysis",
    "SaBert": "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis",
    "Roberta": "cardiffnlp/twitter-xlm-roberta-base-sentiment"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load all models
    global predictors
    print("Loading sentiment analysis models...")
    
    for model_name, model_path in TUNNED_MODELS.items():
        try:
            print(f"Loading {model_name} from {model_path}...")
            model, tokenizer = load_fine_tuned_model(model_path, model_name=model_name)
            # Force CPU usage to avoid CUDA compatibility issues
            predictor = Predictor(model, tokenizer, model_name=model_name, device="cpu")
            predictors[model_name] = predictor
            print(f"✅ {model_name} loaded successfully!")
            print(f"   Model info: {predictor.get_model_info()}")
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
    
    if not predictors:
        raise RuntimeError("No models were successfully loaded!")
    
    print(f"\n🚀 Successfully loaded {len(predictors)} models: {list(predictors.keys())}")
    
    yield
    
    # Shutdown: Clean up resources
    print("Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Configure CORS to allow requests from React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TweetRequest(BaseModel):
    text: str
    user: dict

class TweetResponse(BaseModel):
    id: str
    text: str
    user: dict
    sentiment: str
    model_used: str
    confidence: float
    sentiment_label: str

@app.post("/tweet/{model_name}", response_model=TweetResponse)
async def post_tweet(model_name: str, request: TweetRequest):
    """Receive a tweet, predict sentiment using specified model, and return with ID and sentiment.
    
    Args:
        model_name: The name of the model to use (Tabularisai, SaBert, or Roberta)
        request: Tweet text and user data
    """
    
    # Check if model exists
    if model_name not in predictors:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found. Available models: {list(predictors.keys())}"
        )
    
    predictor = predictors[model_name]
    
    # Run prediction in a separate thread to avoid blocking
    result = await asyncio.to_thread(predictor.predict_single, request.text)
    
    # Get sentiment label (very_negative, negative, neutral, positive, very_positive)
    sentiment_label = result['sentiment_label']
    confidence = result['confidence']
    
    # Map to simplified sentiment for React app
    sentiment_mapping = {
        'very_negative': 'negative',
        'negative': 'negative',
        'neutral': 'neutral',
        'positive': 'positive',
        'very_positive': 'positive'
    }
    
    sentiment = sentiment_mapping.get(sentiment_label, 'neutral')
    
    # Generate unique ID for the tweet
    tweet_id = str(uuid.uuid4())
    
    return TweetResponse(
        id=tweet_id,
        text=request.text,
        user=request.user,
        sentiment=sentiment,
        model_used=model_name,
        confidence=confidence,
        sentiment_label=sentiment_label
    )

@app.get("/")
async def root():
    return {
        "message": "Tweet Sentiment Analysis API",
        "available_models": list(predictors.keys())
    }

@app.get("/models")
async def list_models():
    """List all available models and their info."""
    return {"models": list(TUNNED_MODELS.keys())}

@app.get("/model-info/{model_name}")
async def model_info(model_name: str):
    """Get information about a specific model."""
    if model_name not in predictors:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found. Available models: {list(predictors.keys())}"
        )
    return predictors[model_name].get_model_info()

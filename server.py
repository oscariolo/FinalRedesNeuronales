from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import uuid
import sys
import os

# Add DeepLearning folder to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepLearning'))

from predictor import Predictor

# Global predictor instance
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    global predictor
    print("Loading sentiment analysis model...")
    
    # Option 1: Load a fine-tuned model from a specific path
    # Uncomment and modify the path to your fine-tuned model
    # model_path = "./fine_tuned_models/YourModelName_20240131_120000"
    # predictor = Predictor.from_pretrained_path(model_path, model_name="Tabularisai")
    
    # Option 2: Load base model (for testing or when no fine-tuned model exists)
    from models import get_model_and_tokenizer
    model, tokenizer = get_model_and_tokenizer("Tabularisai")
    # Force CPU usage to avoid CUDA compatibility issues
    predictor = Predictor(model, tokenizer, model_name="Tabularisai", device="cpu")
    
    print("Model loaded successfully!")
    print(f"Model info: {predictor.get_model_info()}")
    
    yield
    
    # Shutdown: Clean up resources (if needed)
    print("Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Configure CORS to allow requests from React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
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

@app.post("/tweet", response_model=TweetResponse)
async def post_tweet(request: TweetRequest):
    """Receive a tweet, predict sentiment, and return with ID and sentiment."""
    
    # Run prediction in a separate thread to avoid blocking
    result = await asyncio.to_thread(predictor.predict_single, request.text)
    
    # Get sentiment label (very_negative, negative, neutral, positive, very_positive)
    sentiment_label = result['sentiment_label']
    
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
        sentiment=sentiment
    )

@app.get("/")
async def root():
    return {"message": "Tweet Sentiment Analysis API"}

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if predictor:
        return predictor.get_model_info()
    return {"error": "Model not loaded"}

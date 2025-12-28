from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.core.batcher import DynamicBatcher
from src.models.nlp import NLPModel
import uvicorn
import os
import asyncio

# config
MAX_BATCH_SIZE = 64
MAX_LATENCY_MS = 5.0
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
ENABLE_BATCHING = os.getenv("ENABLE_BATCHING", "True").lower() == "true"

batcher = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # this runs when the server starts
    global batcher, model
    print("initializing model...")
    
    # create model
    model = NLPModel(model_name=MODEL_NAME)
    model.load()

    if ENABLE_BATCHING:
        print("batching enabled")
        batcher = DynamicBatcher(model, max_batch_size=MAX_BATCH_SIZE, max_latency_ms=MAX_LATENCY_MS)
        # start the background loop
        await batcher.start()
    else:
        print("batching disabled (direct inference)")
        batcher = None
    
    yield # server runs here
    
    # this runs when the server stops
    print("shutting down...")
    if batcher:
        await batcher.stop()

class PredictRequest(BaseModel):
    # defines what the user must send us
    text: str

class PredictResponse(BaseModel):
    # defines what we send back
    label: str
    score: float
    
app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    # the main endpoint for inference
    
    if ENABLE_BATCHING:
        if not batcher:
            raise HTTPException(status_code=503, detail="batcher not initialized")
        try:
            # send text to the batcher and wait for the result
            result = await batcher.predict(request.text)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # no batching: run directly
        if not model:
            raise HTTPException(status_code=503, detail="model not initialized")
        
        try:
            # run in executor to avoid blocking the event loop completely
            
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, model.predict, [request.text])
            return results[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

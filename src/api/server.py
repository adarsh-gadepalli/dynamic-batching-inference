from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.core.batcher import DynamicBatcher
from src.core.continuous_batcher import ContinuousBatcher
from src.models.nlp import NLPModel
from src.models.gen import GenerativeModel
import uvicorn
import os
import asyncio
import torch

# config
# BATCHING_TYPE: "NONE", "DYNAMIC", "CONTINUOUS"
BATCHING_TYPE = os.getenv("BATCHING_TYPE", "DYNAMIC").upper()
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
MAX_LATENCY_MS = float(os.getenv("MAX_LATENCY_MS", "10.0"))


MODEL_NAME = "gpt2" 

batcher = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global batcher, model
    print(f"initializing server with batching_type={BATCHING_TYPE}...")
    
    model = GenerativeModel(model_name=MODEL_NAME)
    model.load()

    if BATCHING_TYPE == "CONTINUOUS":
        print("continuous batching enabled")
        batcher = ContinuousBatcher(model, max_batch_size=MAX_BATCH_SIZE)
        await batcher.start()
        
    elif BATCHING_TYPE == "DYNAMIC":
        print("dynamic batching enabled")

        batcher = DynamicBatcher(model, max_batch_size=MAX_BATCH_SIZE, max_latency_ms=MAX_LATENCY_MS)
        await batcher.start()
        
    else: 
        print("batching disabled (direct inference)")
        batcher = None
    
    yield
    
    print("shutting down...")
    if batcher:
        await batcher.stop()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    result: str 
    
app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        if BATCHING_TYPE in ["DYNAMIC", "CONTINUOUS"]:
            if not batcher:
                raise HTTPException(status_code=503, detail="batcher not initialized")
            
            result = await batcher.predict(request.text)
            return PredictResponse(result=str(result))
            
        else:
            # no batching
            if not model:
                raise HTTPException(status_code=503, detail="model not initialized")
            
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, model.predict, [request.text])
            return PredictResponse(result=results[0])
            
    except Exception as e:
        print(f"error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

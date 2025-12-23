from fastapi import FastAPI, HTTPException 
from contextlib import asynccontextmanager
from src.core.batcher import DynamicBatcher
from src.models.nlp import NLPModel
import uvicorn


# config
MAX_BATCH_SIZE =32
MAX_LATENCY_MS = 10.0
MODEL_NAME = "prajjwal1/bert-tiny"
batcher = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # this runs when the server starts
    global batcher
    print("initializing model and batcher...")
    
    # create model and batcher
    model = NLPModel(model_name=MODEL_NAME)
    batcher = DynamicBatcher(model, max_batch_size=MAX_BATCH_SIZE, max_latency_ms=MAX_LATENCY_MS)
    
    # start the background loop
    await batcher.start()
    
    yield # server runs here
    
    # this runs when the server stops
    print("shutting down...")
    await batcher.stop()


class PredictRequest():
    # defines what the user must send us
    text: str

class PredictResponse():
    # defines what we send back
    label: str
    score: float

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    # the main endpoint for inference
    if not batcher:
        raise HTTPException(status_code=503, detail="model not initialized")
    
    try:
        # send text to the batcher and wait for the result
        result = await batcher.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    app = FastAPI(lifespan=lifespan)
    # run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)


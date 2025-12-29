from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.core.dynamic_batcher import DynamicBatcher
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
ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "0") == "1"


MODEL_NAME = "gpt2" 

batcher = None
model = None
profiler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global batcher, model, profiler
    print(f"initializing server with batching_type={BATCHING_TYPE}...")
    
    model = GenerativeModel(model_name=MODEL_NAME)
    model.load()

    if ENABLE_PROFILING:
        print(f"PROFILING ENABLED for {BATCHING_TYPE}")
        # Capture a trace of the execution
        # schedule: wait 2 steps, warmup 2 steps, active 5 steps
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./traces/{BATCHING_TYPE.lower()}"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()

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
        
    if profiler:
        profiler.stop()
        print(f"Trace saved to ./traces/{BATCHING_TYPE.lower()}")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    result: str 
    
app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        if profiler:
            profiler.step()

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

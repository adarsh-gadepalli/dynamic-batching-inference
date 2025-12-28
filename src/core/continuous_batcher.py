import asyncio
import time
import torch
from typing import List, Any, Dict
from dataclasses import dataclass, field

@dataclass
class ContinuousRequest:
    request_id: str
    input_text: str
    future: asyncio.Future = field(default_factory=asyncio.Future)
    # state for generation
    input_ids: Any = None
    generated_tokens: List[int] = field(default_factory=list)
    finished: bool = False

class ContinuousBatcher:
    def __init__(self, model, max_batch_size: int = 32):
        self.model = model
        self.max_batch_size = max_batch_size
        self.queue = asyncio.Queue()
        self.running = False
        self._loop_task = None
        self.active_requests: List[ContinuousRequest] = []

    async def start(self):
        self.model.load()
        self.running = True
        self._loop_task = asyncio.create_task(self._loop())
        print("continuous batcher started.")

    async def stop(self):
        self.running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    async def predict(self, text: str) -> str:
        req = ContinuousRequest(
            request_id=str(time.time()),
            input_text=text
        )
        await self.queue.put(req)
        return await req.future

    async def _loop(self):
        while self.running:
            # 1. fill empty slots from queue
            while len(self.active_requests) < self.max_batch_size and not self.queue.empty():
                try:
                    req = self.queue.get_nowait()
                    # tokenize immediately
                    inputs = self.model.tokenizer(req.input_text, return_tensors="pt").to(self.model.device)
                    req.input_ids = inputs.input_ids
                    self.active_requests.append(req)
                except asyncio.QueueEmpty:
                    break

            if not self.active_requests:
                await asyncio.sleep(0.001)
                continue

            # padding inputs to match largest in batch for this step.
            
            try:
                # prepare batch inputs
                # find max length in current active set to pad efficiently
                current_input_ids = [req.input_ids for req in self.active_requests]

                # we simulate the "work" by running a forward pass on the longest sequence.
                
                # optimization: simplistic batching
                max_len = max(t.size(1) for t in current_input_ids)
                padded_inputs = []
                for t in current_input_ids:
                    pad_len = max_len - t.size(1)
                    if pad_len > 0:
                        # left pad for generation
                        padded_inputs.append(torch.nn.functional.pad(t, (pad_len, 0), value=self.model.tokenizer.pad_token_id))
                    else:
                        padded_inputs.append(t)
                
                batch_tensor = torch.cat(padded_inputs, dim=0)
                
                # run forward pass (generate 1 token)
                next_tokens_ids, _ = self.model.generate_step(batch_tensor)
                
                # 3. process results and manage state
                finished_indices = []
                for i, req in enumerate(self.active_requests):
                    token_id = next_tokens_ids[i].item()
                    req.generated_tokens.append(token_id)
                    
                    # update input_ids for next step (append new token)
                    req.input_ids = torch.cat([req.input_ids, next_tokens_ids[i].view(1, 1)], dim=1)
                    
                    # check stop conditions (eos or length)
                    if token_id == self.model.tokenizer.eos_token_id or len(req.generated_tokens) >= 20: # cap at 20 tokens for speed
                        req.finished = True
                        finished_indices.append(i)
                        
                        # set result
                        full_text = self.model.tokenizer.decode(req.generated_tokens)
                        req.future.set_result(full_text)

                # remove finished requests 
                # they leave immediately, making room for queue items in the next loop iteration
                for i in sorted(finished_indices, reverse=True):
                    self.active_requests.pop(i)
                    
            except Exception as e:
                print(f"error in continuous loop: {e}")
                
                # fail all to recover
                for req in self.active_requests:
                    if not req.future.done():
                        req.future.set_exception(e)
                self.active_requests = []
                await asyncio.sleep(0.1)


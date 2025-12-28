import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Tuple
from .base import BaseModel

class GenerativeModel(BaseModel):
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.max_length = 20 # aligned with continuous batcher limit for fair comparison
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def load(self):
        print(f"loading generative model {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # set pad token if not set (gpt2 doesn't have one)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("model loaded.")

    @torch.inference_mode()
    def predict(self, texts: List[str]) -> List[str]:
        """standard batch generation (dynamic batching use case)."""
        if not texts:
            return []
            
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=self.max_length,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    @torch.inference_mode()
    def generate_step(self, 
                      input_ids: torch.Tensor, 
                      past_key_values: Any = None, 
                      attention_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Any]:
        """runs a single forward pass for continuous batching."""
        
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True
        )
        
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        
        return next_tokens, outputs.past_key_values


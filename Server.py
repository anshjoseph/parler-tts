from fastapi import FastAPI
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import torch
from contextlib import asynccontextmanager
import uvicorn
from pydantic import BaseModel
import time
from base64 import b64encode
import numpy as np


torch_dtype = None
model = None
tokenizer = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global torch_dtype
    global model
    global tokenizer
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    if torch.backends.mps.is_available():
        device = "mps"
    torch_dtype = torch.float16 if device != "cpu" else torch.float32
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device, dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")    
    yield
    # Clean up the ML models and release the resources
    torch.cuda.empty_cache()

class TTSconfig(BaseModel):
    text:str
    prompt:str
    def tokenized(self):
        input_ids = tokenizer(self.prompt, return_tensors="pt").input_ids.to(torch_dtype)
        prompt_input_ids = tokenizer(self.text, return_tensors="pt").input_ids.to(torch_dtype)
        return {"input_ids":input_ids,"prompt_input_ids":prompt_input_ids}

app = FastAPI(lifespan=lifespan)

@app.post("/tts")
def tts(respose:TTSconfig):
    t1 = time.time()
    output = respose.tokenized()
    print(output)
    generation = model.generate(input_ids=output["input_ids"], prompt_input_ids=output["prompt_input_ids"]).to(torch.float32)
    audio_arr:np.ndarray = generation.cpu().numpy().squeeze()
    return {'audio': b64encode(audio_arr.tobytes()).decode(),'sr':model.config.sampling_rate,"time":time.time() - t1}

if __name__ == "__main__":
    uvicorn.run("Server:app",host='0.0.0.0',port=9000,reload=True)
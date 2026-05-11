import torch
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

app = FastAPI()
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class RequestModel(BaseModel):
    query: str

model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    print(f"Loading {MODEL_NAME} onto GPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded successfully!")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process")
def process(req: RequestModel):
    inputs = tokenizer(req.query, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    Thread(
        target=model.generate,
        kwargs=dict(**inputs, streamer=streamer, max_new_tokens=1024)
    ).start()

    def stream_generator():
        for text in streamer:
            yield text

    return StreamingResponse(stream_generator(), media_type="text/plain")

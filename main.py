import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
login(HUGGINGFACE_API_KEY)

# Load model (tương tự Colab)
MODEL_PATH = os.getenv("MODEL_PATH")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    cfg = PeftConfig.from_pretrained(MODEL_PATH)
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, device_map="auto")
    model = PeftModel.from_pretrained(base, MODEL_PATH)
except:
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

app = FastAPI()

class Prompt(BaseModel):
    text: str

@app.post("/predict")
async def predict(p: Prompt):
    inputs = tokenizer(p.text, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=200)
    return {"response": tokenizer.decode(out[0], skip_special_tokens=True)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

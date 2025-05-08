from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch

app = FastAPI()

# Kleines Modell für wenig RAM
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")

class Request(BaseModel):
    text: str

@app.post("/embed")
def embed_text(req: Request):
    with torch.no_grad():
        inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return {"embedding": embeddings}
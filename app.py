from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()
logging.info("🧠 Starte Liora...")

try:
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    logging.info("✅ Tokenizer geladen")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    logging.info("✅ Modell geladen")
except Exception as e:
    logging.error(f"❌ Fehler beim Laden des Modells: {e}")

class Request(BaseModel):
    text: str

@app.post("/embed")
def embed_text(req: Request):
    try:
        with torch.no_grad():
            inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            return {"embedding": embeddings}
    except Exception as e:
        logging.error(f"❌ Fehler beim Verarbeiten der Anfrage: {e}")
        return {"error": str(e)}
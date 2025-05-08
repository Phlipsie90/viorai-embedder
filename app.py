from fastapi import FastAPI
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI()

# Firebase Key wird über Umgebungsvariable bereitgestellt
cred_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Modell laden
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")

class Query(BaseModel):
    text: str

def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

@app.post("/embed")
def find_best_match(req: Query):
    query_embedding = get_embedding(req.text).reshape(1, -1)

    docs = db.collection("liora_answers").stream()
    best_score = -1
    best_answer = "Keine passende Antwort gefunden."

    for doc in docs:
        data = doc.to_dict()
        if "embedding" in data and isinstance(data["embedding"], list):
            doc_embedding = np.array(data["embedding"]).reshape(1, -1)
            score = cosine_similarity(query_embedding, doc_embedding)[0][0]
            if score > best_score:
                best_score = score
                best_answer = data.get("answer", best_answer)

    return {
        "input": req.text,
        "best_match_score": float(best_score),
        "answer": best_answer
    }
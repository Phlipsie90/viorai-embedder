from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("thenlper/gte-small")

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
def embed_text(req: EmbedRequest):
    embedding = model.encode(req.text).tolist()
    return {"embedding": embedding}
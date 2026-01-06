from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama

# =========================
# Chargements
# =========================
index = faiss.read_index("faiss_index.index")
embeddings = np.load("embeddings.npy")

with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f.readlines()]

sbert = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# API
# =========================
app = FastAPI(title="RAG Baudelaire + Phi-3")

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):

    # 1️⃣ Embedding de la question
    q_emb = sbert.encode([q.question])
    faiss.normalize_L2(q_emb)

    # 2️⃣ Recherche FAISS
    _, indices = index.search(q_emb, 3)
    context = "\n".join([chunks[i] for i in indices[0]])

    # 3️⃣ Prompt RAG
    prompt = f"""
Tu es un assistant littéraire spécialisé dans l'œuvre de Charles Baudelaire (Les Fleurs du Mal).

RÈGLES STRICTES :
- Réponds uniquement à partir du CONTEXTE fourni
- Ne fais aucune interprétation extérieure
- Si plusieurs passages sont fournis, synthétise-les
- Reformule avec des mots simples
- Réponds de façon courte, claire et précise (niveau étudiant de faculté)
- Si la réponse n’est pas dans le contexte, réponds exactement : "Je ne sais pas."

Contexte extrait des Fleurs du Mal :
{context}

Question :
{q.question}

"""

    # 4️⃣ Appel Phi-3
    response = ollama.chat(
        model="phi3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response["message"]["content"]

    return {
        "question": q.question,
        "answer": answer,
        "sources": indices[0].tolist()
    }

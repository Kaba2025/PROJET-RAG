# ---------------------------
# chunking_par_strophe.py
# ---------------------------

import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# 📂 Chemin du fichier
corpus_path = r"C:\Users\KABA\OneDrive\Desktop\PROJET\Data\donnees\LES_FLEURS_DU_MAL.txt"

# Vérification que le fichier existe
if not os.path.exists(corpus_path):
    raise FileNotFoundError(f"Fichier non trouvé : {corpus_path}")

# Lecture du texte
with open(corpus_path, 'r', encoding='utf-8') as f:
    texte = f.read()

# ---------------------------
# Découpage par strophe
# ---------------------------
# Une strophe = lignes séparées par une ligne vide
raw_strophes = texte.split("\n\n")

# Nettoyage
strophes = []
for s in raw_strophes:
    clean = " ".join(line.strip() for line in s.splitlines() if line.strip())
    if len(clean) > 20:  # évite les lignes trop courtes ou titres
        strophes.append(clean)

print(f"Nombre de strophes détectées : {len(strophes)}")
print("Exemple de strophe :\n", strophes[0])

# ---------------------------
# Chunking avec chevauchement
# ---------------------------
def chunk_strophes(strophes, chunk_size=2, overlap=1):
    """
    chunk_size : nombre de strophes par chunk
    overlap : nombre de strophes qui se répètent
    """
    chunks = []
    i = 0
    while i < len(strophes):
        chunk = " ".join(strophes[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

chunks = chunk_strophes(strophes, chunk_size=2, overlap=1)

print(f"Nombre de chunks créés : {len(chunks)}")
print("Premier chunk :\n", chunks[0])

# ---------------------------
# Embeddings SBERT
# ---------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, show_progress_bar=True)

np.save("embeddings.npy", embeddings)
print("Embeddings sauvegardés")

# ---------------------------
# Index FAISS
# ---------------------------
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
faiss.normalize_L2(embeddings)
index.add(embeddings)

faiss.write_index(index, "faiss_index.index")
print(f"Index FAISS créé avec {index.ntotal} vecteurs")

# ---------------------------
# Sauvegarde des chunks
# ---------------------------
with open("chunks.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n")

print("chunks.txt créé avec succès")

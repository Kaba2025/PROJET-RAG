# PROJET-RAG
Ce projet met en place un système RAG (Retrieval-Augmented Generation) pour l’œuvre Les Fleurs du Mal de Charles Baudelaire. Il permet de poser des questions sur le texte et de recevoir des réponses précises et basées uniquement sur le corpus, grâce à l’association d’une recherche de passages pertinents et d’un modèle de langage (LLM Phi-3).

Le projet illustre comment combiner :  

- Le **prétraitement et le découpage** d’un texte littéraire en strophes et chunks pour faciliter la recherche.  
- La **représentation vectorielle (embeddings)** avec Sentence-BERT.  
- L’**indexation et la recherche par similarité** avec FAISS.  
- L’**API FastAPI** pour interroger le corpus en ligne.  
- La génération de réponses via un **LLM**, en s’appuyant uniquement sur le contexte récupéré.  

---

## Fonctionnalités

- Posez des questions sur *Les Fleurs du Mal* via une API REST `/ask`.  
- Récupération automatique des passages pertinents du texte.  
- Réponses courtes, claires et synthétiques.  
- Gestion des sources : chaque réponse indique les passages utilisés pour la génération.  

---

## Structure du projet

RAG-Baudelaire
│
├─ chunking_par_strophe.py # Découpe du texte, création des chunks et embeddings
├─ api_rag.py # API FastAPI pour poser des questions
├─ chunks.txt # Chunks générés à partir du texte
├─ embeddings.npy # Embeddings SBERT des chunks
├─ faiss_index.index # Index FAISS pour la recherche rapide
└─ La génération de réponses via un **LLM** : Ollama / Phi-3

## Technologies utilisées

- Python 3.10+  
- [Sentence-Transformers](https://www.sbert.net/) (all-MiniLM-L6-v2) pour les embeddings  
- [FAISS](https://github.com/facebookresearch/faiss) pour l’indexation et la recherche par similarité  
- [FastAPI](https://fastapi.tiangolo.com/) pour l’API REST  
- [Ollama / Phi-3](https://ollama.com/) pour la génération des réponses 


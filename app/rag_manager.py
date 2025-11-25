import chromadb
from sentence_transformers import SentenceTransformer

class RAGManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./vectorstore")
        self.collection = self.client.get_or_create_collection("ielts_rag")
        self.embedder = SentenceTransformer("BAAI/bge-small-en")

    def embed(self, text: str):
        return self.embedder.encode([text])[0].tolist()

    def add_document(self, id: str, text: str, metadata: dict):
        self.collection.add(
            ids=[id],
            documents=[text],
            embeddings=[self.embed(text)],
            metadatas=[metadata]
        )

    def retrieve(self, query: str, top_k=6):
        results = self.collection.query(
            query_embeddings=[self.embed(query)],
            n_results=top_k
        )
        return results["documents"][0]

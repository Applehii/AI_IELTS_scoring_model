import chromadb
from sentence_transformers import SentenceTransformer


class RAGManager:
    def __init__(self):
        # Lưu vectorstore trong thư mục local
        self.client = chromadb.PersistentClient(path="./vectorstore")
        self.collection = self.client.get_or_create_collection("ielts_rag")
        self.embedder = SentenceTransformer("BAAI/bge-small-en")

    def embed(self, text: str):
        """Convert text → vector embedding."""
        return self.embedder.encode([text])[0].tolist()

    def add_document(self, doc_id: str, text: str, metadata: dict, embedding_text: str = None):
        """
        Add document to Chroma.
        - embedding_text: SUMMARY dùng để embed.
        - text: full document.
        """
        if embedding_text is None:
            embedding_text = text

        embedding = self.embed(embedding_text)

        self.collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],  # metadata phải là str/int/float/bool/None, không được list
        )

    def retrieve(self, query: str, top_k: int = 6, where: dict | None = None):
        """
        Query Chroma, có hỗ trợ filter metadata (where).
        """
        query_args = {
            "query_embeddings": [self.embed(query)],
            "n_results": top_k,
        }

        if where:
            query_args["where"] = where

        results = self.collection.query(**query_args)

        return {
            "ids": results["ids"][0],
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
        }

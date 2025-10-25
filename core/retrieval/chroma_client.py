import chromadb
from chromadb.utils import embedding_functions

class ChromaDBClient:
    def __init__(self, collection_name="medical_chunks"):
        self.client = chromadb.PersistentClient(path="./chroma_db")  # ✅ persistent storage
        #self.embedding_fn = embedding_functions.HuggingFaceEmbeddingFunction(
        #    model_name="BAAI/bge-m3"
        #)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn
        )

    def add_chunk(self, chunk_id, chunk_text, metadata):
        self.collection.add(
            ids=[chunk_id],
            documents=[chunk_text],
            metadatas=[metadata],
        )

    def query(self, query, n_results=3):
        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

import chromadb

class ChromaDBClient:
    def __init__(self, collection_name="medical_chunks"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunk(self, chunk_id, chunk_text, metadata):
        self.collection.add(
            ids=[chunk_id],
            documents=[chunk_text],
            metadatas=[metadata],
        )

from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "? ", "! "],
            chunk_size=600,
            chunk_overlap=50,
            length_function=len,
        )

    def create_chunks(self, raw_text: str):
        return self.splitter.create_documents([raw_text])

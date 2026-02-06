from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class Embeddings:
    def embed(self, documents):
        pass

    def load(self):
        pass


# Concrete class for HuggingFace embeddings
class HuggingFaceEmbedding(Embeddings):
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()

    def embed(self, documents):
        return self.embeddings.embed_documents(documents)

    def load(self):
        return self.embeddings


class GeminiEmbedding(Embeddings):
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    def embed(self, documents):
        return self.embeddings.embed_documents(documents)

    def load(self):
        return self.embeddings
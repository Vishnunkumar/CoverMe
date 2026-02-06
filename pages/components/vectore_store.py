from langchain_community.vectorstores import SKLearnVectorStore


class VectorStore:
    def from_doc(self, documents, embedding, persist_path):
        pass


# Concrete class for vector store retriever
class SklearnVecStore(VectorStore):

    def from_doc(self, documents, embedding, persist_path):
        return SKLearnVectorStore.from_documents(documents, embedding, persist_path)

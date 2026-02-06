from langchain_community.document_loaders import PDFPlumberLoader


class DocumentLoader:
    def load_and_split(self, text_splitter):
        pass


# Concrete class for PDF document loading
class PDFDocumentLoader(DocumentLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self, text_splitter):
        with open(self.file_path, 'rb') as pdf_file:
            loader = PDFPlumberLoader(self.file_path)
            data = loader.load_and_split(text_splitter=text_splitter)
        return data

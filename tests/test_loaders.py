from app.loaders.text_loader import TextFileLoader
from app.loaders.pdf_loader import PDFFileLoader


def test_text_loader():
    loader = TextFileLoader("data/raw_docs/sample.txt")
    docs = loader.load()
    print("TXT Docs:", len(docs))
    print(docs[0].metadata)


def test_pdf_loader():
    loader = PDFFileLoader("data/raw_docs/sample.pdf")
    docs = loader.load()
    print("PDF Docs:", len(docs))
    print(docs[0].metadata)


if __name__ == "__main__":
    test_text_loader()
    test_pdf_loader()

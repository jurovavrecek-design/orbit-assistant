from app.ingest.document_ingestor import DocumentIngestor
from app.vectorstore.chroma_store import ChromaStore


def main():

    print("===================================")
    print("ORBIT Assistant Document Ingestion")
    print("===================================")

    print("\nLoading documents...")

    ingestor = DocumentIngestor()

    docs = ingestor.load_documents()

    print(f"Loaded {len(docs)} documents")

    if len(docs) == 0:
        print("No documents found.")
        return

    print("\nInitializing ChromaDB...")

    store = ChromaStore()

    print("Adding documents to vector store...")

    store.add_documents(docs)

    print("Building hybrid retriever...")

    print("\nDocuments indexed successfully!")
    print("\nDone.")


if __name__ == "__main__":
    main()
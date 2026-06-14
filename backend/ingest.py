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

    #
    # Build hybrid BM25 + Vector retriever
    #
    print("Building hybrid retriever...")  

    print("\nDocuments indexed successfully!")

    print("\nDone.")

for d in docs:

    if (
        d.metadata.get("type") == "pptx"
        and "Recording a Call in ORBIT Online" in d.page_content
    ):

        print("FOUND")
        print(d.metadata)
        print(d.page_content[:1000])
        
if __name__ == "__main__":
    main()
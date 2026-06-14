import pickle

from app.ingest.document_ingestor import DocumentIngestor


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

    print("\nTokenizing documents...")

    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in docs
    ]

    print("Saving BM25 index...")

    with open("bm25.pkl", "wb") as f:

        pickle.dump(
            {
                "documents": docs,
                "tokenized_docs": tokenized_docs
            },
            f
        )

    print("\nBM25 index created successfully!")
    print("\nDone.")


if __name__ == "__main__":
    main()
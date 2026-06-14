from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class ChromaStore:

    def __init__(self):

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )

    def add_documents(self, docs):

        self.db.add_documents(docs)

    def similarity_search(self, question):

        docs = self.db.max_marginal_relevance_search(
            question,
            k=20,
            fetch_k=50
        )

        keywords = question.lower().split()

        docs.sort(
            key=lambda d:
            sum(
                word in d.page_content.lower()
                for word in keywords
            ),
            reverse=True
        )

        return docs[:10]
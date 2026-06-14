from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class ChromaStore:

    def __init__(self):

        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://host.docker.internal:11434"
        )

        self.db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )

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
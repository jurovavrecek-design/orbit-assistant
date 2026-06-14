from app.llm.groq_provider import GroqProvider
from app.retriever.bm25_retriever import BM25Retriever
from app.reranker.reranker import Reranker


class ChatService:

    def __init__(self):

        self.llm = GroqProvider()
        self.retriever = BM25Retriever()
        self.reranker = Reranker()

    def retrieve_docs(self, question):

        docs = self.retriever.search(
            question,
            k=50
        )

        docs = self.reranker.rerank(
            question,
            docs,
            top_k=8
        )

        print()
        print("========== FINAL RETRIEVED ==========")

        if not docs:
            print("No relevant documents found.")

        for doc in docs:

            print(doc.metadata)
            print(doc.page_content[:500])
            print("--------------------------------")

        print("==============================")
        print()

        return docs

    def build_context(self, docs):

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    def ask(self, question):

        docs = self.retrieve_docs(question)

        context = self.build_context(docs)

        answer = self.llm.ask(
            question,
            context
        )

        sources = []
        seen = set()

        for doc in docs:

            source = doc.metadata["source"]

            if "slide" in doc.metadata:
                source += f" (slide {doc.metadata['slide']})"

            if source not in seen:
                seen.add(source)
                sources.append(source)

        return {
            "answer": answer,
            "sources": sources
        }

    def stream(self, question):

        docs = self.retrieve_docs(question)

        context = self.build_context(docs)

        for chunk in self.llm.stream_answer(
            question,
            context
        ):
            yield chunk
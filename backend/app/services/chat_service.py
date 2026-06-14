from app.llm.groq_provider import GroqProvider
from app.retriever.bm25_retriever import BM25Retriever


class ChatService:

    def __init__(self):

        self.llm = GroqProvider()
        self.retriever = BM25Retriever()

    def retrieve_docs(self, question):

        docs = self.retriever.search(
            question
        )

        docs = self.llm.rerank(
            question,
            docs
        )

        print()
        print("========== FINAL RETRIEVED ==========")

        if len(docs) == 0:

            print("No relevant documents found.")

        else:

            for doc in docs:

                print(doc.metadata)

                print(doc.page_content[:500])

                print("--------------------------------")

        print("==============================")
        print()

        return docs

    def build_context(self, docs):

        if len(docs) == 0:
            return ""

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    def ask(self, question):

        docs = self.retrieve_docs(
            question
        )

        if len(docs) == 0:

            return {
                "answer": "The documentation does not contain the answer.",
                "sources": []
            }

        context = self.build_context(
            docs
        )

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

        docs = self.retrieve_docs(
            question
        )

        if len(docs) == 0:

            yield "The documentation does not contain the answer."
            return

        context = self.build_context(
            docs
        )

        for chunk in self.llm.stream_answer(
                question,
                context
        ):
            yield chunk
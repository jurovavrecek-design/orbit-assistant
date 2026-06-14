from app.llm.groq_provider import GroqProvider
from app.retriever.bm25_retriever import BM25Retriever


class ChatService:

    def __init__(self):

        self.llm = GroqProvider()
        self.retriever = BM25Retriever()

    def retrieve_docs(self, question):

        bm25_docs = self.retriever.search(
            question
        )

        print()
        print("========== BM25 ==========")

        for doc in bm25_docs[:10]:
            print(doc.metadata)

        print("==========================")

        reranked_docs = self.llm.rerank(
            question,
            bm25_docs
        )

        if reranked_docs:
            docs = reranked_docs
        else:
            docs = bm25_docs[:5]

        print()
        print("========== FINAL RETRIEVED ==========")

        if docs:
            for doc in docs:

                print(doc.metadata)

                print(
                    doc.page_content[:500]
                )

                print("--------------------------------")

        else:
            print("No relevant documents found.")

        print("==============================")
        print()

        return docs

    def build_context(self, docs):

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    def ask(self, question):

        docs = self.retrieve_docs(
            question
        )

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
                source += (
                    f" (slide {doc.metadata['slide']})"
                )

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

        context = self.build_context(
            docs
        )

        for chunk in self.llm.stream_answer(
            question,
            context
        ):
            yield chunk
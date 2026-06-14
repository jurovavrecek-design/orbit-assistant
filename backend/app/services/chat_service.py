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

        print()
        print("========== RETRIEVED ==========")

        for doc in docs:

            print(doc.metadata)
            print()
            print(doc.page_content[:1500])
            print()
            print("--------------------------------------------")

        print("==============================")
        print()

        return docs

    def build_context(self, docs):

        context = ""

        for i, doc in enumerate(docs, start=1):

            context += f"""
    DOCUMENT {i}

    SOURCE:
    {doc.metadata['source']}

    CONTENT:
    {doc.page_content}

    ------------------------
    """

        return context

    def ask(self, question):

        docs = self.retrieve_docs(
            question
        )

        context = self.build_context(
            docs
        )

        print()
        print("========== CONTEXT ==========")
        print(context[:5000])
        print("=============================")
        print()

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

        context = self.build_context(
            docs
        )

        for chunk in self.llm.stream_answer(
            question,
            context
        ):
            yield chunk
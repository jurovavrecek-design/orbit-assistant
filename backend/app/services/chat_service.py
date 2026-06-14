from app.llm.ollama_provider import OllamaProvider
from app.vectorstore.chroma_store import ChromaStore


class ChatService:

    def __init__(self):

        self.llm = OllamaProvider()
        self.store = ChromaStore()

    def retrieve_docs(self, question):

        docs = self.store.similarity_search(
            question
        )

        print()
        print("========== RETRIEVED ==========")

        for doc in docs:
            print(doc.metadata)

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
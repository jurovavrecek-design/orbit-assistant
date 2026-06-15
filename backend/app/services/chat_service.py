from app.llm.groq_provider import GroqProvider
from app.retriever.bm25_retriever import BM25Retriever


class ChatService:

    def __init__(self):

        self.llm = GroqProvider()
        self.retriever = BM25Retriever()

    def retrieve_docs(self, question):

        #
        # BM25 + rule-based rerank
        #

        docs = self.retriever.search(
            question,
            k=10
        )

        #
        # Remove duplicates (EU/LATAM slides etc.)
        #

        unique_docs = []
        seen = set()

        for doc in docs:

            text_key = doc.page_content[:300]

            if text_key in seen:
                continue

            seen.add(text_key)
            unique_docs.append(doc)

        #
        # Keep only best docs
        #

        docs = unique_docs[:5]

        print()
        print("========== FINAL RETRIEVED ==========")

        if not docs:
            print("No relevant documents found.")

        for doc in docs:

            source = doc.metadata["source"]

            if "slide" in doc.metadata:
                source += f" slide {doc.metadata['slide']}"

            elif "page" in doc.metadata:
                source += f" page {doc.metadata['page'] + 1}"

            print(source)
            print("--------------------------------")

        print("==============================")
        print()

        return docs

    def build_context(self, docs):

        sections = []

        for doc in docs:

            header = doc.metadata["source"]

            if "slide" in doc.metadata:
                header += f" slide {doc.metadata['slide']}"

            elif "page" in doc.metadata:
                header += f" page {doc.metadata['page'] + 1}"

            sections.append(
                f"""
SOURCE:
{header}

CONTENT:
{doc.page_content}
"""
            )

        return "\n\n--------------------------------\n\n".join(
            sections
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

            elif "page" in doc.metadata:
                source += (
                    f" (page {doc.metadata['page'] + 1})"
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
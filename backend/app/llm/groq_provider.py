from langchain_groq import ChatGroq

from app.llm.base_provider import BaseLLMProvider


class GroqProvider(BaseLLMProvider):

    def __init__(self):

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )

    def build_prompt(self, question, context):

        return f"""
You are ORBIT Assistant.

Use ONLY information contained in the documentation.

Do not use general knowledge.

If the answer is not present in the documentation, say:

"The documentation does not contain the answer."

Answer in the same language as the question.

Use bullet points.

Documentation:

{context}

Question:

{question}
"""

    def ask(self, question, context=""):

        prompt = self.build_prompt(
            question,
            context
        )

        response = self.llm.invoke(
            prompt
        )

        return response.content

    def stream_answer(self, question, context=""):

        prompt = self.build_prompt(
            question,
            context
        )

        for chunk in self.llm.stream(prompt):

            if chunk.content:
                yield chunk.content

    def rerank(self, question, docs):

        if len(docs) <= 8:
            return docs

        chunks = []

        for i, doc in enumerate(docs):

            text = doc.page_content[:800]

            chunks.append(
                f"""
Chunk {i}

{text}
"""
            )

        prompt = f"""
Question:

{question}

Select up to 8 chunks which directly answer the question.

Ignore:

- Module Objectives
- Knowledge Check
- Exercise
- generic introductions

Return only numbers separated by commas.

Examples:

0,3,5

Documents:

{"".join(chunks)}
"""

        response = self.llm.invoke(prompt)

        answer = response.content.strip()

        print()
        print("========== RERANK ==========")
        print(answer)
        print("============================")
        print()

        try:

            indexes = [
                int(x.strip())
                for x in answer.split(",")
            ]

            return [
                docs[i]
                for i in indexes
                if i < len(docs)
            ]

        except:

            return docs[:8]
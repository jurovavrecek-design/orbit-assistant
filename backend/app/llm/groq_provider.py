from langchain_groq import ChatGroq

from app.llm.base_provider import BaseLLMProvider


class GroqProvider(BaseLLMProvider):

    def __init__(self):

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )

    def rerank(self, question, docs):

        chunks = []

        for i, doc in enumerate(docs):

            text = doc.page_content[:1000]

            chunks.append(
                f"""
Chunk {i}

Source:
{doc.metadata.get("source")}

Content:
{text}
"""
            )

        prompt = f"""
Question:

{question}

Below are document chunks.

Select the 5 most relevant chunks.

Return ONLY chunk numbers separated by commas.

Documents:

{"".join(chunks)}
"""

        response = self.llm.invoke(prompt)

        try:

            indexes = [
                int(x.strip())
                for x in response.content.split(",")
            ]

        except:

            indexes = list(range(5))

        return [
            docs[i]
            for i in indexes
            if i < len(docs)
        ]

    def build_prompt(self, question, context):

        return f"""
You are ORBIT Assistant.

Use ONLY the supplied documentation.

If documentation does not contain the answer, say so.

Never invent:

- processes
- field names
- IDs
- dates

Use bullet points.

Answer in the same language as the question.

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
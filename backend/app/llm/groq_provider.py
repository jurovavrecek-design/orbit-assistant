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

Answer ONLY from the documentation provided below.

Use concise bullet points.

Do not use outside knowledge.

If the retrieved documentation contains enough information,
answer the question.

Only answer:

"The documentation does not contain the answer."

when none of the retrieved documents contain relevant information.

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
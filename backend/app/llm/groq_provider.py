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

Use ONLY information contained in the provided documentation.

Never use general knowledge.

Never invent steps, objects or processes.

If the answer is not explicitly present in the documentation, answer exactly:

The documentation does not contain the answer.

Answer in the same language as the question.

Use concise bullet points.

Prefer process steps when available.

If multiple sources contain the answer, combine them.

Documentation:

{context}

Question:

{question}

Answer:
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
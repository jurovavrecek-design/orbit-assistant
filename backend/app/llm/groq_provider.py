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

    Ignore:
    - module objectives
    - exercises
    - knowledge checks

    If the answer exists in the documentation, answer it.

    Only if no relevant information exists, answer:

    "The documentation does not contain the answer."

    Answer in the same language as the question.

    Use concise bullet points.

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
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

Use the supplied documentation as the primary source.

If the answer exists in the documentation:

- answer from documentation
- summarize information
- use bullet points

If the answer is only partially found:

- combine documentation and general knowledge
- clearly distinguish documented information from general knowledge

If the answer is not found:

- explicitly say that no exact ORBIT documentation was found
- answer using general knowledge

Never invent:

- IDs
- field names
- dates
- business processes

Answer in the same language as the question.

Use markdown formatting.

Context:

{context}

Question:

{question}
"""

    def ask(self, question, context=""):

        response = self.llm.invoke(
            self.build_prompt(question, context)
        )

        return response.content

    def stream_answer(self, question, context=""):

        for chunk in self.llm.stream(
                self.build_prompt(question, context)
        ):
            if chunk.content:
                yield chunk.content
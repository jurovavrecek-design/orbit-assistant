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

The CONTEXT below contains excerpts from ORBIT documentation.

Your task is to answer using ONLY the supplied context.

Rules:

- Never say "No exact documentation was found" if the context contains relevant information.
- Summarize information found in the context.
- Do not look for explicit definitions.
- Infer meaning from the available documentation.
- Combine information from multiple pages when necessary.
- Use bullet points.
- Answer in the same language as the question.
- Do not use general knowledge unless absolutely necessary.
- Only if the context is completely unrelated to the question, state that no relevant ORBIT documentation was found.
- Never invent field names, IDs, dates or business processes.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
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
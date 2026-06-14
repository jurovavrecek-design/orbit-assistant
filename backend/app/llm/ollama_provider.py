from langchain_ollama import ChatOllama

from app.llm.base_provider import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):

    def __init__(self):

        self.llm = ChatOllama(
            model="qwen3:8b",
            base_url="http://host.docker.internal:11434",
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
- clearly distinguish both

If the answer is not found:

- explicitly say no exact ORBIT documentation was found

Never invent:

- IDs
- field names
- dates
- business processes

Answer in the same language as the question.

Use markdown.

Context:

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
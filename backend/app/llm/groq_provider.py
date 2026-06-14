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

            text = doc.page_content[:500]

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

Good chunks:
- directly answer the question
- contain steps, definitions, or instructions

Bad chunks:
- module objectives
- knowledge checks
- generic introductions
- unrelated topics

Select ONLY chunks that directly answer the question.

Ignore generic modules, objectives and unrelated topics.

Return ONLY chunk numbers separated by commas.

Examples:

Question:
How to create a report?

Good chunk:
Create Report
Add Report Filters

Bad chunks:
Task Management
Time Allocation
Medical Interaction

If only one chunk is relevant, return one number.

If no chunk answers the question, return NONE.

Documents:

{"".join(chunks)}
"""

        response = self.llm.invoke(prompt)

        response_text = response.content.strip()

        print()
        print("========== RERANK RESPONSE ==========")
        print(response_text)
        print("====================================")
        print()

        if response_text.upper() == "NONE":
            return []

        try:

            indexes = [
                int(x.strip())
                for x in response_text.split(",")
            ]

        except:

            indexes = list(range(3))

        return [
            docs[i]
            for i in indexes
            if i < len(docs)
        ]

    def build_prompt(self, question, context):

        return f"""
You are ORBIT Assistant.

Use ONLY information contained in the documentation.

Do not infer business processes.

Do not use general knowledge.

If the answer is not present in the documentation, explicitly say:

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
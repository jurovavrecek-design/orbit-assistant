def rerank(self, question, docs):

    if not docs:
        return []

    docs_text = ""

    for i, doc in enumerate(docs):

        docs_text += (
            f"\nDOCUMENT {i}\n"
            f"{doc.page_content[:1000]}\n"
        )

    response = self.client.chat.completions.create(
        model=self.model,
        messages=[
            {
                "role": "system",
                "content":
                """
Return only document numbers separated by commas.

Example:
0,2,4

If nothing is relevant return NONE.
"""
            },
            {
                "role": "user",
                "content":
                f"""
Question:

{question}

Documents:

{docs_text}
"""
            }
        ],
        temperature=0
    )

    result = (
        response.choices[0]
        .message.content
        .strip()
    )

    print()
    print("========== RERANK RESPONSE ==========")
    print(result)
    print("====================================")

    if "NONE" in result.upper():
        return docs[:5]

    try:

        indexes = [
            int(x.strip())
            for x in result.split(",")
        ]

        reranked_docs = []

        for i in indexes:

            if 0 <= i < len(docs):
                reranked_docs.append(
                    docs[i]
                )

        if reranked_docs:
            return reranked_docs

        return docs[:5]

    except Exception:

        return docs[:5]
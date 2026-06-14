import pickle
import re

from rank_bm25 import BM25Okapi


def tokenize(text):

    return re.findall(
        r"\b[a-z0-9]+\b",
        text.lower()
    )


class BM25Retriever:

    def __init__(self, path="./bm25.pkl"):

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.documents = data["documents"]
        self.tokenized_docs = data["tokenized_docs"]

        self.bm25 = BM25Okapi(
            self.tokenized_docs
        )

    def search(self, question, k=10):

        query_tokens = tokenize(question)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked = sorted(
            zip(scores, self.documents),
            reverse=True,
            key=lambda x: x[0]
        )

        print()
        print("========== BM25 ==========")

        for score, doc in ranked[:20]:
            print(
                round(score, 3),
                doc.metadata
            )

        print("==========================")
        print()

        return [
            doc
            for score, doc in ranked[:k]
            if score > 0
        ]
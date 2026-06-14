import pickle

from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, path="./bm25.pkl"):

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.documents = data["documents"]
        self.tokenized_docs = data["tokenized_docs"]

        self.bm25 = BM25Okapi(
            self.tokenized_docs
        )

    def search(self, question, k=20):

        query_tokens = question.lower().split()

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked = sorted(
            zip(scores, self.documents),
            reverse=True,
            key=lambda x: x[0]
        )

        return [
            doc
            for _, doc in ranked[:k]
        ]
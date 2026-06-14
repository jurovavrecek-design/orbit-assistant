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

        results = []
        seen_titles = set()

        print()
        print("========== BM25 ==========")

        for score, doc in ranked:

            if score <= 0:
                continue

            title = doc.page_content.split("\n")[0].strip()

            if title in seen_titles:
                continue

            seen_titles.add(title)

            print(round(score, 3), doc.metadata)

            results.append(doc)

            if len(results) == k:
                break

        print("==========================")

        return results
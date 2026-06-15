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

    def search(self, question, k=5):

        q = question.lower()

        #
        # Query expansion
        #

        synonyms = {
            "create call": "record call",
            "new call": "record call",
            "log call": "record call",
            "create account": "new account",
            "anf": "service provider",
            "edetail": "edetailing",
            "clm": "edetailing",
            "meeting": "event",
            "survey": "survey target"
        }

        for key, value in synonyms.items():

            if key in q:
                q += " " + value

        if "call" in q:
            q += " sales call medical interaction"

        query_tokens = tokenize(q)

        #
        # BM25 scores
        #

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked = sorted(
            zip(scores, self.documents),
            reverse=True,
            key=lambda x: x[0]
        )

        #
        # Rule-based rerank
        #

        boosted = []

        skip_keywords = [
            "module objectives",
            "knowledge check",
            "exercise",
            "understand",
            "be able to"
        ]

        for score, doc in ranked:

            text = doc.page_content.lower()

            title = (
                doc.metadata
                .get("title", "")
                .lower()
            )

            #
            # Skip generic training slides
            #

            if any(
                keyword in text
                for keyword in skip_keywords
            ):
                continue

            bonus = 0

            #
            # Exact token matches
            #

            for token in query_tokens:

                if token in text:
                    bonus += 0.4

            #
            # Strong title bonus
            #

            for token in query_tokens:

                if token in title:
                    bonus += 2

            #
            # Prefer PPT
            #

            if doc.metadata.get("type") == "pptx":
                bonus += 1

            #
            # Slightly penalize PDF
            #

            if doc.metadata.get("type") == "pdf":
                bonus -= 1

            boosted.append(
                (
                    score + bonus,
                    doc
                )
            )

        boosted.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        print()
        print("========== BM25 ==========")

        for score, doc in boosted[:20]:

            print(
                round(score, 2),
                doc.metadata
            )

        print("==========================")
        print()

        return [
            doc
            for _, doc in boosted[:k]
        ]
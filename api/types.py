from __future__ import annotations

import dataclasses
from typing import Dict, List


@dataclasses.dataclass
class DocumentWithScore:
    title: str
    text: str
    score: float

    def to_response_dict(self):
        return {
            "title": self.title,
            "text": self.text,
            "score": self.score,
        }

    @staticmethod
    def from_dict(dict_object: Dict):
        return DocumentWithScore(
            title=dict_object["title"],
            text=dict_object["text"],
            score=dict_object["score"],
        )


@dataclasses.dataclass
class RetrievalResult:
    query: str
    documents_with_score: List[DocumentWithScore]

    def to_response_dict(self) -> Dict:
        return {
            "query": self.query,
            "documentsWithScore": [
                doc.to_response_dict() for doc in self.documents_with_score
            ],
        }

    @staticmethod
    def from_dict(dict_object: Dict) -> RetrievalResult:
        return RetrievalResult(
            query=dict_object["query"],
            documents_with_score=[
                DocumentWithScore.from_dict(doc)
                for doc in dict_object["documentsWithScore"]
            ],
        )

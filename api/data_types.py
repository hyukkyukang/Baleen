from __future__ import annotations

from typing import *

import hkkang_utils.data as data_utils


@data_utils.dataclass
class DocumentWithScore:
    title: str
    text: str
    score: float


@data_utils.dataclass
class RetrievalResult:
    query: str
    documents_with_score: List[DocumentWithScore]

from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

from pydantic import BaseModel, Field, Extra
from pydantic import BaseConfig

from haystack.schema import Answer, Document


BaseConfig.arbitrary_types_allowed = True
BaseConfig.json_encoders = {np.ndarray: lambda x: x.tolist(), pd.DataFrame: lambda x: x.to_dict(orient="records")}


PrimitiveType = Union[str, int, float, bool]


class PipelineHyperParams(BaseModel):
    faq_embedding_dim: int = 384
    extractive_embedding_dim: int = 768
    extractive_reader_option: str = "deepset/roberta-base-squad2"
    faq_retriever_option: str = "sentence-transformers/all-MiniLM-L6-v2"
    faq_similarity_function: str = "cosine"
    extractive_similarity_function: str = "dot_product"
    top_k: int = 5


class QuestionAnswerPair(BaseModel):
    question: str
    answer: str
    alternative_question: str
    approved: bool  # e.g. validate user suggested Q&A pairs
    game: str


class RequestBaseModel(BaseModel):
    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid


class QueryRequest(RequestBaseModel):
    query: str
    params: Optional[dict] = None
    debug: Optional[bool] = False


class FilterRequest(RequestBaseModel):
    filters: Optional[Dict[str, Union[PrimitiveType, List[PrimitiveType], Dict[str, PrimitiveType]]]] = None


class CreateLabelSerialized(RequestBaseModel):
    id: Optional[str] = None
    query: str
    document: Document
    is_correct_answer: bool
    is_correct_document: bool
    origin: Literal["user-feedback", "gold-label"]
    answer: Optional[Answer] = None
    no_answer: Optional[bool] = None
    pipeline_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    meta: Optional[dict] = None
    filters: Optional[dict] = None


class QueryResponse(BaseModel):
    query: str
    answers: List[Answer] = []
    documents: List[Document] = []
    debug: Optional[Dict] = Field(None, alias="_debug")

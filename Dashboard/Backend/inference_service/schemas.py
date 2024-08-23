from uuid import UUID

from pydantic import BaseModel
from typing import List

class VisionModelSchema(BaseModel):
    model_name: str
    model_id: str
    model_path: str

class VisionModelSchemaIn(VisionModelSchema):
    pass


class VisionModelSchemaOut(VisionModelSchema):
    id: UUID
    updated_at: int
    created_at: int


class InferenceResultSchema(BaseModel):
    labels: List[str]
    results: List[float]
    binary_results: List[int]
    percentage_coverage: List[float]


class InferenceResultSchemaIn(InferenceResultSchema):
    model_id: str
    img_id: str

class InferenceResultSchemaOut(InferenceResultSchema):
    id: UUID
    updated_at: int
    created_at: int 
    model_id: str
    img_id: str
    img_field: str

class ResultsSchema(BaseModel):
    img_id: str
    model_list: List[VisionModelSchemaOut]
    results: List[InferenceResultSchemaOut]

class ResultsSchemaIn(ResultsSchema):
    pass

class ResultsSchemaOut(ResultsSchema):
    id: UUID
    updated_at: int
    created_at: int



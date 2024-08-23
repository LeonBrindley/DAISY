from uuid import UUID

from pydantic import BaseModel


class ProductSchema(BaseModel):
    name: str
    description: str

class ProductSchemaIn(ProductSchema):
    pass

class ProductSchemaOut(ProductSchema):
    id: UUID
    updated_at: int
    created_at: int

# Actual Image Schema

class ImageSchema(BaseModel):
    img_url: str
    img_field: str
    coordinates: list[float]
    time: int

class ImageSchemaIn(ImageSchema):
    pass

class ImageSchemaOut(ImageSchema):
    id: UUID
    updated_at: int
    created_at: int

class InferenceProgressSchema(BaseModel):
    currently_training: bool
    percentage_completed: float
        
class InferenceProgressSchemaOut(InferenceProgressSchema):
    pass
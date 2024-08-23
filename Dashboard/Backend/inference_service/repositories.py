from fastapi import HTTPException
import time
import uuid
from typing import Dict, Any, Union, List
import json

from inference_service.schemas import VisionModelSchema, VisionModelSchemaIn, VisionModelSchemaOut, InferenceResultSchema, InferenceResultSchemaIn, InferenceResultSchemaOut, ResultsSchema, ResultsSchemaIn, ResultsSchemaOut
from inference_service.tables import VisionModel, InferenceResult, Results

# class ImageRepository:
#     table: ImageTable = ImageTable
#     schema_out: ImageSchemaOut = ImageSchemaOut

#     @staticmethod
#     def _preprocess_create(values: Dict[str, Any]) -> Dict[str, Any]:
#         timestamp_now = int(time.time())
#         values["id"] = str(uuid.uuid4())
#         values["created_at"] = timestamp_now
#         values["updated_at"] = timestamp_now
#         # Serialize the coordinates list into a JSON string
#         values["coordinates"] = json.dumps(values["coordinates"])
#         print(f"Debug: Preprocessed data for creation: {values}")  # Debug logging
#         return values
    
#     @classmethod
#     def create(cls, image_in: ImageSchemaIn) -> ImageSchemaOut:
#         data = cls._preprocess_create(image_in.dict())
#         model = cls.table(**data)
#         print(f"Debug: Data to be saved to table: {model.attribute_values}")  # Debug logging
#         try:
#             print(f"Debug: Attempting to save model: {model}")
#             model.save()
#         except Exception as e:
#             print(f"Error: Failed to save model. Exception: {e}")
#             raise
#         # Deserialize the coordinates after retrieving from the model
#         result_data = model.attribute_values
#         result_data["coordinates"] = json.loads(result_data["coordinates"])
#         return cls.schema_out(**result_data)
    
#     @classmethod
#     def get_img_url_by_id(cls, entry_id: str) -> str:
#         try:
#             # Retrieve the item from the table using the provided entry ID
#             model = cls.table.get(entry_id)
#             img_url = model.img_url
#             print(f"Debug: Retrieved img_url for entry_id {entry_id}: {img_url}")
#             return img_url
#         except Exception as e:
#             print(f"Error: Failed to retrieve img_url for entry_id {entry_id}. Exception: {e}")
#             raise

#     @classmethod
#     def get_all(cls) -> List[ImageSchemaOut]:
#         try:
#             # Scan the table to retrieve all items
#             scan_results = cls.table.scan()
#             images = []
#             for item in scan_results:
#                 data = item.attribute_values
#                 # Deserialize the coordinates after retrieving from the model
#                 data["coordinates"] = json.loads(data["coordinates"])
#                 images.append(cls.schema_out(**data))
#             return images
#         except Exception as e:
#             print(f"Error: Failed to retrieve all images. Exception: {e}")
#             raise HTTPException(status_code=500, detail="Failed to retrieve images.")

#     @classmethod
#     def get(cls, entry_id: Union[str, uuid.UUID]) -> ImageSchemaOut:
#         print("Debug: Retrieving entry with ID:", entry_id)
#         model = cls.table.get(str(entry_id))
#         print("Debug: Retrieved model attribute values:", model.attribute_values)
#         return cls.schema_out(**model.attribute_values)
    
#     @classmethod
#     def delete_all_images(cls):
#         try:
#             # Fetch the item to ensure it exists before deleting
#             scan_results = cls.table.scan()
#             for item in scan_results:
#                 item.delete()
#             print(f"Deleted all from DynamoDB")
#         except Exception as e:
#             print(f"Error deleting all: {e}")
#             raise
    

    
#     @classmethod
#     def delete(cls, entry_id: Union[str, uuid.UUID]):
#         try:
#             # Fetch the item to ensure it exists before deleting
#             model = cls.table.get(str(entry_id))
#             model.delete()
#             print(f"Deleted {entry_id} from DynamoDB")
#         except Exception as e:
#             print(f"Error deleting {entry_id}: {e}")
#             raise

    

# class ProductRepository:
#     table: ProductTable = ProductTable
#     schema_out: ProductSchemaOut = ProductSchemaOut

#     @staticmethod
#     def _preprocess_create(values: Dict[str, Any]) -> Dict[str, Any]:
#         timestamp_now = time.time()
#         values["id"] = str(uuid.uuid4())
#         values["created_at"] = timestamp_now
#         values["updated_at"] = timestamp_now
#         return values

#     @classmethod
#     def create(cls, product_in: ProductSchemaIn) -> ProductSchemaOut:
#         data = cls._preprocess_create(product_in.dict())
#         model = cls.table(**data)
#         model.save()
#         return cls.schema_out(**model.attribute_values)

#     @classmethod
#     def get(cls, entry_id: Union[str, uuid.UUID]) -> ProductSchemaOut:
#         model = cls.table.get(str(entry_id))
#         return cls.schema_out(**model.attribute_values)
from uuid import UUID
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os 
from minio import Minio
from minio.error import S3Error
from starlette import status
from typing import List
from fastapi import FastAPI, UploadFile, File
import zipfile
import io
from fastapi.responses import StreamingResponse
from fastapi.responses import Response
from fastapi.responses import Response
import os

import base64

from app.repositories import ProductRepository, ImageRepository
from app.schemas import ProductSchemaIn, ProductSchemaOut, ImageSchemaOut, ImageSchemaIn, InferenceProgressSchemaOut
from app.tables import InferenceProgress

app = FastAPI()
# main.py (continued)

origins = [
    "http://localhost:3000",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8081",
]


minio_client = Minio(
    "minio:9000",
    access_key=os.getenv("MINIO_ROOT_USER", "admin"),
    secret_key=os.getenv("MINIO_ROOT_PASSWORD", "password"),
    secure=False
)

# Ensure the bucket exists
bucket_name = "imgs"
bucket_predictions =  "predictions"
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Adjust as necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "ngrok-skip-browser-warning"],
)

def validate_image_data(image_data: ImageSchemaIn):
    if not image_data.img_url:
        raise ValueError("img_url must be provided.")
    if not isinstance(image_data.coordinates, list) or len(image_data.coordinates) != 2:
        raise ValueError("coordinates must be a list of two integers.")
    if not all(isinstance(coord, float) for coord in image_data.coordinates):
        raise ValueError("Each coordinate must be an integer.")
    if not isinstance(image_data.time, int):
        raise ValueError("time must be an integer.")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded ZIP file locally
        zip_file_path = f"/tmp/{file.filename}"
        with open(zip_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Extract ZIP file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                # Ignore directories like __MACOSX and other irrelevant entries
                if member.is_dir() or member.filename.startswith("__MACOSX"):
                    continue
                
                # Extract files
                extracted_path = zip_ref.extract(member, "/tmp")
                
                # Remove leading directories from the filename
                file_name = os.path.basename(extracted_path)
                
                # Upload each file to MinIO
                with open(extracted_path, "rb") as f:
                    minio_client.put_object(
                        bucket_name,
                        file_name,
                        f,
                        os.path.getsize(extracted_path)
                    )
        
        # Cleanup temporary files
        os.remove(zip_file_path)
        
        return {"message": "Files uploaded and extracted successfully"}
    except S3Error as err:
        print(err)
        return {"error": str(err)}
    except Exception as e:
        print(e)
        return {"error": str(e)}


@app.get("/get_image_file/{image_id}")
def get_image_file(image_id: str):
    try:
        # Retrieve image filename from the database using the provided image_id
        image_name = ImageRepository.get_img_url_by_id(image_id)
        
        # Fetch the image from MinIO
        response = minio_client.get_object(bucket_name, image_name)
        image_bytes = response.read()
        response.close()
        response.release_conn()

        # Encode image as Base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        return {"image": base64_image}
    except S3Error as err:
        print(err)
        raise HTTPException(status_code=404, detail=f"Image {image_name} not found in MinIO")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/get_prediction_image_file/{prediction_id}")
def get_prediction_image_file(prediction_id: str, label: str):
    try:
        # Retrieve image filename from the database using the provided image_id

        image_path = prediction_id + "/" + label + ".jpg"

        print(image_path)

        response = minio_client.get_object(bucket_predictions, image_path)
        image_bytes = response.read()
        response.close()
        response.release_conn()

        # Encode image as Base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        return {"image": base64_image}
    except S3Error as err:
        print(err)
        raise HTTPException(status_code=404, detail=f"Image {image_path} not found in MinIO")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error")



@app.post(
    "/insert_img", 
    status_code=status.HTTP_201_CREATED,
    response_model=ImageSchemaOut
)
def create_image(image_in: ImageSchemaIn) -> ImageSchemaOut:
    try:
        validate_image_data(image_in)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    image_out = ImageRepository.create(image_in)
    return image_out


@app.get(
    "/get_img/{image_id}",
    status_code=status.HTTP_200_OK,
    response_model=ImageSchemaOut,
)
def create_image(image_id: UUID) -> ImageSchemaOut:
    image_out = ImageRepository.get(image_id)
    return image_out

@app.get(
    "/get_all_images",
    status_code=status.HTTP_200_OK,
    response_model=List[ImageSchemaOut],
)
def get_all_images() -> List[ImageSchemaOut]:
    images = ImageRepository.get_all()
    return images


@app.get(
    '/get_inference_progress',
    status_code = status.HTTP_200_OK,
    response_model= InferenceProgressSchemaOut
)
def get_progress() -> InferenceProgressSchemaOut:
    inference_progress_list = InferenceProgress.scan()
    for result in inference_progress_list:
        inference_progress = InferenceProgress.get(result.id)

    return {
        "currently_training": inference_progress.currently_training,
        "percentage_completed": inference_progress.percentage_completed
    }

@app.delete("/delete_img/{image_id}")
def delete_image(image_id: str):
    try:
        # Attempt to delete the 
        # object from the MinIO bucket
        img_name = ImageRepository.get_img_url_by_id(image_id)
        
        minio_client.remove_object(bucket_name, img_name)
        print(f"Deleted {img_name} from MinIO")

        # Attempt to delete the corresponding entry from the DynamoDB table
        ImageRepository.delete(image_id)
        print(f"Deleted {image_id} from DynamoDB")

        return {"message": f"Image {image_id} deleted successfully"}
    except S3Error as err:
        print(err)
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found in MinIO")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found in DynamoDB")


@app.delete("/delete_all_images")
def delete_all_images():
    try:
        # List all objects in the bucket
        objects = minio_client.list_objects(bucket_name)
        for obj in objects:
            object_name = obj.object_name
            
            # Print the object name for debugging
            print(f"Deleting object {object_name} from MinIO and DynamoDB")
            
            # Remove the object from MinIO
            minio_client.remove_object(bucket_name, object_name)
            
        # Delete each object from MinIO and DynamoDB
        ImageRepository.delete_all_images()
        print(f"Deleted all from MinIO and DynamoDB")

        return {"message": "All images deleted successfully"}
    except S3Error as err:
        print(err)
        raise HTTPException(status_code=500, detail="Error deleting all images from MinIO")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error deleting all images from DynamoDB")

@app.post(
    "/v1/products",
    status_code=status.HTTP_201_CREATED,
    response_model=ProductSchemaOut,
)
def create_product(product_in: ProductSchemaIn) -> ProductSchemaOut:
    product_out = ProductRepository.create(product_in)
    return product_out


@app.get(
    "/v1/products/{product_id}",
    status_code=status.HTTP_200_OK,
    response_model=ProductSchemaOut,
)
def create_product(product_id: UUID) -> ProductSchemaOut:
    product_out = ProductRepository.get(product_id)
    return product_out
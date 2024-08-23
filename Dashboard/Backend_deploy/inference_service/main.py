from uuid import UUID

from fastapi import FastAPI, UploadFile, File, HTTPException, status
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
import os
from uuid import uuid4
import datetime
import base64
from torchvision import transforms
from PIL import Image
from io import BytesIO
import io
import torch
import torchvision.models as models
from inference_service.omar_dir.alexnet import load as load_AlexNet
from inference_service.schemas import InferenceResultSchema, VisionModelSchema, VisionModelSchemaIn, VisionModelSchemaOut, InferenceResultSchemaIn, InferenceResultSchemaOut, ResultsSchemaIn, ResultsSchemaOut
from inference_service.tables import VisionModel, InferenceResult, Results, ImageTable, InferenceProgress
import inference_service.omar_dir.inference as inference
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import re 
import boto3
from botocore.exceptions import ClientError
import numpy as np


app = FastAPI()

origins = [
    
    "http://localhost:3000",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8081",
    "https://sensor-c.web.app"
]


config = {
    "AWS_ACCESS_KEY_ID": 'AKIAZI2LBYEBCNVSRWM2',
    "AWS_SECRET_ACCESS_KEY": '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',
    "AWS_REGION": 'eu-north-1'
}

try:
    s3_client = boto3.client(
      's3',
      region_name=config['AWS_REGION'],
      aws_access_key_id=config["AWS_ACCESS_KEY_ID"],
      aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"],
    )

    # Example operation: List buckets
    response = s3_client.list_buckets()
    print("Buckets:", [bucket["Name"] for bucket in response["Buckets"]])
except ClientError as e:
    print(f"ClientError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")



# Ensure the bucket exists
bucket_name_imgs = "sensor-cdt-daisy-imgs"
bucket_name_models = "sensor-cdt-daisy-models"
bucket_name_prediction_imgs = "sensor-cdt-daisy-predictions"


try:
    s3_client.head_bucket(Bucket=bucket_name_imgs)
    print(f"Bucket {bucket_name_imgs} already exists.")
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == '404':
        # If the bucket does not exist, create it
        s3_client.create_bucket(
            Bucket=bucket_name_imgs,
            CreateBucketConfiguration={'LocationConstraint': config["AWS_REGION"]})
        print(f"Bucket {bucket_name_imgs} created successfully.")
    else:
        # Handle other potential errors
        print(f"Unexpected error: {e}")

try:
    s3_client.head_bucket(Bucket=bucket_name_models)
    print(f"Bucket {bucket_name_models} already exists.")
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == '404':
        # If the bucket does not exist, create it
        s3_client.create_bucket(
            Bucket=bucket_name_models,
            CreateBucketConfiguration={'LocationConstraint': config["AWS_REGION"]})
        print(f"Bucket {bucket_name_models} created successfully.")
    else:
        # Handle other potential errors
        print(f"Unexpected error: {e}")

try:
    s3_client.head_bucket(Bucket=bucket_name_prediction_imgs)
    print(f"Bucket {bucket_name_prediction_imgs} already exists.")
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == '404':
        # If the bucket does not exist, create it
        s3_client.create_bucket(
            Bucket=bucket_name_prediction_imgs,
            CreateBucketConfiguration={'LocationConstraint': config["AWS_REGION"]})
        print(f"Bucket {bucket_name_prediction_imgs} created successfully.")
    else:
        # Handle other potential errors
        print(f"Unexpected error: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Adjust as necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_coordinates(filename):
    pattern = re.compile(r'_(\d+)_(\d+)\.jpg$')
    match = pattern.search(filename)
    if match:
        x, y = map(int, match.groups())
        return x, y
    return None

@app.get("/")
async def read_root():
    return {"message": "Inference Service Running"}


@app.get(
        "/inference_results",
        response_model=List[InferenceResultSchemaOut],
        status_code=status.HTTP_200_OK
        )
async def get_all_inference_results():

    try:
        inference_results = list(InferenceResult.scan())

        inference_result_dicts = [
            {
                "id": inference_result.id,
                "model_id": inference_result.model_id,
                "img_id": inference_result.img_id,
                "img_field": inference_result.img_field,
                "labels": inference_result.labels,
                "results": inference_result.results,
                "binary_results": inference_result.binary_results,
                "percentage_coverage": inference_result.percentage_coverage,
                "created_at": inference_result.created_at,
                "updated_at": inference_result.updated_at
            }
            for inference_result in inference_results
        ]
        return inference_result_dicts
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error")




@app.get(
    "/models",
    response_model=List[VisionModelSchemaOut],
    status_code=status.HTTP_200_OK
)
async def get_all_models():
    try:
        models = list(VisionModel.scan())
        # Convert each model to a dictionary
        model_dicts = [
            {
                "id": model.id,
                "model_name": model.model_name,
                "model_id": model.model_id,
                "model_path": model.model_path,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
            }
            for model in models
        ]
        return model_dicts
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error")




@app.delete(
    "/models/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_model(model_id: str):
    try:
        model = VisionModel.get(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Delete the model file from MinIO
        
        s3_client.remove_object(Bucket=bucket_name_models, Key=model.model_path.split('/')[-1])
        
        # Delete the model record from the table
        model.delete()
        
        return {"message": "Model deleted successfully"}
    except S3Error as err:
        print(err)
        raise HTTPException(status_code=500, detail="Error interacting with MinIO")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error")


# Upload a new model to the Minio bucket

@app.post(
    "/upload_model",
    response_model=VisionModelSchemaOut,
    status_code=status.HTTP_201_CREATED
)
async def upload_model(file: UploadFile = File(...), model_name: str = None):
    try:
        zip_file_path = f"/tmp/{file.filename}"
        with open(zip_file_path, "wb") as buffer:
            buffer.write(await file.read())

        with open(zip_file_path, "rb") as f:

            s3_client.put_object(
                Bucket=bucket_name_models,
                Key=file.filename,  # Use the original file name
                Body=f,
            )
            # Cleanup temporary file
        os.remove(zip_file_path)


        model_id = str(uuid4())
        model_path = f"{bucket_name_models}/{file.filename}"
        # created_at = updated_at = int(datetime.now().timestamp())
        created_at = updated_at = int(datetime.datetime.now().timestamp())


        print("Created model with ID:", model_id)
        print('Created at:', created_at)

        # Save the model details in the VisionModel table
        model = VisionModel(
            id=model_id,
            model_name=model_name or file.filename,
            model_id=model_id,
            model_path=model_path,
            created_at=created_at,
            updated_at=updated_at
        )
        model.save()
        
        # Return the saved model details
        return {
            "id": model_id,
            "model_name": model_name or file.filename,
            "model_id": model_id,
            "model_path": model_path,
            "created_at": created_at,
            "updated_at": updated_at
        }
        
        return {"message": "ZIP file uploaded successfully"}
    except S3Error as err:
        print(err)
        return {"error": str(err)}
    except Exception as e:
        print(e)
        return {"error": str(e)}



@app.post("/predict_all", status_code=status.HTTP_200_OK)
async def predict_all():

    inference_progress_list = InferenceProgress.scan()

    for result in inference_progress_list:
        inference_progress = InferenceProgress.get(result.id)
        inference_progress.currently_training = True
        inference_progress.percentage_completed = 0.0
        inference_progress.save()

    try:
        image_list = list(ImageTable.scan())
        model_list = list(VisionModel.scan())
        inference_past_results = list(InferenceResult.scan())

        image_list_length = len(image_list)
        model_list_length = len(model_list)

        n_predictions = image_list_length * model_list_length
        counter = 0

        for image in image_list:
            response = s3_client.get_object(Bucket=bucket_name_imgs, Key=image.img_url)
            image_data = response["Body"].read()
            response["Body"].close()

            for model_object in model_list:
                # model_id = model

                model_path = model_object.model_path.split('/')[-1]

                skip = False
                for result in inference_past_results:
                    if result.model_id == model_path and result.img_id == image.img_url:
                        skip = True
                if skip:
                    print("Skipping inference: result already existent")

                else:

                    print('The model path is: ', model_path)
                    inference_id = str(uuid4())

                    model_response = s3_client.get_object(Bucket=bucket_name_models, Key=model_path)
                    model_data = model_response["Body"].read()
                    model_response["Body"].close()

                    model_id = model_path
                    print(model_id)

                    with zipfile.ZipFile(BytesIO(model_data)) as zip_ref:
                        zip_ref.extractall("/tmp/")

                    model_folder = model_id.split('.zip')[0]
                    print("model folder", model_folder)

                    model_file_name = os.listdir("/tmp/" + model_folder +"/")

                    print("model file name", model_file_name)
                    
                    model_file_path = "/tmp/" + model_folder + "/" + model_file_name[0]
                    print("model file path", model_file_path)

                    image_path = "/tmp/image.jpg"
                    with open(image_path, "wb") as image_file:
                        image_file.write(image_data)

                    model = inference.load_model(model_file_path)
                    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))

                    paths, full_predictions, probabilities = inference.predict(model, image_path, "/tmp/work")

                            # saving prediction images
                    rect_size = 224  # Original rectangle size
                    small_rect_size = 100 # Size of smaller segments
                    labels = ["grass", "clover", "soil", "dung"]    
                    img = mpimg.imread('/tmp/image.jpg')  # Load the image using matplotlib
                    max_x = max(extract_coordinates(fn)[0] for fn in paths) + rect_size
                    max_y = max(extract_coordinates(fn)[1] for fn in paths) + rect_size


                    percentage_coverage = []

                    # Loop through each class
                    for class_idx, label in enumerate(labels):

                        
                        # Create probability map initialized to 0
                        probability_map = np.zeros((max_y, max_x))
                        segment_map = np.full((max_y, max_x), -1)  # Track which segment contributed to the max probability

                        # First pass: Divide rectangles into smaller segments and update probability map
                        seg_num = 0
                        for segment_idx, (filename, prob) in enumerate(zip(paths, full_predictions)):
                            if seg_num > 1:
                                coords = extract_coordinates(filename)
                                if coords:
                                    x, y = coords


                                    # Divide the rectangle into smaller cells
                                    for i in range(0, rect_size, small_rect_size):
                                        for j in range(0, rect_size, small_rect_size):
                                            cell_x = x + j
                                            cell_y = y + i

                                            # Iterate over the pixels in the smaller cell
                                            for m in range(cell_y, cell_y + small_rect_size):
                                                for n in range(cell_x, cell_x + small_rect_size):
                                                    # Ensure we are within image bou
                                                    if 0 <= m < max_y and 0 <= n < max_x:
                                                        # Update the probability map if this probability is higher

                                                        if prob[class_idx] > probability_map[m, n]:
                                                            probability_map[m, n] = prob[class_idx]
                                                            segment_map[m, n] = segment_idx  # Track the segment contributing this max
                                                            # print(prob[class_idx], probability_map[m,n])
                                    
                            seg_num +=1
                        print(seg_num)
                        
                        # Calculate the percentage of pixels with a probability over 0.9
                        total_pixels = probability_map.size
                        high_prob_pixels = np.sum(probability_map > 0.9)
                        percentage_high_prob = (high_prob_pixels / total_pixels) * 100
                        percentage_coverage.append(percentage_high_prob)
                        print(f"Percentage of pixels with probability > 0.9 for {label}: {percentage_high_prob:.2f}%")

                        # Second pass: Plot only the parts of rectangles that contributed to the max probability
                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.imshow(img, extent=[0, max_x, max_y, 0])

                        for segment_idx, (filename, prob) in enumerate(zip(paths, full_predictions)):
                            coords = extract_coordinates(filename)
                            if coords:
                                x, y = coords

                                # Plot smaller cells only where they contributed to the max probability
                                for i in range(0, rect_size, small_rect_size):
                                    for j in range(0, rect_size, small_rect_size):
                                        cell_x = x + j
                                        cell_y = y + i

                                        # Check if this smaller cell contributed to the max probability
                                        mask = (segment_map[cell_y:cell_y + small_rect_size, cell_x:cell_x + small_rect_size] == segment_idx)
                                        if np.any(mask):  # Only proceed if this cell contributed to the max

                                            prob_value = prob[class_idx]
                                            color = (1, 0, 0, prob_value)  # RGBA with red intensity based on the probability value
                                            alpha = (prob_value -0.3) if prob_value > 0.3 else 0.0
                                            rect = plt.Rectangle((cell_x, cell_y), small_rect_size, small_rect_size,
                                                                facecolor=color, alpha=alpha)
                                            ax.add_patch(rect)

                        ax.set_aspect('equal')
                        # plt.gca().invert_yaxis()
                        plt.xticks([])
                        plt.yticks([])
                        plt.title('')

                        plt.savefig(f"/tmp/{label}.jpg")
                        # Save the plot to S3
                        key = f"/{inference_id}/{label}.jpg"
                        
                        with open(f"/tmp/{label}.jpg", "rb") as file:
                            s3_client.put_object(Bucket=bucket_name_prediction_imgs, Key=key, Body=file )


                    probabilities = probabilities.tolist()
                    binary_results = [1 if prob > 0.9 else 0 for prob in probabilities]
                    labels = ["grass", "clover", "soil", "dung"]

                    InferenceResult(
                        id=inference_id,
                        model_id=model_object.model_name,
                        img_id=image.img_url,
                        img_field = image.img_field,
                        labels=labels,
                        results=probabilities,
                        binary_results=binary_results,
                        percentage_coverage = percentage_coverage,
                        created_at=int(datetime.datetime.now().timestamp()),
                        updated_at=int(datetime.datetime.now().timestamp())
                    ).save()
                counter = counter + 1
                inference_progress.percentage_completed = (counter/n_predictions)*100
                inference_progress.save()


        inference_progress.currently_training = False
        inference_progress.percentage_completed = 0.0
        inference_progress.save()

        return {"message": "Inference completed successfully"}
    except S3Error as err:
        print(err)
        inference_progress.currently_training = False
        inference_progress.percentage_completed = 0.0
        inference_progress.save()
        raise HTTPException(status_code=500, detail="Error interacting with MinIO")
    except Exception as e:
        print(e)
        inference_progress.currently_training = False
        inference_progress.percentage_completed = 0.0
        inference_progress.save()
        raise HTTPException(status_code=500, detail="Internal server error")



@app.delete("/delete_all_predictions")
def delete_all_predictions():
    try:
        inference_list = list(InferenceResult.scan())

        for inference_res in inference_list:

            # minio_client.delete_object(bucket_name_prediction_imgs, inference_res.id)
            # print("Successfully removed object", inference_res.id )
            inference_res.delete()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error deleting all images from DynamoDB")


@app.post("/predict/{img_url}", status_code=status.HTTP_200_OK)
async def predict(img_url: str, img_id: str, model_id: str):
    try:
        # Retrieve the image from MinIO
        inference_past_results = list(InferenceResult.scan())

        skip = False

        model_id = model_id.split('/')[-1]
        print(model_id)

        if len(inference_past_results) >0:
            for result in inference_past_results:
                if result.model_id == model_id and result.img_id == img_url:
                    skip = True
        if skip:
            print("Skipping inference: result already existent")
        else:

            inference_id = str(uuid4())


            response = s3_client.get_object(Bucket=bucket_name_imgs, Key=img_url)
            image_data = response['Body'].read()
            response['Body'].close()
            # response['Body'].release_conn()

            # Load the model from MinIO
            model_response = s3_client.get_object(Bucket=bucket_name_models, Key=model_id)
            model_data = model_response['Body'].read()
            model_response['Body'].close()

            with zipfile.ZipFile(BytesIO(model_data)) as zip_ref:
                zip_ref.extractall("/tmp/")

            model_folder = model_id.split('.zip')[0]
            print("model folder", model_folder)

            model_file_name = os.listdir("/tmp/" + model_folder +"/")

            print("model file name", model_file_name)
            
            model_file_path = "/tmp/" + model_folder + "/" + model_file_name[0]
            print("model file path", model_file_path)
            # Save the image to a path in the /tmp folder
            image_path = "/tmp/image.jpg"
            with open(image_path, "wb") as image_file:
                image_file.write(image_data)

            # Initialize the model architecture

            model = inference.load_model(model_file_path)

            model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))

            # Post-process the inference results
            paths, full_predictions, probabilities = inference.predict(model, image_path, "/tmp/work")

            # saving prediction images
            print(full_predictions)
            rect_size = 224
            labels = ["grass", "clover", "soil", "dung"]
            img = mpimg.imread('/tmp/image.jpg')  # Load the image using matplotlib
            max_x = max(extract_coordinates(fn)[0] for fn in paths) + rect_size
            max_y = max(extract_coordinates(fn)[1] for fn in paths) + rect_size

            percentage_coverage = []

            for class_idx, label in enumerate(labels):
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(img, extent=[0, max_x, max_y, 0])  # extent adjusts the image to fit plot coordinates
                for filename, prob in zip(paths, full_predictions):
                    coords = extract_coordinates(filename)
                    if coords:
                        x, y = coords

                        alpha = (prob[class_idx] -0.3) if prob[class_idx] > 0.3 else 0.0
                        color = 'red' if prob[class_idx] > 0.5 else 'none'
                        rect = plt.Rectangle((x, y), rect_size, rect_size, linewidth=1, edgecolor='blue', facecolor=color, alpha=alpha)
                        ax.add_patch(rect)

                # Set the limits of the plot based on the maximum coordinates found
                max_x = max(extract_coordinates(fn)[0] for fn in paths) + rect_size
                max_y = max(extract_coordinates(fn)[1] for fn in paths) + rect_size

                ax.set_xlim(0, max_x)
                ax.set_ylim(0, max_y)
                ax.set_aspect('equal')
                plt.gca().invert_yaxis()  # Invert y axis to match typical image coordinate system
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

                plt.xticks([])
                plt.yticks([])
                plt.title('')
                plt.savefig(f"/tmp/{label}.jpg")
                # Save the plot to S3
                key = f"/{inference_id}/{label}.jpg"
                
                with open(f"/tmp/{label}.jpg", "rb") as file:
                    s3_client.put_object(Bucket=bucket_name_prediction_imgs, Key=key, Body=file )


            probabilities = probabilities.tolist()
            # results = probabilities.cpu().numpy().tolist()
            
            binary_results = [1 if prob > 0.9 else 0 for prob in probabilities]
            labels = ["grass", "clover", "soil", "dung"]
            
            # Save the results in the InferenceResult table
            InferenceResult(
                id=inference_id,
                model_id=model_id,
                img_id= img_url,
                img_field = img_url,
                labels=labels,
                results=probabilities,
                binary_results=binary_results,
                percentage_coverage = percentage_coverage,
                created_at=int(datetime.datetime.now().timestamp()),
                updated_at=int(datetime.datetime.now().timestamp())
            ).save()

            print(inference_id)
            return {
            "inference_id": inference_id,
            "model_id": model_id,
            "img_id": img_url,
            "img_field": img_url,
            "labels": labels,
            "results": probabilities,
            "binary_results": binary_results
        }
    except S3Error as err:
        print(err)
        raise HTTPException(status_code=500, detail="Error retrieving data from MinIO")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error")






# get the image from the Minio bucket using the img_url
# load the model from the Minio bucket
# perform inference
# save the results to the InferenceResult table
# return the results

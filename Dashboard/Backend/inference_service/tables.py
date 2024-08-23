
from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute, ListAttribute, NumberAttribute, MapAttribute, UnicodeSetAttribute, BooleanAttribute
from uuid import uuid4
from datetime import datetime
from inference_service.settings import config


class BaseTable(Model):
    class Meta:
        host = config.DB_HOST if config.ENVIRONMENT in ["local", "test"] else None
        region = config.AWS_REGION


class ImageTable(BaseTable):
    """
    Represents a DynamoDB table for an Image
    """
    class Meta(BaseTable.Meta):
        table_name = "image-table"

    id = UnicodeAttribute(hash_key=True)
    img_url = UnicodeAttribute(null=False)
    img_field = UnicodeAttribute(null=False)
    coordinates = UnicodeAttribute(null=False)
    time = NumberAttribute(null=False)
    created_at = NumberAttribute(null=False)
    updated_at = NumberAttribute(null=False)

# Define the VisionModel table
class VisionModel(BaseTable):
    class Meta(BaseTable.Meta):
        table_name = "VisionModel"

    id = UnicodeAttribute(hash_key=True, default=lambda: str(uuid4()))
    model_name = UnicodeAttribute(null=False)
    model_id = UnicodeAttribute(null=False)
    model_path = UnicodeAttribute(null=False)
    created_at = NumberAttribute(default=lambda: int(datetime.now().timestamp()))
    updated_at = NumberAttribute(default=lambda: int(datetime.now().timestamp()))

    # Add the accuracy

# Define the InferenceResult table
class InferenceResult(BaseTable):
    class Meta(BaseTable.Meta):
        table_name = "InferenceResult"

    id = UnicodeAttribute(hash_key=True, default=lambda: str(uuid4()))
    model_id = UnicodeAttribute()
    img_id = UnicodeAttribute()
    img_field = UnicodeAttribute()
    labels = ListAttribute(of=UnicodeAttribute)
    results = ListAttribute(of=NumberAttribute) 
    binary_results = ListAttribute(of=NumberAttribute) # Maybe not needed
    percentage_coverage = ListAttribute(of=NumberAttribute)
    created_at = NumberAttribute(default=lambda: int(datetime.now().timestamp()))
    updated_at = NumberAttribute(default=lambda: int(datetime.now().timestamp()))

# Define the Results table
class Results(BaseTable):
    class Meta(BaseTable.Meta):
        table_name = "Results"

    id = UnicodeAttribute(hash_key=True, default=lambda: str(uuid4()))
    img_id = UnicodeAttribute()
    model_list = ListAttribute(of=MapAttribute)
    results = ListAttribute(of=MapAttribute)
    created_at = NumberAttribute(default=lambda: int(datetime.now().timestamp()))
    updated_at = NumberAttribute(default=lambda: int(datetime.now().timestamp()))


class InferenceProgress(BaseTable):
    class Meta(BaseTable.Meta):
        table_name = "Inference_Progress"
    id = UnicodeAttribute(hash_key=True, default=lambda: str(uuid4()))
    currently_training = BooleanAttribute(default=False)
    percentage_completed = NumberAttribute(default=0.0)
# Create tables if they don't exist


if not InferenceProgress.exists():
    InferenceProgress.create_table(
        read_capacity_units=1,
        write_capacity_units=1,
        wait=True
    )
    print("InferenceProgress created")

    inference_progress = InferenceProgress(
                currently_training = False,
                percentage_completed = 0.0
            )
    inference_progress.save()
else:
    print("InferenceProgress already exists")

if not VisionModel.exists():
    VisionModel.create_table(
        read_capacity_units=1, 
        write_capacity_units=1, 
        wait=True
    )
    print("VisionModel created successfully.")
else:
    print("VisionModel already exists.")

if not InferenceResult.exists():
    InferenceResult.create_table(
        read_capacity_units=1, 
        write_capacity_units=1, 
        wait=True
    )
    print("Results Table created successfully.")
else:
    print("Results Table already exists.")

if not Results.exists():
    Results.create_table(
        read_capacity_units=1, 
        write_capacity_units=1, 
        wait=True
    )
    print("Results Table created successfully.")
else:
    print("Results Table already exists.")
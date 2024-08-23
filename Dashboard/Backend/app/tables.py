from pynamodb.attributes import NumberAttribute, UnicodeAttribute, BooleanAttribute
from pynamodb.indexes import AllProjection, GlobalSecondaryIndex
from pynamodb.models import Model
from uuid import uuid4
from app.settings import config


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
    coordinates = UnicodeAttribute(null=False)
    img_field = UnicodeAttribute(null=False)
    time = NumberAttribute(null=False)
    created_at = NumberAttribute(null=False)
    updated_at = NumberAttribute(null=False)

if not ImageTable.exists():
    ImageTable.create_table(
        read_capacity_units=1, 
        write_capacity_units=1, 
        wait=True
    )
    print("Table created successfully.")


class ProductNameIndex(GlobalSecondaryIndex["ProductTable"]):
    """
    Represents a global secondary index for ProductTable
    """

    class Meta:
        index_name = "product-name-index"
        read_capacity_units = 10
        write_capacity_units = 10
        projection = AllProjection()

    name = UnicodeAttribute(hash_key=True)
    updated_at = NumberAttribute(range_key=True)


class InferenceProgress(BaseTable):
    class Meta(BaseTable.Meta):
        table_name = "Inference_Progress"
    id = UnicodeAttribute(hash_key=True, default=lambda: str(uuid4()))
    currently_training = BooleanAttribute(default=False)
    percentage_completed = NumberAttribute(default=0.0)



class ProductTable(BaseTable):
    """
    Represents a DynamoDB table for a Product
    """

    class Meta(BaseTable.Meta):
        table_name = "product-table"

    id = UnicodeAttribute(hash_key=True)
    name = UnicodeAttribute(null=False)
    description = UnicodeAttribute(null=False)
    created_at = NumberAttribute(null=False)
    updated_at = NumberAttribute(null=False)

    product_name_index = ProductNameIndex()

if not ProductTable.exists():
    ProductTable.create_table(
        read_capacity_units=1, 
        write_capacity_units=1, 
        wait=True
    )
    print("Table created successfully.")
else:
    print("Table already exists.")
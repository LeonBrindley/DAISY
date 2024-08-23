from pydantic import BaseSettings


class GlobalConfig(BaseSettings):
    DB_HOST: str = "http://dynamodb:8000"
    ENVIRONMENT: str = "test"
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = "key"
    AWS_SECRET_ACCESS_KEY_ID: str = "secret"

config = GlobalConfig()
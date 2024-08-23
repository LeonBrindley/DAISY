from pydantic import BaseSettings


class GlobalConfig(BaseSettings):
    DB_HOST: str = 'https://dynamodb.eu-north-1.amazonaws.com'
    ENVIRONMENT: str = 'test'
    AWS_REGION: str = ''
    AWS_ACCESS_KEY_ID: str = ''
    AWS_SECRET_ACCESS_KEY: str = ''

    
config = GlobalConfig()
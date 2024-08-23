from pydantic import BaseSettings


class GlobalConfig(BaseSettings):
    DB_HOST: str = 'https://dynamodb.eu-north-1.amazonaws.com'
    ENVIRONMENT: str = 'test'
    AWS_REGION: str = 'eu-north-1'
    AWS_ACCESS_KEY_ID: str = 'AKIAZI2LBYEBH3UXKYV3'
    AWS_SECRET_ACCESS_KEY: str = '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'

    
config = GlobalConfig()
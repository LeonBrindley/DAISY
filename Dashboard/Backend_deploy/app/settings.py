from pydantic import BaseSettings


class GlobalConfig(BaseSettings):
    DB_HOST: str = 'https://dynamodb.eu-north-1.amazonaws.com'
    ENVIRONMENT: str = 'test'
    AWS_REGION: str = 'eu-north-1'
    AWS_ACCESS_KEY_ID: str = 'AKIAZI2LBYEBCNVSRWM2'
    AWS_SECRET_ACCESS_KEY: str = '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'


config = GlobalConfig()

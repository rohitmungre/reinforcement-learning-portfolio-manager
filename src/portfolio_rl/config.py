from pydantic import BaseSettings

class Settings(BaseSettings):
    alphavantage_key: str

    class Config:
        env_file = ".env"

settings = Settings()

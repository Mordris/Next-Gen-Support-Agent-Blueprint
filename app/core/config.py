# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # API key for our own service
    API_KEY: str = "default_secret_key"


# Instantiate the settings
settings = Settings()

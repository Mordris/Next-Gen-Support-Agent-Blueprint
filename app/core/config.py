# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # This tells Pydantic to load variables from a .env file
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8")

    # Define your settings here
    API_KEY: str = "default_secret_key"  # A default is good for local dev


# Instantiate the settings
settings = Settings()

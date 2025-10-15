from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    OPEN_API_KEY : str
    youtube_url : str

    class Config:

        env_file = ".env"

def get_settings():
    return Settings()
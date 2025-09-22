from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATA_DIR: Path = Path("./data")
    RAW_DIR: Path = Path("./data/raw")
    PROCESSED_DIR: Path = Path("./data/processed")
    MLFLOW_TRACKING_URI: str = "./mlruns"
    MODEL_NAME: str = "nemo_pd"
    RANDOM_SEED: int = 42

    # pydantic v2 config
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

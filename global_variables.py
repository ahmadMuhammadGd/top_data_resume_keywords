from dotenv import load_dotenv

load_dotenv()

import os

__current_directory = os.path.dirname(os.path.realpath(__file__))

def __construct_full_path(env:str)->str:
    return os.path.join(__current_directory  ,os.getenv(env))

LANDING_DIR                 =   __construct_full_path(env="LANDING_DIR")
LUKES_DATASET_NAME          =   os.getenv("LUKES_DATASET_NAME")
LUKES_DATASET_REF           =   os.getenv("LUKES_DATASET_REF")
KAGGLE_CONFIG_DIR           =   os.getenv("KAGGLE_CONFIG_DIR")
CLEAN_LOCATIONS_API         =   os.getenv("CLEAN_LOCATIONS_API")
CLEAN_LOCATION_DATASET_NAME =   os.getenv("CLEAN_LOCATION_DATASET_NAME")
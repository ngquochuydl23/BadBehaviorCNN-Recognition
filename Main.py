import kaggle
import os

def download_dataset():
    os.system("kaggle datasets download -d tadatho/anomaly")

download_dataset()
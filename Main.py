import kaggle
import os
from zipfile import ZipFile
import logging
import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)


def download_dataset():
    logging.info("Start downloading dataset")
    target_directory = "dataset"
    zip_file = "anomaly.zip"

    os.system("kaggle datasets download -d tadatho/anomaly")
    os.makedirs(target_directory, exist_ok=True)

    logging.info(f'Extract into the directory: {target_directory}')
    with ZipFile(zip_file, 'r') as zip:
        zip.extractall(target_directory)
        zip.printdir()

    os.remove(zip_file)

    logging.info(f'Done! Extracted files are in the directory: {target_directory}')

def resize_video_frame():
    logging.info("Start resizing video frame")
    IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
    max_sequence = 10

    classes_dir = ['Action-20240116T122550Z-001', 'Danger Action-20240116T090758Z-001', 'None Action-20240116T120846Z-001','Normal Action-20240116T090803Z-001']
    dataset_dir = "./dataset"

download_dataset()
resize_video_frame()

#vgg = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
#gg.summary()
#vgg.save_weights("vgg16.h5")
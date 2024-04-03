import cv2
import os
from zipfile import ZipFile
import logging
import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from kaggle.api.kaggle_api_extended import KaggleApi


logging.basicConfig(format='[%(asctime)s] %(levelname)s %(message)s', level=logging.DEBUG)

def download_dataset():
    api = KaggleApi()
    api.authenticate()

    logging.info("Start downloading dataset")
    target_directory = "dataset"
    zip_file = "anomaly.zip"

    os.makedirs(target_directory, exist_ok=True)
    os.system("kaggle datasets download -d tadatho/anomaly")

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

    classes_dir = ['Action-20240116T122550Z-001', 'Danger Action-20240116T090758Z-001',
                   'None Action-20240116T120846Z-001', 'Normal Action-20240116T090803Z-001']
    dataset_dir = "./dataset"


# download_dataset()

resize_video_frame()

vgg = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
vgg.summary()
vgg.save_weights("vgg16.weights.h5")

def frames_extraction(path):
    max_sequence = 10
    width = 224
    height = 224
    video = cv2.VideoCapture(path)

    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frame is: ", frames_count)

    skip_frame_window = max((frames_count / max_sequence), 1)
    print("Frame will skip is: ", skip_frame_window)

    frame_list = []
    for frame_i in range(max_sequence):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_i * skip_frame_window)
        success, frame = video.read()
        if not success:
            print("[Error!]")
            break

        # Adding some method processing
        resize_frame = cv2.resize(frame, (height, width))
        normalized_frame = resize_frame / 255

        img = normalized_frame.reshape(1, 224, 224, 3)
        pre = vgg.predict(img)
        frame_list.append(pre.reshape(25088))

        # frame_list.append(normalized_frame)
    video.release()
    return frame_list

def create_dataset():
    '''
    This function will extract the data of the selected classes and create the required dataset.
    Returns:
        features:          A list containing the extracted frames of the videos.
        labels:            A list containing the indexes of the classes associated with the videos.
        video_files_paths: A list containing the paths of the videos in the disk.
    '''

    # Specify the height and width to which each video frame will be resized in our dataset.
    IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224

    # Specify the number of frames of a video that will be fed to the model as one sequence.
    SEQUENCE_LENGTH = 10
    CLASSES_LIST = [
        'Action-20240116T122550Z-001',
        'Danger Action-20240116T090758Z-001',
        'None Action-20240116T120846Z-001',
        'Normal Action-20240116T090803Z-001'

    ]
    DATASET_DIR = "./dataset"

    # Declared Empty Lists to store the features, labels and video file path values.
    features = []
    labels = []
    video_files_paths = []

    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(CLASSES_LIST):

        # Display the name of the class whose data is being extracted.
        print(f'Extracting Data of Class: {class_name}')
        count = 0

        # Get the list of video files present in the specific class name directory.

        folder = os.listdir(os.path.join(DATASET_DIR, class_name))
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name, folder[0]))


        # Iterate through all the files present in the files list.
        for file_name in files_list:

            # Get the complete video path.
            video_file_path = os.path.join(DATASET_DIR, class_name, folder[0], file_name)

            # Extract the frames of the video file.

            frames = frames_extraction(video_file_path)

            # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
            # So ignore the vides having frames less than the SEQUENCE_LENGTH.
            if len(frames) == SEQUENCE_LENGTH:

                # Append the data to their repective lists.
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)
                count+=1


    # Converting the list to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)

    # Return the frames, class index, and video file path.
    return features, labels, video_files_paths

features, labels, video_files_paths = create_dataset()
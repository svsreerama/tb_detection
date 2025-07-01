import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
DATA_DIR = '/mnt/e/e_project/tb/archive/Dataset of Tuberculosis Chest X-rays Images'
CLASSES = ['Normal Chest X-rays', 'TB Chest X-rays']

def load_data():
    images, labels = [], []
    for label, category in enumerate(CLASSES):
        path = os.path.join(DATA_DIR, category)
        count = 0
        for file in os.listdir(path):
            if count >= 300:
                break
            img_path = os.path.join(path, file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
                count += 1
            except:
                continue
    return np.array(images) / 255.0, np.array(labels)

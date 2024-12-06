import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import pandas as pd

df1 = pd.read_csv('/content/finaaalllllll_train.csv')
df2 = pd.read_csv('/content/finaaalllllll_val.csv')
import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
import cv2

# write a function that loads the dataset into detectron2's standard format
def get_data_dicts(csv_file, img_dir):
    df = pd.read_csv(csv_file)
    df['filename'] = df['Frame'].map(lambda x: img_dir + x)

    classes = df['Label'].unique().tolist()

    df['class_int'] = df['Label'].map(lambda x: classes.index(x))

    dataset_dicts = []
    for idx, filename in enumerate(df['filename'].unique().tolist()):
        record = {}

        #height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = 1024
        record["width"] = 1024

        objs = []
        for index, row in df[(df['filename']==filename)].iterrows():
          obj= {
              'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
              'bbox_mode': BoxMode.XYXY_ABS,
              'category_id': row['class_int'],
              "iscrowd": 0
          }
          objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
from detectron2.data import DatasetCatalog, MetadataCatalog

classes = df1['Label'].unique().tolist()

DatasetCatalog.register('trainn_data', lambda: get_data_dicts('/content/finaaalllllll_train.csv' , '/content/drive/MyDrive/projet capgemini/resized_images_train/'))
MetadataCatalog.get('trainn_data').set(thing_classes=classes)
train_metadata = MetadataCatalog.get('trainn_data')
DatasetCatalog.register('valll_data', lambda: get_data_dicts('/content/finaaalllllll_val.csv' , '/content/drive/MyDrive/projet capgemini/resized_images_val/'))
MetadataCatalog.get('valll_data').set(thing_classes=classes)
tesst_metadata = MetadataCatalog.get('valll_data')
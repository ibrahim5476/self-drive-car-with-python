!python -m pip install pyyaml==5.1

!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

def parse_csv_annotations(csv_file):
    annotations = pd.read_csv(csv_file)
    return annotations

def resize_image_and_annotations(image, annotations, size=(1024, 1024)):
    orig_height, orig_width = image.shape[:2]
    resized_image = cv2.resize(image, size)
    x_scale = size[0] / orig_width
    y_scale = size[1] / orig_height
    resized_annotations = []
    for box in annotations:
        x1, y1 = int(box[0][0] * x_scale), int(box[0][1] * y_scale)
        x2, y2 = int(box[1][0] * x_scale), int(box[1][1] * y_scale)
        resized_annotations.append([(x1, y1), (x2, y2), box[2]])
    return resized_image, resized_annotations

def flip_image_and_annotations(image, annotations, flip_code):
    flipped_image = cv2.flip(image, flip_code)
    img_height, img_width = image.shape[:2]
    flipped_annotations = []
    for box in annotations:
        x1, y1 = box[0]
        x2, y2 = box[1]
        if flip_code == 1:  # Horizontal flip
            x1, x2 = img_width - x2, img_width - x1
        elif flip_code == 0:  # Vertical flip
            y1, y2 = img_height - y2, img_height - y1
        flipped_annotations.append([(x1, y1), (x2, y2), box[2]])
    return flipped_image, flipped_annotations

def rotate_image_and_annotations(image, annotations, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    rotated_annotations = []
    for box in annotations:
        x1, y1 = box[0]
        x2, y2 = box[1]
        x1_rot = int(x1 * rotation_matrix[0, 0] + y1 * rotation_matrix[0, 1] + rotation_matrix[0, 2])
        y1_rot = int(x1 * rotation_matrix[1, 0] + y1 * rotation_matrix[1, 1] + rotation_matrix[1, 2])
        x2_rot = int(x2 * rotation_matrix[0, 0] + y2 * rotation_matrix[0, 1] + rotation_matrix[0, 2])
        y2_rot = int(x2 * rotation_matrix[1, 0] + y2 * rotation_matrix[1, 1] + rotation_matrix[1, 2])
        rotated_annotations.append([(x1_rot, y1_rot), (x2_rot, y2_rot), box[2]])
    return rotated_image, rotated_annotations

def display_image_with_annotations(image, annotations):
    for box in annotations:
        cv2.rectangle(image, box[0], box[1], (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image with Annotations')
    plt.show()

def save_image_without_annotations(image, output_path):
    cv2.imwrite(output_path, image)

def save_annotations(annotations, output_path):
    with open(output_path, 'w') as f:
        for box in annotations:
            line = f"({box[0][0]},{box[0][1]}),({box[1][0]},{box[1][1]}),{box[2]}\n"
            f.write(line)

def get_image_data(df):
    data = {}
    for _, row in df.iterrows():
        file_name = row['frame']
        coordinates = {
            'xmin': row['xmin'],
            'xmax': row['xmax'],
            'ymin': row['ymin'],
            'ymax': row['ymax']
        }
        class_label = row['class_id']
        if file_name not in data:
            data[file_name] = []
        data[file_name].append([(coordinates['xmin'], coordinates['ymin']),
                                (coordinates['xmax'], coordinates['ymax']),
                                class_label])
    return data

image_folder = '/input/self-driving-cars/images'
output_image_folder = 'output_imagess'
output_annotation_folder = 'output_annotationss'
annotations_files = ['/input/self-driving-cars/labels_train.csv', '/input/self-driving-cars/labels_trainval.csv', '/input/self-driving-cars/labels_val.csv']

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_annotation_folder, exist_ok=True)

# Combine annotations from all CSV files
annotations = pd.concat([parse_csv_annotations(file) for file in annotations_files], ignore_index=True)

# Get the image data
parsed_data = get_image_data(annotations)

# Limit to 3000 images
sampled_image_data = dict(list(parsed_data.items())[:750])

# Apply transformations and save
def apply_transformations_and_save(parsed_data):
    for file_name, annotations in tqdm(parsed_data.items(), desc='Processing transformations'):
        image_path = os.path.join(image_folder, file_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Image file {image_path} not found.")
            continue

        # Resize
        resized_image, resized_annotations = resize_image_and_annotations(image, annotations)
        save_image_without_annotations(resized_image, os.path.join(output_image_folder, f'resized_{file_name}'))
        save_annotations(resized_annotations, os.path.join(output_annotation_folder, f'resized_{file_name}.txt'))

        # Horizontal Flip
        h_flipped_image, h_flipped_annotations = flip_image_and_annotations(image, annotations, 1)
        save_image_without_annotations(h_flipped_image, os.path.join(output_image_folder, f'h_flipped_{file_name}'))
        save_annotations(h_flipped_annotations, os.path.join(output_annotation_folder, f'h_flipped_{file_name}.txt'))

        # Vertical Flip
        v_flipped_image, v_flipped_annotations = flip_image_and_annotations(image, annotations, 0)
        save_image_without_annotations(v_flipped_image, os.path.join(output_image_folder, f'v_flipped_{file_name}'))
        save_annotations(v_flipped_annotations, os.path.join(output_annotation_folder, f'v_flipped_{file_name}.txt'))

        # Random Rotation
        angle = random.choice([90, 180, 270])
        rotated_image, rotated_annotations = rotate_image_and_annotations(image, annotations, angle)
        save_image_without_annotations(rotated_image, os.path.join(output_image_folder, f'rotated_{angle}_{file_name}'))
        save_annotations(rotated_annotations, os.path.join(output_annotation_folder, f'rotated_{angle}_{file_name}.txt'))

        
apply_transformations_and_save(sampled_image_data)
print("Processing completed!")
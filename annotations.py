import os
import cv2
import json
import numpy as np
from glob import glob
from skimage import measure
from tqdm import tqdm

# Define the dataset path
dataset_path = 'C:\\Users\\22211\\Documents\\BT-dataset2\\MRI Image Dataset for Brain Tumor'
# Replace with the actual path to the downloaded dataset

# Create directories for images and annotations if they don't exist
os.makedirs('annotations', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Function to detect tumors using thresholding and contour detection
def detect_tumors(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)

    # Finding contours
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotations = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small contours
            mask = np.zeros(image.shape, dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)

            # Find bounding box
            (x, y, w, h) = cv2.boundingRect(contour)

            # Polygon approximation
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            segmentation = approx.flatten().tolist()

            # COCO annotation format
            annotation = {
                "bbox": [x, y, w, h],
                "segmentation": [segmentation],
                "area": cv2.contourArea(contour),
                "iscrowd": 0,
                "category_id": 1  # Assuming 'tumor' is the only category
            }
            annotations.append(annotation)
    
    return annotations

# Function to generate COCO JSON
def generate_coco_json(dataset_path, output_json):
    images = []
    annotations = []

    image_files = glob(os.path.join(dataset_path, '*.jpg'))  # Assuming images are in JPG format
    image_id = 1
    annotation_id = 1

    for image_file in tqdm(image_files):
        image = cv2.imread(image_file)
        height, width, _ = image.shape

        # Create image info
        image_info = {
            "id": image_id,
            "file_name": os.path.basename(image_file),
            "width": width,
            "height": height
        }
        images.append(image_info)

        # Detect tumors
        tumor_annotations = detect_tumors(image_file)
        for tumor in tumor_annotations:
            tumor["image_id"] = image_id
            tumor["id"] = annotation_id
            annotations.append(tumor)
            annotation_id += 1

        # Copy images to the new directory
        cv2.imwrite(os.path.join('images', os.path.basename(image_file)), image)

        image_id += 1

    # COCO JSON format
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "tumor",
            "supercategory": "none"
        }]
    }

    # Write JSON file
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=4)

# Generate COCO JSON annotations
generate_coco_json(dataset_path, 'C:\\Users\\22211\\Documents\\BT-dataset2\\MRI Image Dataset for Brain Tumor\\Training\\annotations.json.txt')

print("JSON annotations generated successfully.")

import json
import cv2
import os
import random
import matplotlib.pyplot as plt

def visualize_coco_annotation():
    # Paths
    base_path = "E:/thesis models/RT-DETR-Pipeline/CamoCrops_2025"
    json_path = os.path.join(base_path, "labels/annotations_test_new.json")
    image_dir = os.path.join(base_path, "Images/Test_New")

    # Load COCO annotations
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Get random image
    image_info = random.choice(coco_data['images'])
    image_id = image_info['id']
    image_path = os.path.join(image_dir, image_info['file_name'])

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get annotations for this image
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    
    # Create category id to name mapping
    cat_mapping = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Plot image
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Draw boxes and labels
    for ann in annotations:
        bbox = ann['bbox']
        category = cat_mapping[ann['category_id']]
        
        # Draw rectangle
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                           fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        
        # Add label
        plt.text(bbox[0], bbox[1]-5, category,
                color='red', fontsize=12, backgroundcolor='white')

    plt.title(f'Image: {image_info["file_name"]}\nFound {len(annotations)} objects')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    visualize_coco_annotation()

import os
import shutil
import xml.etree.ElementTree as ET

CLASSES = {
    "ampalaya": 0, "atis": 1, "avocado": 2, "bell_pepper": 3,
    "calamansi": 4, "cucumber": 5, "dalandan": 6, "french_bean": 7,
    "green_chili": 8, "green_mango": 9, "guava": 10, "guyabano": 11,
    "okra": 12, "pea": 13, "sayote": 14
}

def convert_bbox_to_yolo(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x*dw, y*dh, w*dw, h*dh)

def convert_and_copy(source_base, dest_base):
    # Create destination directories
    for split in ['train', 'test', 'validate']:
        os.makedirs(f"{dest_base}/images/{split}", exist_ok=True)
        os.makedirs(f"{dest_base}/labels/{split}", exist_ok=True)

    # Mapping of source to destination folders
    splits_mapping = {
        'Train': 'train',
        'Test_New': 'test',
        'Validation': 'validate'
    }

    for src_split, dst_split in splits_mapping.items():
        xml_dir = f"{source_base}/Annotations/{src_split}"
        img_src_dir = f"{source_base}/Images/{src_split}"
        
        img_dest_dir = f"{dest_base}/images/{dst_split}"
        label_dest_dir = f"{dest_base}/labels/{dst_split}"

        print(f"\nProcessing {src_split} split:")
        
        # First, copy all images from source to destination
        for img_file in os.listdir(img_src_dir):
            shutil.copy2(
                os.path.join(img_src_dir, img_file),
                os.path.join(img_dest_dir, img_file)
            )

        # Then process XML files
        for xml_file in os.listdir(xml_dir):
            if not xml_file.endswith('.xml'):
                continue

            # Parse XML
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()

            # Get image size
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            # Create YOLO annotation file
            txt_path = os.path.join(label_dest_dir, xml_file.replace('.xml', '.txt'))
            with open(txt_path, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    class_id = CLASSES[class_name]
                    
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), 
                         float(xmlbox.find('ymin').text), 
                         float(xmlbox.find('xmax').text), 
                         float(xmlbox.find('ymax').text))
                    yolo_bbox = convert_bbox_to_yolo((width, height), b)
                    f.write(f"{class_id} {' '.join([str(x) for x in yolo_bbox])}\n")

def verify_dataset(dest_base):
    splits = ['train', 'test', 'validate']
    all_correct = True
    stats = {split: {'images': 0, 'labels': 0, 'mismatches': 0} for split in splits}
    
    for split in splits:
        img_dir = os.path.join(dest_base, 'images', split)
        label_dir = os.path.join(dest_base, 'labels', split)
        
        # Get file lists (now checking all files in images directory)
        images = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
        labels = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}
        
        # Count files
        stats[split]['images'] = len(images)
        stats[split]['labels'] = len(labels)
        
        # Check mismatches
        img_without_label = images - labels
        label_without_img = labels - images
        stats[split]['mismatches'] = len(img_without_label) + len(label_without_img)
        
        if img_without_label or label_without_img:
            all_correct = False
            print(f"\nMismatches in {split} split:")
            if img_without_label:
                print(f"Images without labels: {img_without_label}")
            if label_without_img:
                print(f"Labels without images: {label_without_img}")
    
    print("\nDataset Statistics:")
    for split, counts in stats.items():
        print(f"\n{split.capitalize()} split:")
        print(f"Images: {counts['images']}")
        print(f"Labels: {counts['labels']}")
        print(f"Mismatches: {counts['mismatches']}")
    
    if all_correct:
        print("\nVerification Successful! All files are properly paired.")
    else:
        print("\nVerification Found Issues! Please check the mismatches above.")
    
    return all_correct

def create_combined_trainval(dest_base):
    print("\nCreating combined train+val dataset...")
    combined_dir = os.path.join(dest_base, 'images/combined_trainval')
    combined_labels_dir = os.path.join(dest_base, 'labels/combined_trainval')
    
    # Create combined directories
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(combined_labels_dir, exist_ok=True)
    
    # Copy train files
    train_images = os.path.join(dest_base, 'images/train')
    train_labels = os.path.join(dest_base, 'labels/train')
    
    for img in os.listdir(train_images):
        shutil.copy2(os.path.join(train_images, img), combined_dir)
        base_name = os.path.splitext(img)[0]
        label = base_name + '.txt'
        if os.path.exists(os.path.join(train_labels, label)):
            shutil.copy2(os.path.join(train_labels, label), combined_labels_dir)
    
    # Copy validation files
    val_images = os.path.join(dest_base, 'images/validate')
    val_labels = os.path.join(dest_base, 'labels/validate')
    
    for img in os.listdir(val_images):
        shutil.copy2(os.path.join(val_images, img), combined_dir)
        base_name = os.path.splitext(img)[0]
        label = base_name + '.txt'
        if os.path.exists(os.path.join(val_labels, label)):
            shutil.copy2(os.path.join(val_labels, label), combined_labels_dir)
    
    print("Combined train+val dataset created successfully!")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_base = os.path.join(base_dir, "toConvertDataset/CamoCrops_2025")
    dest_base = os.path.join(base_dir, "datasets", "camocrops")
    
    print(f"Source base path: {source_base}")
    print(f"Destination base path: {dest_base}")
    print(f"Script location: {base_dir}")
    
    convert_and_copy(source_base, dest_base)
    print("Dataset conversion and organization completed!")
    
    verify_dataset(dest_base)
    create_combined_trainval(dest_base)
import os
import shutil
import xml.etree.ElementTree as ET

CLASSES = {
    "ampalaya": 0, "atis": 1, "avocado": 2, "bell_pepper": 3,
    "calamansi": 4, "cucumber": 5, "dalandan": 6, "french_bean": 7,
    "green_chili": 8, "green_mango": 9, "guava": 10, "guyabano": 11,
    "okra": 12, "pea": 13, "sayote": 14
}

def get_image_file(img_src_dir, base_name):
    extensions = ['.jpg', '.jpeg', '.png', '.webp']
    for ext in extensions:
        potential_file = base_name + ext
        if os.path.exists(os.path.join(img_src_dir, potential_file)):
            return potential_file
    return None

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
        print(f"XML directory: {xml_dir}")
        print(f"Image source directory: {img_src_dir}")

        # Process each XML file
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

            # Copy corresponding image
            base_name = os.path.splitext(xml_file)[0]
            img_file = get_image_file(img_src_dir, base_name)
            if img_file:
                source_path = os.path.join(img_src_dir, img_file)
                dest_path = os.path.join(img_dest_dir, img_file)
                print(f"Copying {img_file}")
                shutil.copy2(source_path, dest_path)

def verify_dataset(dest_base):
    splits = ['train', 'test', 'validate']
    all_correct = True
    stats = {split: {'images': 0, 'labels': 0, 'mismatches': 0} for split in splits}
    
    for split in splits:
        img_dir = os.path.join(dest_base, 'images', split)
        label_dir = os.path.join(dest_base, 'labels', split)
        
        # Get file lists
        images = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))}
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
    
    # Print summary
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

if __name__ == "__main__":
    # Set base directory (where your script is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use relative paths from that base
    source_base = os.path.join(base_dir, "CamoCrops_2025")
    dest_base = os.path.join(base_dir, "datasets", "camocrops")
    
    # Print paths to verify
    print(f"Source base path: {source_base}")
    print(f"Destination base path: {dest_base}")
    print(f"Script location: {base_dir}")
    
    convert_and_copy(source_base, dest_base)
    print("Dataset conversion and organization completed!")
    
    # Verify the dataset
    verify_dataset(dest_base)

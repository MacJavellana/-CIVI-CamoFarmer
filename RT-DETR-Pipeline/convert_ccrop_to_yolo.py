import os
import shutil
import xml.etree.ElementTree as ET

CLASSES = {
    "BELLPEPPER": 0,
    "CHILI PEPPER": 1
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
    # Create destination directories for regular splits
    for split in ['train', 'test', 'validate', 'combined']:
        os.makedirs(f"{dest_base}/images/{split}", exist_ok=True)
        os.makedirs(f"{dest_base}/labels/{split}", exist_ok=True)

    # Process each class directory
    for class_name in CLASSES.keys():
        class_dir = os.path.join(source_base, class_name)
        xml_files = [f for f in os.listdir(class_dir) if f.endswith('.xml')]
        
        total_files = len(xml_files)
        train_split = int(0.8 * total_files)
        val_split = int(0.1 * total_files)
        
        train_files = xml_files[:train_split]
        val_files = xml_files[train_split:train_split + val_split]
        test_files = xml_files[train_split + val_split:]
        combined_files = train_files + val_files  # Combine train and validate
        
        splits = {
            'train': train_files,
            'validate': val_files,
            'test': test_files,
            'combined': combined_files
        }
        
        for split_name, files in splits.items():
            for xml_file in files:
                # Parse XML
                tree = ET.parse(os.path.join(class_dir, xml_file))
                root = tree.getroot()
                
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                
                # Create YOLO annotation file
                txt_path = os.path.join(dest_base, 'labels', split_name, xml_file.replace('.xml', '.txt'))
                
                with open(txt_path, 'w') as f:
                    for obj in root.findall('object'):
                        class_id = CLASSES[class_name]
                        xmlbox = obj.find('bndbox')
                        b = (float(xmlbox.find('xmin').text),
                             float(xmlbox.find('ymin').text),
                             float(xmlbox.find('xmax').text),
                             float(xmlbox.find('ymax').text))
                        yolo_bbox = convert_bbox_to_yolo((width, height), b)
                        f.write(f"{class_id} {' '.join([str(x) for x in yolo_bbox])}\n")
                
                # Copy corresponding image
                for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    img_file = xml_file.replace('.xml', ext)
                    img_path = os.path.join(class_dir, img_file)
                    if os.path.exists(img_path):
                        shutil.copy2(
                            img_path,
                            os.path.join(dest_base, 'images', split_name, img_file)
                        )
                        break


def verify_dataset(dest_base):
    splits = ['train', 'test', 'validate', 'combined']
    all_correct = True
    stats = {split: {'images': 0, 'labels': 0, 'mismatches': 0} for split in splits}
    
    for split in splits:
        img_dir = os.path.join(dest_base, 'images', split)
        label_dir = os.path.join(dest_base, 'labels', split)
        
        images = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
        labels = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}
        
        stats[split]['images'] = len(images)
        stats[split]['labels'] = len(labels)
        
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


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_base = os.path.join(base_dir, "toConvertDataset", "ccrop")
    dest_base = os.path.join(base_dir, "datasets", "ccrop")
    
    print(f"Source base path: {source_base}")
    print(f"Destination base path: {dest_base}")
    
    convert_and_copy(source_base, dest_base)
    print("Dataset conversion and organization completed!")
    
    verify_dataset(dest_base)

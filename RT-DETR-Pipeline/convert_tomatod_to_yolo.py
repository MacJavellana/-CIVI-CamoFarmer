import os
import shutil

def setup_directories(dest_base):
    # Create destination directories
    for split in ['train', 'test', 'validate', 'combined']:
        os.makedirs(f"{dest_base}/images/{split}", exist_ok=True)
        os.makedirs(f"{dest_base}/labels/{split}", exist_ok=True)

def copy_dataset(source_base, dest_base):
    # Map source to destination split names
    split_mapping = {
        'train': 'train',
        'test': 'test',
        'val': 'validate'
    }
    
    # Copy files for each split
    for src_split, dest_split in split_mapping.items():
        # Copy images
        src_img_path = os.path.join(source_base, src_split, 'images')
        dest_img_path = os.path.join(dest_base, 'images', dest_split)
        
        # Copy labels
        src_label_path = os.path.join(source_base, src_split, 'labels')
        dest_label_path = os.path.join(dest_base, 'labels', dest_split)
        
        # Copy all images and labels
        for img in os.listdir(src_img_path):
            shutil.copy2(
                os.path.join(src_img_path, img),
                os.path.join(dest_img_path, img)
            )
        
        for label in os.listdir(src_label_path):
            shutil.copy2(
                os.path.join(src_label_path, label),
                os.path.join(dest_label_path, label)
            )
        
        # If this is train or validate, also copy to combined
        if src_split in ['train', 'val']:
            dest_combined_img = os.path.join(dest_base, 'images', 'combined')
            dest_combined_label = os.path.join(dest_base, 'labels', 'combined')
            
            for img in os.listdir(src_img_path):
                shutil.copy2(
                    os.path.join(src_img_path, img),
                    os.path.join(dest_combined_img, img)
                )
            
            for label in os.listdir(src_label_path):
                shutil.copy2(
                    os.path.join(src_label_path, label),
                    os.path.join(dest_combined_label, label)
                )

def verify_dataset(dest_base):
    splits = ['train', 'test', 'validate', 'combined']
    all_correct = True
    stats = {split: {'images': 0, 'labels': 0, 'mismatches': 0} for split in splits}
    
    for split in splits:
        img_dir = os.path.join(dest_base, 'images', split)
        label_dir = os.path.join(dest_base, 'labels', split)
        
        images = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
        labels = {os.path.splitext(f)[0] for f in os.listdir(label_dir)}
        
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
    source_base = os.path.join(base_dir, "toConvertDataset", "tomatOD_yolo")
    dest_base = os.path.join(base_dir, "datasets", "tomatod")
    
    print(f"Source base path: {source_base}")
    print(f"Destination base path: {dest_base}")
    
    setup_directories(dest_base)
    copy_dataset(source_base, dest_base)
    print("Dataset organization completed!")
    
    verify_dataset(dest_base)

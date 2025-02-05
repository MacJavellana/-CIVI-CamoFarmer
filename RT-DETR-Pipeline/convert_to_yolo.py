#!/usr/bin/python
import os
import xml.etree.ElementTree as ET

CLASSES = {
    "ampalaya": 0,
    "atis": 1,
    "avocado": 2,
    "bell_pepper": 3,
    "calamansi": 4,
    "cucumber": 5,
    "dalandan": 6,
    "french_bean": 7,
    "green_chili": 8,
    "green_mango": 9,
    "guava": 10,
    "guyabano": 11,
    "okra": 12,
    "pea": 13,
    "sayote": 14
}

def convert_bbox_to_yolo(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(xml_path, txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(txt_path, 'w') as txt_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in CLASSES:
                continue
            cls_id = CLASSES[cls]
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), 
                 float(xmlbox.find('ymin').text), 
                 float(xmlbox.find('xmax').text), 
                 float(xmlbox.find('ymax').text))
            bb = convert_bbox_to_yolo((w,h), b)
            txt_file.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

def convert(xml_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    
    for xml_file in xml_files:
        print(f"Processing {xml_file}")
        xml_path = os.path.join(xml_dir, xml_file)
        txt_path = os.path.join(output_dir, xml_file.replace('.xml', '.txt'))
        convert_annotation(xml_path, txt_path)

def main():
    base_path = "E:/thesis models/RT-DETR-Pipeline/CamoCrops_2025"
    splits = ['Train', 'Validation', 'Test_new']
    
    for split in splits:
        xml_dir = os.path.join(base_path, 'Annotations', split)
        txt_dir = os.path.join(base_path, 'labels', split)
        print(f"\nProcessing {split} split...")
        convert(xml_dir, txt_dir)
        print(f"Completed {split} conversion!")

if __name__ == '__main__':
    main()

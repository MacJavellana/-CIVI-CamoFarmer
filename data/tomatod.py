import os
import cv2
import json
import torch
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from utils.genutils import write_print
import pickle
from pycocotools.coco import COCO as PYCOCO



TOMATOD_CLASSES = ('unripe', 'semi-ripe', 'fully-ripe')

TOMATOD_CLASSES_I = (1, 2, 3)


class TomatodAnnotationTransform(object):
    """Transform JSON annotations to [xmin, ymin, xmax, ymax, class]"""

    def __init__(self):
        """Class constructor for TomatODAnnotationTransform"""
        super(TomatodAnnotationTransform, self).__init__()

    def __call__(self,
                 targets,
                 width,
                 height):
        """Executed when the class is called as a function

        Arguments:
            targets {list} -- list of targets
            width {int} -- Width of the corresponding image; Used for scaling
            the coordinates of bounding boxes
            height {int} -- Height of the corresponding image; Used for
            scaling the coordinates of bounding boxes

        Returns:
            list -- list of bounding boxes formatted as
            [xmin, ymin, xmax, ymax, class]
        """

        labels = []

        for target in targets:
            bbox = [(float(target[0])) / width,
                    (float(target[1])) / height,
                    (float(target[2])) / width,
                    (float(target[3])) / height,
                    int(target[4])]
            labels += [bbox]

        return labels



class TOMATOD(Dataset):
    """Custom dataset loader for TomatoD dataset."""

    def __init__(self, data_path, new_size, mode, image_transform):
        """
        Arguments:
            data_path {string} -- Path to the dataset root directory.
            mode {string} -- Mode of dataset: train, val, or test.
            image_transform {object} -- Image transformation function.
        """
        super(TOMATOD, self).__init__()

        assert mode in ["train", "val", "test"], "Invalid mode. Choose from train, val, or test."

        self.data_path = data_path
        self.mode = mode
        self.new_size = new_size
        self.image_transform = image_transform
        self.target_transform = TomatodAnnotationTransform()

        class_to_index = dict(zip(TOMATOD_CLASSES_I, range(len(TOMATOD_CLASSES_I))))

        # Define annotation and image paths
        self.annotation_path = osp.join(self.data_path, f'annotations_{mode}.json')  # Use different JSON files
        self.image_dir = osp.join(self.data_path,
                                   'images',
                                   self.mode)

        if self.mode in ['val', 'test']:
            self.pycoco = PYCOCO(self.annotation_path)

        # Load annotation file
        with open(self.annotation_path) as file:
            self.data = json.load(file)

        # Create dictionary to store image metadata
        self.image_metadata = {
            str(img["id"]): img["file_name"] for img in self.data["images"]
        }
        self.ids = list(self.image_metadata.keys())  # Use image IDs from annotations

        # Convert annotations to dict
        self.dict_targets = {img_id: [] for img_id in self.ids}
        for instance in self.data['annotations']:
            image_id = str(instance['image_id'])
            if image_id not in self.dict_targets:
                    self.dict_targets[image_id] = []
            

            x_min = instance['bbox'][0]
            y_min = instance['bbox'][1]
            x_max = instance['bbox'][0] + instance['bbox'][2]
            y_max = instance['bbox'][1] + instance['bbox'][3]
            mapped_class = class_to_index[instance['category_id']]
            bbox = [x_min, y_min, x_max, y_max, mapped_class]
            self.dict_targets[image_id] += [bbox]
            


    def __len__(self):
        """Returns the number of images in the dataset

        Returns:
            int -- number of images in the dataset
        """

        return len(self.ids)

    def __getitem__(self,
                    index):
        """Gets the image and its corresponding annotation found in
        position index of the list of images in the dataset

        Arguments:
            index {int} -- index of the image in the list of images in the
            dataset

        Returns:
            torch.Tensor, np.ndarray, -- tensor representation of the image,
            list of bounding boxes of objects in the image formatted as
            [xmin, ymin, xmax, ymax, class]
        """

        image, target, _, _ = self.pull_item(index)
        return image, target

    def pull_item(self,
                  index):
        """Gets the image found in position index of the list of images in the
        dataset together with its corresponding annotation, its height, and its
        width

        Arguments:
            index {int} -- index of the image in the list of images in the
            dataset

        Returns:
            torch.Tensor, np.ndarray, int, int -- tensor representation of
            the image, list of bounding boxes of objects in the image
            formatted as [xmin, ymin, xmax, ymax, class], height, width
        """

        image_id = self.ids[index]
        file_name = self.image_metadata[image_id]
        # print(file_name)

        # target_path = self.annotation_path.format(image_id)
        image_path = osp.join(self.image_dir, file_name)
        # print("Image Path:" + image_path)

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        height, width, _ = image.shape

        target = self.dict_targets[image_id]
        target = self.target_transform(target, width, height)

        if self.image_transform is not None:
            target = np.array(target)
            boxes = target[:, :4]
            labels = target[:, 4]
            image, boxes, labels = self.image_transform(image, boxes, labels)

            # to rgb
            image = image[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(image).permute(2, 0, 1), target, height, width

    def pull_image(self,
                   index):
        """Gets the image found in position index of the list of images in the
        dataset

        Arguments:
            index {int} -- index of the image in the list of images in the
            dataset

        Returns:
            np.ndarray -- image
        """

        image_id = self.ids[index]
        image_path = self.image_path.format(image_id)
        return cv2.imread(image_path, cv2.IMREAD_COLOR)

    def pull_annotation(self,
                        index):
        """Gets the annotation of the image found in position index of the
        list of images in the dataset. The coordinates of the annotation is
        not scaled by the height and width of the image

        Arguments:
            index {int} -- index of the image in the list of images in the
            dataset

        Returns:
            string, list -- id of the image, list of bounding boxes of objects
            in the image formatted as [xmin, ymin, xmax, ymax, class]
        """

        image_id = self.ids[index]
        target = self.dict_targets[int(image_id)]
        target = self.target_transform(target, 1, 1)

        return image_id, target

    def pull_tensor(self,
                    index):
        """Gets a tensor of the image found in position index of the list of
        images in the dataset

        Arguments:
            index {int} -- index of the image in the list of images in the
            dataset

        Returns:
            torch.Tensor -- tensor representation of the image
        """

        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


def save_results(all_boxes, dataset, results_path, output_txt):
    """
    Saves detection results in COCO-style JSON format and per-class text files.

    Arguments:
    - all_boxes: List of detections for each class.
    - dataset: The dataset instance containing image metadata.
    - results_path: Path to save the results.
    - output_txt: File to log output.

    Returns:
    - detection_file: Path to the saved detections.json file.
    """

    detections_list = []

    # Iterate through each class
    for class_i, class_name in enumerate(TOMATOD_CLASSES):
        class_id = TOMATOD_CLASSES_I[class_i]  # Get numerical class ID
        text = f'Writing results for {class_name} (Class {class_id})'
        write_print(output_txt, text)

        # Create a per-class results file
        filename = osp.join(results_path, f"{class_name}.txt")
        with open(filename, 'wt') as f:

            # Iterate over images in dataset
            for image_i, image_id in enumerate(dataset.ids):
                detections = all_boxes[class_i + 1][image_i]

                # Check if there are detections for this class in the image
                if len(detections) != 0:
                    for k in range(detections.shape[0]):
                        # Extract bounding box coordinates
                        x_min = float(detections[k, 0])
                        y_min = float(detections[k, 1])
                        x_max = float(detections[k, 2])
                        y_max = float(detections[k, 3])

                        width = x_max - x_min
                        height = y_max - y_min
                        score = float(detections[k, -1])  # Confidence score

                        # Save in text file (image_id, confidence, x_min, y_min, x_max, y_max)
                        output = f"{image_id} {score:.3f} {x_min:.1f} {y_min:.1f} {x_max:.1f} {y_max:.1f}\n"
                        f.write(output)

                        # Save in COCO-style JSON format
                        detections_list.append({
                            "image_id": int(image_id),
                            "category_id": int(class_id),
                            "bbox": [x_min, y_min, width, height],
                            "score": score
                        })

    # Save all detections in a single JSON file (COCO format)
    detection_file = osp.join(results_path, "detections.json")
    with open(detection_file, "w") as f:
        json.dump(detections_list, f, indent=4)

    # Print first 5 detections for verification
    # print("Sample Detections (COCO format):", detections_list[:5])
    write_print(output_txt, f"Detection results saved in {detection_file}")

    return detection_file
    
def iou(boxA, boxB):
    """Computes IoU (Intersection over Union) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_tomatod(results_path, dataset, output_txt, iou_threshold=0.5):
    """
    Evaluates the TomatoD dataset without using COCOeval.

    Arguments:
    - results_path: Path to the directory containing detection results.
    - dataset: TomatoD dataset instance.
    - output_txt: File to log output.
    - iou_threshold: IoU threshold for True Positives.

    Returns:
    - metrics: Dictionary containing AP, Precision, Recall, and mAP.
    """

    write_print(output_txt, "\nEvaluating TomatoD Dataset...\n")

    # Load ground truth annotations
    with open(dataset.annotation_path, "r") as f:
        coco_data = json.load(f)

    ground_truth = {str(ann["image_id"]): [] for ann in coco_data["annotations"]}
    for ann in coco_data["annotations"]:
        bbox = ann["bbox"]
        x_min, y_min, width, height = bbox
        ground_truth[str(ann["image_id"])].append({
            "bbox": [x_min, y_min, x_min + width, y_min + height],
            "category_id": ann["category_id"]
        })

    # Load detection results
    detection_file = osp.join(results_path, "detections.json")
    if not osp.exists(detection_file):
        write_print(output_txt, f"Error: Detection file {detection_file} not found!")
        return None

    with open(detection_file, "r") as f:
        detections = json.load(f)

    # Organize detections by image
    detected_objects = {str(ann["image_id"]): [] for ann in detections}
    for det in detections:
        detected_objects[str(det["image_id"])].append(det)

    # Compute Precision, Recall, and AP
    class_wise_metrics = {class_id: {"TP": 0, "FP": 0, "FN": 0, "confidences": []} for class_id in TOMATOD_CLASSES_I}

    for image_id in ground_truth.keys():
        gt_bboxes = ground_truth.get(image_id, [])
        pred_bboxes = detected_objects.get(image_id, [])

        matched_gt = set()

        for pred in sorted(pred_bboxes, key=lambda x: x["score"], reverse=True):
            pred_bbox = pred["bbox"]
            pred_class = pred["category_id"]
            max_iou = 0
            best_match = None

            for i, gt in enumerate(gt_bboxes):
                if gt["category_id"] == pred_class:
                    iou_score = iou(pred_bbox, gt["bbox"])
                    if iou_score > max_iou:
                        max_iou = iou_score
                        best_match = i

            if max_iou >= iou_threshold and best_match is not None and best_match not in matched_gt:
                class_wise_metrics[pred_class]["TP"] += 1
                matched_gt.add(best_match)
            else:
                class_wise_metrics[pred_class]["FP"] += 1

            class_wise_metrics[pred_class]["confidences"].append(pred["score"])

        for gt in gt_bboxes:
            if not any(iou(gt["bbox"], pred["bbox"]) >= iou_threshold for pred in pred_bboxes if pred["category_id"] == gt["category_id"]):
                class_wise_metrics[gt["category_id"]]["FN"] += 1

    # Compute Precision, Recall, and AP for each class
    metrics = {}
    ap_values = []
    for class_id, stats in class_wise_metrics.items():
        TP, FP, FN = stats["TP"], stats["FP"], stats["FN"]
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        precisions = []
        recalls = []
        sorted_confidences = sorted(stats["confidences"], reverse=True)

        for threshold in sorted_confidences:
            TP_thresh = sum(1 for c in stats["confidences"] if c >= threshold)
            FP_thresh = FP
            FN_thresh = FN

            precision_thresh = TP_thresh / (TP_thresh + FP_thresh) if (TP_thresh + FP_thresh) > 0 else 0
            recall_thresh = TP_thresh / (TP_thresh + FN_thresh) if (TP_thresh + FN_thresh) > 0 else 0

            precisions.append(precision_thresh)
            recalls.append(recall_thresh)

        AP = sum(precisions) / len(precisions) if precisions else 0
        ap_values.append(AP)

        metrics[f"AP (Class {class_id})"] = AP
        metrics[f"Precision (Class {class_id})"] = Precision
        metrics[f"Recall (Class {class_id})"] = Recall


    mAP = sum(ap_values) / len(ap_values) if ap_values else 0
    metrics["mAP"] = mAP

    # Save evaluation results
    with open(osp.join(results_path, "eval_results.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    return metrics

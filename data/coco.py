import os
import cv2
import json
import torch
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from utils.genutils import write_print
from pycocotools.coco import COCO as PYCOCO


COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')

COCO_CLASSES_I = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18,
                  19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                  37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52,
                  53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                  72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
                  88, 89, 90)


class COCOAnnotationTransform(object):
    """This class converts the data from Annotation CSV files to a list
    of [xmin, ymin, xmax, ymax, class]"""

    def __init__(self):
        """Class constructor for COCOAnnotationTransform"""
        super(COCOAnnotationTransform, self).__init__()

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


class COCO(Dataset):
    """COCO dataset

    Extends:
        Dataset
    """

    def __init__(self,
                 data_path,
                 year,
                 new_size,
                 mode,
                 image_transform):
        """Class constructor for COCO

        Arguments:
            data_path {string} -- path to the dataset
            year {string} -- contains the year of the dataset
            new_size {int} -- new height and width of the image
            mode {string} -- experiment mode - either train or test
            image_transform {object} -- produces different dataset
            augmentation techniques
        """

        super(COCO, self).__init__()

        self.data_path = data_path
        self.year = year
        self.new_size = new_size
        self.mode = mode
        self.image_transform = image_transform
        self.target_transform = COCOAnnotationTransform()
        class_to_index = dict(zip(COCO_CLASSES_I, range(len(COCO_CLASSES_I))))

        if self.mode == 'test':
            self.mode = 'val'

        mode_year = self.mode + self.year
        self.annotation_path = osp.join(self.data_path,
                                        'annotations',
                                        'instances_' + mode_year + '.json')
        self.image_path = osp.join(self.data_path,
                                   'images',
                                   self.mode + self.year,
                                   '{}.jpg')

        if self.mode == 'val':
            self.pycoco = PYCOCO(self.annotation_path)

        path = osp.join(self.data_path, 'annotations', self.mode + self.year)
        list_images = os.listdir(path)

        self.ids = []
        for image_id in list_images:
            image_id = image_id.split('.')[0]
            self.ids.append(image_id)

        self.dict_targets = {}
        with open(self.annotation_path) as file:
            data = json.load(file)
            for instance in data['annotations']:
                image_id = instance['image_id']
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

        # target_path = self.annotation_path.format(image_id)
        image_path = self.image_path.format(image_id)

        image = cv2.imread(image_path)
        height, width, _ = image.shape

        target = self.dict_targets[int(image_id)]
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


def save_results(all_boxes,
                 dataset,
                 results_path,
                 output_txt):

    detections_list = []

    # for each class
    for class_i, class_name in enumerate(COCO_CLASSES):

        text = 'Writing {:s} COCO results file'.format(class_name)
        write_print(output_txt, text)
        filename = osp.join(results_path, class_name + '.txt')

        with open(filename, 'wt') as f:

            # get detections for the class in an image
            for image_i, image_id in enumerate(dataset.ids):
                detections = all_boxes[class_i + 1][image_i]

                # if there are detections for the class in the image
                if len(detections) != 0:
                    for k in range(detections.shape[0]):
                        output = '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'

                        output = output.format(image_id,
                                               detections[k, -1],
                                               detections[k, 0],
                                               detections[k, 1],
                                               detections[k, 2],
                                               detections[k, 3])

                        f.write(output)

                        x1 = float(detections[k, 0])
                        y1 = float(detections[k, 1])
                        x2 = float(detections[k, 2]) - x1
                        y2 = float(detections[k, 3]) - y1

                        detections_list += [[int(image_id),
                                             x1,
                                             y1,
                                             x2,
                                             y2,
                                             float(detections[k, -1]),
                                             int(COCO_CLASSES_I[class_i])]]

    return np.array(detections_list)

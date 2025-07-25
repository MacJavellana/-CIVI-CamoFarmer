import os
import cv2
import torch
import pickle
import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from utils.genutils import write_print


VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor')


class VOCAnnotationTransform(object):
    """This class converts the data from Annotation XML files to a list
    of [xmin, ymin, xmax, ymax, class]"""

    def __init__(self,
                 keep_difficult=False):
        """Class constructor for VOCAnnotationTransform

        Keyword Arguments:
            keep_difficult {bool} -- determines if the dataset will contain
            difficult examples or not (default: {False})
        """

        super(VOCAnnotationTransform, self).__init__()
        self.class_to_index = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self,
                 target,
                 width,
                 height):
        """Executed when the class is called as a function

        Arguments:
            target {xml.etree.ElementTree.Element} -- annotation file
            width {int} -- Width of the corresponding image; Used for scaling
            the coordinates of bounding boxes
            height {int} -- Height of the corresponding image; Used for
            scaling the coordinates of bounding boxes

        Returns:
            list -- list of bounding boxes formatted as
            [xmin, ymin, xmax, ymax, class]
        """

        labels = []

        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1

            if not difficult or self.keep_difficult and difficult:
                name = obj.find('name').text.lower().strip()
                bbox = obj.find('bndbox')
                bndbox = [(int(bbox.find('xmin').text) - 1) / width,
                          (int(bbox.find('ymin').text) - 1) / height,
                          (int(bbox.find('xmax').text) - 1) / width,
                          (int(bbox.find('ymax').text) - 1) / height,
                          self.class_to_index[name]]
                labels += [bndbox]

        return labels


class PascalVOC(Dataset):
    """PascalVOC dataset

    Extends:
        Dataset
    """

    def __init__(self,
                 data_path,
                 image_sets,
                 new_size,
                 mode,
                 image_transform,
                 target_transform=VOCAnnotationTransform(),
                 keep_difficult=False):
        """Class constructor for PascalVOC

        Arguments:
            data_path {string} -- path to the dataset
            image_sets {tuple} -- contains the year and the subset - either
            trainval or test
            new_size {int} -- new height and width of the image
            mode {string} -- experiment mode - either train or test
            image_transform {object} -- produces different dataset
            augmentation techniques

        Keyword Arguments:
            target_transform {object} -- transforms data from Annotation XML
            files to a list of bounding boxes formatted as
            [xmin, ymin, xmax, ymax, class]
            (default: {VOCAnnotationTransform()})
            keep_difficult {bool} -- determines if the dataset will contain
            difficult examples (default: {False})
        """

        super(PascalVOC, self).__init__()

        self.data_path = data_path
        self.image_sets = image_sets
        self.new_size = new_size
        self.mode = mode
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult

        self.annotation_path = osp.join('{}',
                                        '{}',
                                        'Annotations',
                                        '{}.xml')
        self.image_path = osp.join('{}',
                                   '{}',
                                   'JPegImages',
                                   '{}.jpg')

        self.text_path = osp.join('{}',
                                  '{}',
                                  'ImageSets',
                                  'Main',
                                  '{}.txt')

        self.ids = []
        for(year, name) in self.image_sets:
            version = 'VOC{}'.format(year)
            path = osp.join(self.data_path, version)
            with open(self.text_path.format(path, name, name)) as f:
                for line in f:
                    self.ids.append((path, name, line.strip()))

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

        target_path = self.annotation_path.format(*image_id)
        image_path = self.image_path.format(*image_id)

        target = ET.parse(target_path).getroot()
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        if self.target_transform is not None:
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
        image_path = self.image_path.format(*image_id)
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
        target_path = self.annotation_path.format(*image_id)
        target = ET.parse(target_path).getroot()
        target = self.target_transform(target, 1, 1)

        return image_id[2], target

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

    # for each class
    for class_i, class_name in enumerate(VOC_CLASSES):

        text = 'Writing {:s} VOC results file'.format(class_name)
        write_print(output_txt, text)
        filename = osp.join(results_path, class_name + '.txt')

        with open(filename, 'wt') as f:

            # get detections for the class in an image
            for image_i, image_id in enumerate(dataset.ids):
                detections = all_boxes[class_i + 1][image_i]
                # print('HELLO', detections)

                # if there are detections for the class in the image
                if len(detections) != 0:
                    for k in range(detections.shape[0]):
                        output = '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'

                        # the VOCdevkit expects 1-based indices
                        output = output.format(image_id[2],
                                               detections[k, -1],
                                               detections[k, 0] + 1,
                                               detections[k, 1] + 1,
                                               detections[k, 2] + 1,
                                               detections[k, 3] + 1)

                        f.write(output)


def parse_annotation(file_name):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(file_name)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def voc_ap(recall,
           precision,
           use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for threshold in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= threshold) == 0:
                threshold_precision = 0
            else:
                threshold_precision = np.max(precision[recall >= threshold])
            ap = ap + threshold_precision / 11.

    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_eval(detection_path,
             path,
             annotation_path,
             list_path,
             class_name,
             cache_dir,
             output_txt,
             iou_threshold=0.5,
             use_07_metric=True):

    # create or get the cache_file
    if not osp.isdir(cache_dir):
        os.mkdir(cache_dir)
    cache_file = osp.join(cache_dir, 'annotations.pkl')

    # read list of images
    with open(list_path, 'r') as f:
        lines = f.readlines()
    image_names = [x.strip() for x in lines]

    # if cache_file does not exists
    if not osp.isfile(cache_file):
        targets = {}

        # per image, read annotations from XML file
        write_print(output_txt, 'Reading annotations')
        for i, image_name in enumerate(image_names):
            temp_path = annotation_path.format(path, 'test', image_name)
            targets[image_name] = parse_annotation(temp_path)

        # save annotations to cache_file
        temp_string = 'Saving cached annotations to {:s}\n'.format(cache_file)
        write_print(output_txt, temp_string)
        with open(cache_file, 'wb') as f:
            pickle.dump(targets, f)

    # else if cache_file exists
    else:
        with open(cache_file, 'rb') as f:
            targets = pickle.load(f)

    class_targets = {}
    n_positive = 0

    # get targets for objects with class equal to class_name in image_name
    for image_name in image_names:
        target = [x for x in targets[image_name] if x['name'] == class_name]
        bbox = np.array([x['bbox'] for x in target])
        difficult = np.array([x['difficult'] for x in target]).astype(np.bool)
        det = [False] * len(target)
        n_positive += sum(~difficult)
        class_targets[image_name] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

    # read detections from class_name.txt
    detection_file = detection_path.format(class_name)
    with open(detection_file, 'r') as f:
        lines = f.readlines()

    # if there are detections
    if any(lines) == 1:

        # get ids, confidences, and bounding boxes
        values = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in values]
        confidences = np.array([float(x[1]) for x in values])
        bboxes = np.array([[float(z) for z in x[2:]] for x in values])

        # sort by confidence
        sorted_index = np.argsort(-confidences)
        bboxes = bboxes[sorted_index, :]
        image_ids = [image_ids[x] for x in sorted_index]

        num_detections = len(image_ids)
        tp = np.zeros(num_detections)
        fp = np.zeros(num_detections)

        # go through detections and mark TPs and FPs
        for i in range(num_detections):

            # get target bounding box
            image_target = class_targets[image_ids[i]]
            bbox_target = image_target['bbox'].astype(float)

            # get detected bounding box
            bbox = bboxes[i, :].astype(float)
            overlap_max = -np.inf

            if bbox_target.size > 0:

                # get the overlapping region
                # compute the area of intersection
                x_min = np.maximum(bbox_target[:, 0], bbox[0])
                y_min = np.maximum(bbox_target[:, 1], bbox[1])
                x_max = np.minimum(bbox_target[:, 2], bbox[2])
                y_max = np.minimum(bbox_target[:, 3], bbox[3])
                width = np.maximum(x_max - x_min, 0.)
                height = np.maximum(y_max - y_min, 0.)
                intersection = width * height

                # get the area of the gt and the detection
                # compute the union
                area_bbox = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                area_bbox_target = ((bbox_target[:, 2] - bbox_target[:, 0])
                                    * (bbox_target[:, 3] - bbox_target[:, 1]))
                union = area_bbox + area_bbox_target - intersection

                # compute the iou
                iou = intersection / union
                overlap_max = np.max(iou)
                j_max = np.argmax(iou)

            # if the maximum overlap is over the overlap threshold
            if overlap_max > iou_threshold:
                # if it is not difficult
                if not image_target['difficult'][j_max]:
                    # if it is not yet detected, count as a true positive
                    if not image_target['det'][j_max]:
                        tp[i] = 1.
                        image_target['det'][j_max] = 1
                    # else, count as a false positive
                    else:
                        fp[i] = 1.

            # else, count as a false positive
            else:
                fp[i] = 1.

        # compute precision and recall
        # avoid divide by zero if the first detection matches a difficult gt
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / float(n_positive)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        ap = voc_ap(recall=recall,
                    precision=precision,
                    use_07_metric=use_07_metric)

    else:
        recall = -1.
        precision = -1.
        ap = -1.

    return recall, precision, ap


def do_python_eval(results_path,
                   dataset,
                   output_txt,
                   mode,
                   iou_threshold,
                   use_07_metric):

    # annotation cache directory
    cache_dir = osp.join(results_path,
                         'annotations_cache')

    # path to XML annotation folder
    annotation_path = dataset.annotation_path

    # path to VOC + year
    path = osp.join(dataset.data_path,
                    'VOC{}'.format(dataset.image_sets[0][0]))

    # text file containing the list of (test) images
    list_path = dataset.text_path.format(path, mode, mode)

    # The PASCAL VOC metric changed in 2010
    write_print(output_txt, '\nVOC07 metric? '
                + ('Yes\n' if use_07_metric else 'No\n'))

    # for each class, compute the recall, precision, and ap
    aps = []
    for class_name in VOC_CLASSES:
        detection_path = osp.join(results_path, class_name + '.txt')
        recall, precision, ap = voc_eval(detection_path=detection_path,
                                         path=path,
                                         annotation_path=annotation_path,
                                         list_path=list_path,
                                         class_name=class_name,
                                         cache_dir=cache_dir,
                                         output_txt=output_txt,
                                         iou_threshold=iou_threshold,
                                         use_07_metric=use_07_metric)
        aps += [ap]

        write_print(output_txt, 'AP for {} = {:.4f}'.format(class_name, ap))

        pickle_file = osp.join(results_path, class_name + '_pr.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump({'rec': recall, 'prec': precision, 'ap': ap}, f)

    write_print(output_txt, 'Mean AP = {:.4f}'.format(np.mean(aps)))

    return aps, np.mean(aps)

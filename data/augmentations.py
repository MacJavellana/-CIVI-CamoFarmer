import cv2
import torch
import numpy as np
from numpy import random


def intersect(box_a,
              box_b):

    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    intersection = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)

    return intersection[:, 0] * intersection[:, 1]


def jaccard_numpy(box_a,
                  box_b):

    intersection = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    union = area_a + area_b - intersection

    return intersection / union


class Compose(object):
    """This class applies a list of transformation to an image."""

    def __init__(self,
                 transforms):
        """Class constructor of Compose

        Arguments:
            transforms {list} -- list of transformation to be applied to the
            image
        """

        super(Compose, self).__init__()
        self.transforms = transforms

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- transformed image pixels,
            list of bounding boxes of objects in the image formatted as
            [xmin, ymin, xmax, ymax], list of the corresponding classes of the
            bounding boxes
        """

        for transform in self.transforms:
            image, boxes, labels = transform(image, boxes, labels)

        return image, boxes, labels


class ConvertToFloat(object):
    """This class casts an np.ndarray to floating-point data type."""

    def __init__(self):
        """Class constructor for ConvertToFloat"""

        super(ConvertToFloat, self).__init__()

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- image pixels casted to
            floating-point data type, list of bounding boxes of objects in
            the image formatted as [xmin, ymin, xmax, ymax], list of the
            corresponding classes of the bounding boxes
        """

        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    """This class subtracts the mean from the pixel values of the image"""

    def __init__(self,
                 mean):
        """Class constructor for SubtractMeans

        Arguments:
            mean {tuple} -- mean
        """

        super(SubtractMeans, self).__init__()
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- image pixels subtracted
            by the mean, list of bounding boxes of objects in the image
            formatted as [xmin, ymin, xmax, ymax], list of the corresponding
            classes of the bounding boxes
        """

        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    """This class converts the coordinates of bounding boxes to
    non-scaled values"""

    def __init__(self):
        """Class constructor for ToAbsoluteCoords"""

        super(ToAbsoluteCoords, self).__init__()

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- image pixels, list of
            bounding boxes of objects in the image formatted as
            [xmin, ymin, xmax, ymax] converted to non-scaled
            coordinates, list of the corresponding classes of the
            bounding boxes
        """

        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    """This class converts the coordinates of bounding boxes to
    scaled values with respect to the width and the height of the image"""

    def __init__(self):
        """Class constructor for ToPercentCoords"""

        super(ToPercentCoords, self).__init__()

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- image pixels, list of
            bounding boxes of objects in the image formatted as
            [xmin, ymin, xmax, ymax] converted to scaled coordinates,
            list of the corresponding classes of the bounding boxes
        """

        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    """This class resizes the image to output size"""

    def __init__(self,
                 size=300):
        """Class constructor for Resize

        Keyword Arguments:
            size {int} -- output size (default: {300})
        """

        super(Resize, self).__init__()
        self.size = size

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- pixels of the resized image,
            list of bounding boxes of objects in the image formatted as
            [xmin, ymin, xmax, ymax], list of the corresponding classes of the
            bounding boxes
        """

        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class RandomSaturation(object):
    """This class adjusts the saturation of the image. This expects an image
    in HSV color space, where the 2nd channel (saturation channel) should have
    values between 0.0 to 1.0"""

    def __init__(self,
                 lower=0.5,
                 upper=1.5):
        """Class constructor for RandomSaturation

        Keyword Arguments:
            lower {int} -- lower bound of the interval used in generating
            a random number from a uniform distribution to adjust saturation
            (default: {0.5})
            upper {number} -- upper bound of the interval used in generating
            a random number from a uniform distribution to adjust saturation.
            (default: {1.5})
        """

        super(RandomSaturation, self).__init__()
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, 'contrast upper must be >= lower.'
        assert self.lower >= 0, 'contrast lower must be non-negative.'

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray. This
            should be in HSV color space.

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- image pixels in HSV color
            space with adjusted saturation channel, list of bounding boxes of
            objects in the image formatted as [xmin, ymin, xmax, ymax],
            list of the corresponding classes of the bounding boxes
        """

        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

            # limits the value of the saturation channel to 1.0
            image[:, :, 1] = np.clip(image[:, :, 1], a_min=0.0, a_max=1.0)

        return image, boxes, labels


class RandomHue(object):
    """This class adjusts the hue of the image. This expects an image
    in HSV color space, where the 1st channel (hue channel) should have
    values between 0.0 to 360.0"""

    def __init__(self,
                 delta=18.0):
        """Class constructor for RandomSaturation

        Keyword Arguments:
            delta {int} -- lower bound and upper bound of the interval used
            in generating a random number from a uniform distribution to adjust
            hue (default: {18.0})
        """

        super(RandomHue, self).__init__()
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray. This
            should be in HSV color space.

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- image pixels in HSV color
            space with adjusted hue channel, list of bounding boxes of
            objects in the image formatted as [xmin, ymin, xmax, ymax],
            list of the corresponding classes of the bounding boxes
        """

        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)

            # limits the value of the hue channel to 360
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

        return image, boxes, labels


class RandomLightingNoise(object):
    """This class randomly swaps the channels of the image to create a
    lighting noise effect. This class calls the class SwapChannels."""

    def __init__(self):
        """Class constructor for RandomLightingNoise"""

        super(RandomLightingNoise, self).__init__()
        self.permutations = ((0, 1, 2), (0, 2, 1),
                             (1, 0, 2), (1, 2, 0),
                             (2, 0, 1), (2, 1, 0))

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- pixels of the image with
            swapped channels, list of bounding boxes of objects in the image
            formatted as [xmin, ymin, xmax, ymax], list of the corresponding
            classes of the bounding boxes
        """

        if random.randint(2):
            swap = self.permutations[random.randint(len(self.permutations))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)

        return image, boxes, labels


class ConvertColor(object):
    """This class converts the image to another color space. This class
    supports the conversion from BGT to HSV color space and vice-versa."""

    def __init__(self,
                 current='BGR',
                 transform='HSV'):
        """Class constructor for ConvertColor

        Keyword Arguments:
            current {str} -- the input color space of the image
            (default: {'BGR'})
            transform {str} -- the output color space of the image
            (default: {'HSV'})
        """

        super(ConvertColor, self).__init__()
        self.current = current
        self.transform = transform

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- pixels of the image converted
            to the output color space, list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax], list of the
            corresponding classes of the bounding boxes
        """

        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError

        return image, boxes, labels


class RandomContrast(object):
    """This class adjusts the contrast of the image. This multiplies a random
    constant to the pixel values of the image."""

    def __init__(self,
                 lower=0.5,
                 upper=1.5):
        """Class constructor for RandomContrast

        Keyword Arguments:
            lower {int} -- lower bound of the interval used in generating
            a random number from a uniform distribution to adjust contrast
            (default: {0.5})
            upper {number} -- upper bound of the interval used in generating
            a random number from a uniform distribution to adjust contrast
            (default: {1.5})
        """

        super(RandomContrast, self).__init__()
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, 'contrast upper must be >= lower.'
        assert self.lower >= 0, 'contrast lower must be non-negative.'

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- image pixels with adjusted
            contrast, list of bounding boxes of objects in the image
            formatted as [xmin, ymin, xmax, ymax], list of the corresponding
            classes of the bounding boxes
        """

        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)

            # multiplies the random constant to the pixel values
            image *= alpha

        return image, boxes, labels


class RandomBrightness(object):
    """This class adjusts the brightness of the image. This adds a random
    constant to the pixel values of the image."""

    def __init__(self,
                 delta=32.0):
        """Class constructor for RandomBrightness

        Keyword Arguments:
            delta {int} -- lower bound and upper bound of the interval used
            in generating a random number from a uniform distribution to adjust
            brightness (default: {32.0})
        """

        super(RandomBrightness, self).__init__()
        assert delta >= 0.0 and delta <= 255.0
        self.delta = delta

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- image pixels with adjusted
            brightness, list of bounding boxes of objects in the image
            formatted as [xmin, ymin, xmax, ymax], list of the corresponding
            classes of the bounding boxes
        """

        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)

            # adds the random constant to the pixel values
            image += delta

        return image, boxes, labels


class ToCV2Image(object):
    """This class converts the torch.Tensor representation of an image to
    np.ndarray. The channels of the image are also converted from RGB to
    BGR"""

    def __init__(self):
        """Class constructor for ToCV2Image"""

        super(ToCV2Image, self).__init__()

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {torch.Tensor} -- image pixels represented as torch.Tensor.

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- image pixels represented
            as np.ndarray, list of bounding boxes of objects in the image
            formatted as [xmin, ymin, xmax, ymax], list of the corresponding
            classes of the bounding boxes
        """

        # permute() is used to switch the channels from RGB to BGR
        image = image.cpu().numpy().astype(np.float32).transpose((2, 1, 0))
        return image, boxes, labels


class ToTensor(object):
    """This class converts the np.ndarray representation of an image to
    torch.Tensor. The channels of the image are also converted from BGR to
    RGB"""

    def __init__(self):
        """Class constructor for ToTensor"""

        super(ToTensor, self).__init__()

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            torch.Tensor, np.ndarray, np.ndarray -- image pixels represented
            as torch.Tensor, list of bounding boxes of objects in the image
            formatted as [xmin, ymin, xmax, ymax], list of the corresponding
            classes of the bounding boxes
        """

        # permute() is used to switch the channels from BGR to RGB
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 1, 0)
        return image, boxes, labels


class RandomSampleCrop(object):

    def __init__(self):

        super(RandomSampleCrop, self).__init__()
        self.sample_options = [0.1, 0.3, 0.7, 0.9, None]

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        height, width, _ = image.shape

        while True:
            mode = random.choice([0, 1])

            if mode == 0:
                return image, boxes, labels

            else:
                min_iou = random.choice(self.sample_options)
                if min_iou is None:
                    min_iou = float('-inf')
                max_iou = float('inf')

                for _ in range(50):
                    current_image = image

                    w = random.uniform(0.3 * width, width)
                    h = random.uniform(0.3 * height, height)

                    if h / w < 0.5 or h / w > 2:
                        continue

                    left = random.uniform(width - w)
                    top = random.uniform(height - h)

                    x1 = int(left)
                    y1 = int(top)
                    x2 = int(left + w)
                    y2 = int(top + h)
                    crop = np.array([x1, y1, x2, y2])

                    overlap = jaccard_numpy(boxes, crop)

                    if overlap.min() < min_iou and max_iou < overlap.max():
                        continue

                    current_image = current_image[y1:y2, x1:x2, :]
                    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                    m1 = (x1 < centers[:, 0]) * (y1 < centers[:, 1])
                    m2 = (x2 > centers[:, 0]) * (y2 > centers[:, 1])
                    mask = m1 * m2

                    if not mask.any():
                        continue

                    current_boxes = boxes[mask, :].copy()
                    current_labels = labels[mask]

                    current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                      crop[:2])
                    current_boxes[:, :2] -= crop[:2]

                    current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                      crop[2:])
                    current_boxes[:, 2:] -= crop[:2]

                    return current_image, current_boxes, current_labels


class Expand(object):
    """This class produces an image"""

    def __init__(self,
                 mean):

        super(Expand, self).__init__()
        self.mean = mean

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):

        if random.randint(2):
            height, width, channels = image.shape
            ratio = random.uniform(1, 4)
            left = random.uniform(0, width * ratio - width)
            top = random.uniform(0, height * ratio - height)

            dimensions = (int(height * ratio), int(width * ratio), channels)
            expand_image = np.zeros(dimensions, dtype=image.dtype)
            expand_image[:, :, :] = self.mean

            x1 = int(left)
            x2 = int(left + width)
            y1 = int(top)
            y2 = int(top + height)
            expand_image[y1:y2, x1:x2] = image
            image = expand_image

            if boxes is not None:
                boxes = boxes.copy()
                boxes[:, :2] += (x1, y1)
                boxes[:, 2:] += (x1, y1)

        return image, boxes, labels


class RandomMirror(object):
    """This class randomly flips the image horizontally. This also flips the
    coordinate of the bounding boxes of the objects."""

    def __init__(self):
        """Class constructor for RandomMirror"""

        super(RandomMirror, self).__init__()

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- image pixels flipped
            horizontally, list of bounding boxes of objects in the image
            formatted as [xmin, ymin, xmax, ymax] and flipped horizontally,
            list of the corresponding classes of the bounding boxes
        """

        _, width, _ = image.shape

        if random.randint(2):
            image = image[:, ::-1, :]
            if boxes is not None:
                boxes = boxes.copy()

                # flip the x coordinates of the bounding boxes
                # using the width of the image
                boxes[:, 0::2] = width - boxes[:, 2::-2]

        return image, boxes, labels


class SwapChannels(object):
    """This class swaps the channels of the image"""

    def __init__(self,
                 swaps):
        """Class constructor for SwapChannels

        Arguments:
            swaps {tuple} -- new order of the channels
        """

        super(SwapChannels, self).__init__()
        self.swaps = swaps

    def __call__(self,
                 image):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Returns:
            np.ndarray -- pixels of the image with swapped channels
        """

        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    """This class applies different transformation to the image. This includes
    adjustment of contrast, saturation, hue, and brightness, and the
    switching of channels."""

    def __init__(self):
        """Class constructor for PhotometricDistort"""

        super(PhotometricDistort, self).__init__()
        self.pd = [RandomContrast(),
                   ConvertColor(transform='HSV'),
                   RandomSaturation(),
                   RandomHue(),
                   ConvertColor(current='HSV', transform='BGR'),
                   RandomContrast()]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self,
                 image,
                 boxes,
                 labels):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax]
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- image pixels applied with
            different random transformations, list of bounding boxes of
            objects in the image formatted as [xmin, ymin, xmax, ymax],
            list of the corresponding classes of the bounding boxes
        """

        image = image.copy()
        image, boxes, labels = self.rand_brightness(image, boxes, labels)

        # applies RandomContrast() as the first transformation
        if random.randint(2):
            distort = Compose(self.pd[:-1])

        # applies RandomContrast() as the last transformation
        else:
            distort = Compose(self.pd[1:])
        image, boxes, labels = distort(image, boxes, labels)

        return self.rand_light_noise(image, boxes, labels)


class Augmentations(object):
    """This class applies different augmentation techniques to the image.
    This is used for training the model."""

    def __init__(self,
                 size,
                 mean):
        """Class constructor for Augmentations

        Arguments:
            size {int} -- output size
            mean {tuple} -- mean
        """

        super(Augmentations, self).__init__()
        self.size = size
        self.mean = mean
        self.augment = Compose([ConvertToFloat(),
                                ToAbsoluteCoords(),
                                PhotometricDistort(),
                                Expand(self.mean),
                                RandomSampleCrop(),
                                RandomMirror(),
                                ToPercentCoords(),
                                Resize(self.size),
                                SubtractMeans(self.mean)])

    def __call__(self,
                 image,
                 boxes,
                 labels):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax]
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- image pixels applied with
            different augmentation techniques, list of bounding boxes of
            objects in the image formatted as [xmin, ymin, xmax, ymax],
            list of the corresponding classes of the bounding boxes
        """
        #         # ✅ Print original image size and bounding boxes
        # print(f"\n🟢 [Before Augmentation] Image Size: {image.shape}")
        # print(f"🟢 [Before Augmentation] Bounding Boxes:")
        # for bbox in boxes:
        #     print(f"  {bbox}")  # Format: [xmin, ymin, xmax, ymax]

        # # Apply augmentations (including resizing)
        # image, boxes, labels = self.augment(image, boxes, labels)

        # # ✅ Print resized image size and transformed bounding boxes
        # print(f"\n🔵 [After Augmentation] Image Size: {image.shape}")
        # print(f"🔵 [After Augmentation] Bounding Boxes:")
        # for bbox in boxes:
        #     print(f"  {bbox}")  # Format: [xmin, ymin, xmax, ymax]

        return self.augment(image, boxes, labels)


class BaseTransform(object):
    """This class applies different base transformation techniques to the
    image. This includes resizing the image and subtracting the mean from the
    image pixels. This is used for testing the model."""

    def __init__(self,
                 size,
                 mean):
        """Class constructor for BaseTransform

        Arguments:
            size {int} -- output size
            mean {tuple} -- mean
        """

        super(BaseTransform, self).__init__()
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self,
                 image,
                 boxes=None,
                 labels=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            boxes {np.ndarray} -- list of bounding boxes of objects in the
            image formatted as [xmin, ymin, xmax, ymax] (default: {None})
            labels {np.ndarray} -- list of the corresponding classes of the
            bounding boxes (default: {None})

        Returns:
            np.ndarray, np.ndarray, np.ndarray -- image pixels applied with
            different augmentation techniques, list of bounding boxes of
            objects in the image formatted as [xmin, ymin, xmax, ymax],
            list of the corresponding classes of the bounding boxes
        """

        dimensions = (self.size, self.size)
        image = cv2.resize(image, dimensions).astype(np.float32)
        image -= self.mean
        image = image.astype(np.float32)
        return image, boxes, labels

import torch
from torch.autograd import Function
from utils.bbox_utils import decode


class Detect(Function):

    @staticmethod
    def forward(ctx,
                class_count,
                conf_data,
                loc_data,
                anchors,
                variance=[0.1, 0.2]):

        loc_data = loc_data.data
        conf_data = conf_data.data
        anchors_data = anchors.data
        batch_size = loc_data.shape[0]

        num_anchors = anchors_data.shape[0]
        boxes = torch.zeros(batch_size, num_anchors, 4)
        scores = torch.zeros(batch_size, num_anchors, class_count)

        if batch_size == 1:
            conf_preds = conf_data.unsqueeze(0)
        else:
            conf_preds = conf_data.view(batch_size, num_anchors, class_count)
            boxes.expand(batch_size, num_anchors, 4)
            scores.expand(batch_size, num_anchors, class_count)

        for i in range(batch_size):
            decoded_boxes = decode(loc_data[i], anchors_data, variance)
            conf_scores = conf_preds[i].clone()

            boxes[i] = decoded_boxes
            scores[i] = conf_scores

        return boxes, scores

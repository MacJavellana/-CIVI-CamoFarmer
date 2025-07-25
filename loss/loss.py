from loss.multibox_loss import MultiBoxLoss


def get_loss(config):

    loss = None

    if config['loss_config'] == 'multibox':
        loss = MultiBoxLoss(class_count=config['class_count'],
                            iou_threshold=config['iou_threshold'],
                            pos_neg_ratio=config['pos_neg_ratio'],
                            use_gpu=config['use_gpu'])

    return loss

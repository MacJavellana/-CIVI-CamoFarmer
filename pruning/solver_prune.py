"""
Pruning Script for SSD
------------------------------------------

This module provides the main training and evaluation solver for SSD models
with integrated L1 unstructured pruning capabilities.

Classes:
    Solver: Main solver class for training and evaluating pruned SSD models.
"""

import os
import time
import torch
import pickle
import datetime
import numpy as np
from tqdm import tqdm
import os.path as osp
import torch.optim as optim
from torchvision.ops import nms
from torch.optim.lr_scheduler import MultiStepLR
from utils.genutils import to_var, write_print, mkdir, load_pretrained_model
import torch.nn.utils.prune as prune
import copy
from utils.timer import Timer
from loss.loss import get_loss
from models.model import get_model
from layers.anchor_box import AnchorBox
from data.coco import save_results as coco_save
from data.pascal_voc import save_results as voc_save
from data.pascal_voc import do_python_eval as do_voc_eval
from pycocotools.cocoeval import COCOeval as do_coco_eval

from data.tomatod import save_results as tomatod_save
from data.tomatod import evaluate_tomatod as do_tomatod_eval 

from data.ccrop import do_python_eval as do_ccrop_eval
from data.ccrop import save_results as ccrop_save

from data.camocrops import do_python_eval as do_camocrops_eval
from data.camocrops import save_results as camocrops_save

from torchinfo import summary

import io
import contextlib


class Solver(object):
    """Main solver for training and evaluating SSD models with L1 unstructured pruning.
    
    Provides comprehensive functionality for training, evaluation, and pruning
    of SSD models on various datasets.
    
    """

    DEFAULTS = {}

    def __init__(self, version, data_loader, config, output_txt):
        """Initialize the Solver with configuration and data loader.
        
        Args:
            version (str): Version identifier for the model.
            data_loader: Data loader object for training/validation data.
            config (dict): Configuration dictionary containing training parameters.
            output_txt: Output text handler for logging.
        """

        super(Solver, self).__init__()
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.version = version
        self.data_loader = data_loader
        self.config = config
        self.output_txt = output_txt

        self.build_model()

                # Start with a pre-trained model
        if self.pretrained_model is not None:
            load_pretrained_model(model=self.model,
                                model_save_path=self.model_save_path,
                                pretrained_model=self.pretrained_model,
                                output_txt=self.output_txt)

            toprune = self.model

            # Count non-zero params before pruning
            nonzero_params_before = self.count_nonzero_params(toprune)
            print(f"Non-zero parameters before pruning: {nonzero_params_before}")

            # Apply pruning if enabled in config
            if self.config.get('prune_model', False):
                prune_amount = self.config.get('prune_amount', 0.1)  # Default 10%
                prune_mode = self.config.get('prune_mode', 'all')
                self.apply_pruning(toprune, prune_amount, prune_mode)

                # Count non-zero params after pruning
                nonzero_params_after = self.count_nonzero_params(toprune)
                print(f"Non-zero parameters after pruning: {nonzero_params_after}")
                print(f"Number of pruned parameters: {nonzero_params_before - nonzero_params_after}")

                # Construct filename: <dataset>_<mode>_pruned_model_<amount>.pth
                dataset_name = self.config.get('dataset', 'model')  # fallback if not present
                prune_tag = f"{dataset_name}_{prune_mode}_pruned_model_{int(prune_amount * 100)}.pth"
                pruned_model_path = osp.join(self.model_save_path, prune_tag)

                self.save_pruned_model(toprune, pruned_model_path)
                # print(f"Pruned model saved at: {pruned_model_path}")








        # changes made to fix device mismatches (10/03/2025)
        elif self.coco_weights is None:
            use_gpu = torch.cuda.is_available() and self.use_gpu  # Check if GPU should be used
            if config['model'] == 'SSD':
                # print("Using VGG weights. (vgg16_reducedfc.pth)")  
                self.model.init_weights(self.model_save_path, self.basenet)
            elif config['model'] == 'SSD-EfficientNet':
                # print("Using EfficientNet weights. (efficientnet_b0.pth)")  
                self.model.init_weights(self.model_save_path, self.basenet)
            elif config['model'] == 'SSD-MobileNet':
                # print("Using MobileNet weights. (mobilenet_v2.pth)")  
                self.model.init_weights(self.model_save_path, self.basenet)
            elif config['model'] == 'SSD-ShuffleNet':
                # print("Using ShuffleNet weights. (shufflenet_v2_x1_0.pth)")  
                self.model.init_weights(self.model_save_path, self.basenet)
            else:
                print("Skipping VGG weights. Using Custom backbone with ImageNet-1K pretrained weights.")
                self.model.init_weights(self.model_save_path,
                                    self.basenet)

    def count_nonzero_params(self, model):
        """Count the number of non-zero parameters in the model.
        
        Args:

            model: PyTorch model to count parameters for.
            
        Returns:

            int: Number of non-zero parameters with requires_grad=True.
        """
        nonzero_params = sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad)
        return nonzero_params


    def save_pruned_model(self, model, path):
        """Save pruned model to the specified path.
        
        Args:

            model: PyTorch model to save.
            path (str): File path to save the model state dictionary.
        """
        torch.save(model.state_dict(), path)
        print(f"Pruned model saved to {path}")


    def apply_pruning(self, model, prune_amount=0.1, mode=None):
        """Apply L1 unstructured pruning to Conv2d layers of SSD300.

        Args:
            model: The SSD model to prune.
            prune_amount (float, optional): Proportion of weights to prune. Defaults to 0.1 (10%).
            mode (str, optional): Target component for pruning. Options are 'backbone', 'neck', 
                'head', or None/'all' for entire model. Defaults to None.
                
        Raises:
            ValueError: If mode is not one of the supported options.
        """
        import torch.nn as nn

        # Determine which modules to prune
        if mode is None or mode == "all":
            modules = [m for _, m in model.named_modules()]
            label = "entire model"
        elif mode == "backbone":
            modules = model.base
            label = "backbone"
        elif mode == "neck":
            modules = model.extras
            label = "neck"
        elif mode == "head":
            modules = list(model.class_head) + list(model.loc_head)
            label = "head"
        else:
            raise ValueError("Mode must be one of: 'backbone', 'neck', 'head', or None/'all'")

        # Apply pruning
        count = 0
        for i, layer in enumerate(modules):
            if isinstance(layer, nn.Conv2d):
                prune.l1_unstructured(layer, name='weight', amount=prune_amount)
                prune.remove(layer, 'weight')
                count += 1
                print(f"Pruned {label}[{i}] - {layer.__class__.__name__}")
        
        print(f"Pruning applied to {label}. {prune_amount * 100:.1f}% of weights in {count} Conv2d layers were pruned.")


    def build_model(self):
        """Instantiate the model, loss criterion, and optimizer.
        
        Initializes anchor boxes, model architecture, loss function, and optimizer
        based on the configuration settings.
        """

        # instantiate anchor boxes
        anchor_boxes = AnchorBox(new_size=self.new_size,
                                 config=self.anchor_config,
                                 scale_initial=self.scale_initial,
                                 scale_min=self.scale_min,
                                 scale_max=self.scale_max)
        self.anchor_boxes = anchor_boxes.get_boxes()

        if torch.cuda.is_available() and self.use_gpu:
            self.anchor_boxes = self.anchor_boxes.cuda()

        # instatiate model
        self.model = get_model(config=self.config,
                               anchors=self.anchor_boxes,
                               output_txt=self.output_txt)


        # instatiate loss criterion
        self.criterion = get_loss(config=self.config)

        # instatiate optimizer
        self.optimizer = optim.SGD(params=self.model.parameters(),
                                   lr=self.lr,
                                   momentum=self.momentum,
                                   weight_decay=self.weight_decay)

        self.scheduler = MultiStepLR(self.optimizer,
                                     milestones=self.learning_sched,
                                     gamma=self.sched_gamma)

        # print network
        self.print_network(self.model)

        # use gpu if enabled
        if torch.cuda.is_available() and self.use_gpu:
            self.model.cuda()
            self.criterion.cuda()
        
        # debugging (backbone replacement 10/03/2025)
        print(f"Generated number of anchors: {self.anchor_boxes.shape[0]}")
        print(f"Model device: {next(self.model.parameters()).device}")

    def print_network(self, model):
        """
        Prints the structure of the network and the total number of parameters + GFLOPs
        """
        num_params = 0

        # Compute FLOPs with PyTorch profiler (batch_size=1) 12/03/2025 (5:44am)
        # changed back to torchinfo 12/03/2025 (6:47am)
        model_summary = summary(model, input_size=(1, self.input_channels, self.new_size, self.new_size), verbose=0)

        total_flops = (model_summary.total_mult_adds * 2) / 1e9 

        for p in model.parameters():
            num_params += p.numel()
        write_print(self.output_txt, str(model))
        write_print(self.output_txt,
                    'The number of parameters: {}'.format(num_params))
        write_print(self.output_txt,
                    'The number of FLOPs: {:.3f} GFLOPs'.format(total_flops))
        
    def print_params(self, model):
        """
        Prints the structure of the network and the total number of parameters + GFLOPs
        """
        num_params = 0

        # Compute FLOPs with PyTorch profiler (batch_size=1) 12/03/2025 (5:44am)
        # changed back to torchinfo 12/03/2025 (6:47am)
        model_summary = summary(model, input_size=(1, self.input_channels, self.new_size, self.new_size), verbose=0)

        total_flops = (model_summary.total_mult_adds * 2) / 1e9 

        for p in model.parameters():
            num_params += p.numel()
        write_print(self.output_txt,
                    'The number of parameters: {}'.format(num_params))
        write_print(self.output_txt,
                    'The number of FLOPs: {:.3f} GFLOPs'.format(total_flops))

    # def load_pretrained_model(self,
    #                           model,
    #                           model_save_path,
    #                           pretrained_model):
    #     """
    #     loads a pre-trained model from a .pth file
    #     """
    #     model.load_state_dict(torch.load(os.path.join(
    #         model_save_path, '{}.pth'.format(pretrained_model))))
    #     write_print(self.output_txt,
    #                 'loaded trained model {}'.format(pretrained_model))

    def adjust_learning_rate(self,
                             optimizer,
                             gamma,
                             step):
        """Sets the learning rate to the initial LR decayed by 10 at every
            specified step
       
        """
        lr = self.lr * (gamma ** (step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def print_loss_log(self,
                       start_time,
                       iters_per_epoch,
                       e,
                       i,
                       class_loss,
                       loc_loss,
                       loss):
        """
        Prints the loss and elapsed time for each epoch
        """

        """
        Prints the loss and elapsed time for each epoch
        """
        total_iter = self.num_epochs * iters_per_epoch
        cur_iter = e * iters_per_epoch + i

        elapsed = time.time() - start_time
        total_time = (total_iter - cur_iter) * elapsed / (cur_iter + 1)
        epoch_time = (iters_per_epoch - i) * elapsed / (cur_iter + 1)

        epoch_time = str(datetime.timedelta(seconds=epoch_time))
        total_time = str(datetime.timedelta(seconds=total_time))
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed {}/{} -- {}, Epoch [{}/{}], Iter [{}/{}], " \
              "class_loss: {:.4f}, loc_loss: {:.4f}, " \
              "loss: {:.4f}".format(elapsed,
                                    epoch_time,
                                    total_time,
                                    e + 1,
                                    self.num_epochs,
                                    i + 1,
                                    iters_per_epoch,
                                    class_loss.item(),
                                    loc_loss.item(),
                                    loss.item())

        write_print(self.output_txt, log)

    def save_model(self, e):
        """
        Saves a model per e epoch
        """
        path = os.path.join(
            self.model_save_path,
            '{}/{}.pth'.format(self.version, e + 1)
        )

        torch.save(self.model.state_dict(), path)

    # def model_step(self,
    #                images,
    #                targets,
    #                count):
        """
        A step for each iteration
        """

        """
        if count == 0:
            # update parameters
            # self.optimizer.step()
            self.scaler.step(self.optimizer)

            # empty the gradients of the model through the optimizer
            self.optimizer.zero_grad()

            self.scaler.update()

            count = self.batch_multiplier

        with autocast():

            # forward pass
            class_preds, loc_preds = self.model(images)

            # compute loss
            class_targets = [target[:, -1] for target in targets]
            loc_targets = [target[:, :-1] for target in targets]
            losses = self.criterion(class_preds=class_preds,
                                    class_targets=class_targets,
                                    loc_preds=loc_preds,
                                    loc_targets=loc_targets,
                                    anchors=self.anchor_boxes)

            class_loss, loc_loss, loss = losses

        # compute gradients using back propagation
        # loss.backward()
        class_loss = class_loss / self.batch_multiplier
        loc_loss = loc_loss / self.batch_multiplier
        loss = loss / self.batch_multiplier
        self.scaler.scale(loss).backward()
        count = count - 1

        # return loss
        return class_loss, loc_loss, loss, count
    """

    def model_step(self,
                   images,
                   targets,
                   count):
        """
        Signifies a step for each iteration
        """

        # empty the gradients of the model through the optimizer
        self.optimizer.zero_grad()

        # with autocast():

        # forward pass
        class_preds, loc_preds = self.model(images)

        # compute loss
        class_targets = [target[:, -1] for target in targets]
        loc_targets = [target[:, :-1] for target in targets]
        losses = self.criterion(class_preds=class_preds,
                                class_targets=class_targets,
                                loc_preds=loc_preds,
                                loc_targets=loc_targets,
                                anchors=self.anchor_boxes)

        class_loss, loc_loss, loss = losses

        # compute gradients using back propagation
        loss.backward()
        # self.scaler.scale(loss).backward()

        # update parameters
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        self.optimizer.step()
        # self.scaler.step(self.optimizer)

        # self.scaler.update()

        # return loss
        return class_loss, loc_loss, loss, count

    def train(self):
        """
        Training process.
        """

        # set model in training mode
        self.model.train()

        self.losses = []
        count = self.batch_multiplier

        iters_per_epoch = len(self.data_loader)

        # start with a trained model if exists
        if self.pretrained_model:
            start = 0 #int(self.pretrained_model.split('/')[-1])
        else:
            start = 0

        sched = 0

        if self.warmup_epoch != 0:
            self.lr /= 10
            write_print(self.output_txt,
                        'Learning rate reduced to ' + str(self.lr))
            self.adjust_learning_rate(optimizer=self.optimizer,
                                      gamma=self.sched_gamma,
                                      step=sched)

        # start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            for i, (images, targets) in enumerate(tqdm(self.data_loader)):
                images = to_var(images, self.use_gpu)
                targets = [to_var(target, self.use_gpu) for target in targets]

                class_loss, loc_loss, loss, count = self.model_step(images,
                                                                    targets,
                                                                    count)

            self.scheduler.step()

            # print out loss log
            if (e + 1) % self.loss_log_step == 0:
                self.print_loss_log(start_time=start_time,
                                    iters_per_epoch=iters_per_epoch,
                                    e=e,
                                    i=i,
                                    class_loss=class_loss,
                                    loc_loss=loc_loss,
                                    loss=loss)

                self.losses.append([e, class_loss, loc_loss, loss])

            # save model
            if (e + 1) % self.model_save_step == 0:
                self.save_model(e)

            if self.warmup_epoch != 0 and (e + 1) == self.warmup_epoch:
                self.lr *= 10
                write_print(self.output_txt,
                            'Learning rate increased to ' + str(self.lr))
                self.adjust_learning_rate(optimizer=self.optimizer,
                                          gamma=self.sched_gamma,
                                          step=sched)

            num_sched = len(self.learning_sched)
            if num_sched != 0 and sched < num_sched:
                if (e + 1) == self.learning_sched[sched]:

                    self.lr /= 10
                    write_print(self.output_txt,
                                'Learning rate reduced to ' + str(self.lr))
                    sched += 1
                    # self.adjust_learning_rate(optimizer=self.optimizer,
                    #                           gamma=self.sched_gamma,
                    #                           step=sched)

        # print losses
        write_print(self.output_txt, '\n--Losses--')
        for e, class_loss, loc_loss, loss in self.losses:
            loss_string = ' {:.4f} {:.4f} {:.4f}'.format(class_loss,
                                                         loc_loss,
                                                         loss)
            write_print(self.output_txt, str(e) + loss_string)

    def eval(self,
             dataset,
             max_per_image,
             score_threshold):

        num_images = len(dataset)
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(self.class_count)]

        # prepare timers, paths, and files
        timer = {'detection': Timer(), 'nms': Timer()}

        results_path = osp.join(self.model_test_path,
                                self.pretrained_model)
        mkdir(results_path)
        detection_file = osp.join(results_path,
                                  'detections.pkl')

        detect_times = []
        nms_times = []

        with torch.no_grad():

            # for each image
            for i in range(num_images):

                # get image
                image, target, h, w = dataset.pull_item(i)
                image = to_var(image.unsqueeze(0), self.use_gpu)

                # get and time detection
                timer['detection'].tic()
                bboxes, scores = self.model(image)
                detect_time = timer['detection'].toc(average=False)
                detect_times.append(detect_time)

                # convert to CPU tensors
                bboxes = bboxes[0]
                scores = scores[0]
                # bboxes = bboxes.cpu().numpy()
                # scores = scores.cpu().numpy()

                # scale each detection back up to the image
                # scale = torch.Tensor([w, h, w, h]).cpu().numpy()
                scale = torch.Tensor([w, h, w, h])
                bboxes *= scale

                # perform and time NMS
                timer['nms'].tic()

                for j in range(1, self.class_count):

                    # get scores greater than score_threshold
                    selected_i = np.where(scores[:, j] > score_threshold)[0]

                    # if there are scores greater than score_threshold
                    if len(selected_i) > 0:
                        bboxes_i = bboxes[selected_i]
                        scores_i = scores[selected_i, j]
                        detections_i = (bboxes_i, scores_i[:, np.newaxis])
                        detections_i = np.hstack(detections_i)
                        # detections_i = detections_i.astype(np.float32,
                        #                                    copy=False)

                        # keep = nms(detections=detections_i,
                        #            threshold=self.nms_threshold,
                        #            force_cpu=True)
                        # print(type(bboxes_i), type(scores_io: object))
                        keep = nms(boxes=bboxes_i,
                                   scores=scores_i,
                                   iou_threshold=self.nms_threshold)

                        keep = keep[:50]
                        detections_i = detections_i[keep, :]
                        if len(detections_i.shape) == 1:
                            all_boxes[j][i] = np.expand_dims(detections_i, 0)
                        else:
                            all_boxes[j][i] = detections_i

                    elif len(selected_i) == 0:
                        all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)

                # if we need to limit the maximum per image
                if max_per_image > 0:

                    # get all the scores for the image across all classes
                    scores_i = np.hstack([all_boxes[j][i][:, -1]
                                          for j in range(1, self.class_count)])

                    # if the number of detections is greater than max_per_image
                    if len(scores_i) > max_per_image:

                        # get the score of the max_per_image-th image
                        threshold_i = np.sort(scores_i)[-max_per_image]

                        # keep detections with score greater than threshold_i
                        for j in range(1, self.class_count):
                            keep = np.where(all_boxes[j][i][:, -1]
                                            >= threshold_i)[0]
                            all_boxes[j][i] = all_boxes[j][i][keep, :]

                nms_time = timer['nms'].toc(average=False)
                nms_times.append(nms_time)

                temp_string = 'detection: {:d}/{:d} {:.4f}s {:.4f}s'
                temp_string = temp_string.format(i + 1,
                                                 num_images,
                                                 detect_time,
                                                 nms_time)

                write_print(self.output_txt, temp_string)

        with open(detection_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        write_print(self.output_txt, '\nEvaluating detections')

        # perform evaluation
        if self.dataset == 'voc':
            voc_save(all_boxes=all_boxes,
                     dataset=dataset,
                     results_path=results_path,
                     output_txt=self.output_txt)

            aps, mAP = do_voc_eval(results_path=results_path,
                                   dataset=dataset,
                                   output_txt=self.output_txt,
                                   mode='test',
                                   iou_threshold=self.iou_threshold,
                                   use_07_metric=self.use_07_metric)

            write_print(self.output_txt, '\nResults:')
            for ap in aps:
                write_print(self.output_txt, '{:.4f}'.format(ap))
            write_print(self.output_txt, '{:.4f}'.format(np.mean(aps)))

        if self.dataset == 'ccrop':
            ccrop_save(all_boxes=all_boxes,
                     dataset=dataset,
                     results_path=results_path,
                     output_txt=self.output_txt)

            aps, mAP = do_ccrop_eval(results_path=results_path,
                                   dataset=dataset,
                                   output_txt=self.output_txt,
                                   mode='test',
                                   iou_threshold=self.iou_threshold,
                                   use_07_metric=self.use_07_metric)

            write_print(self.output_txt, '\nResults:')
            for ap in aps:
                write_print(self.output_txt, '{:.4f}'.format(ap))
            write_print(self.output_txt, '{:.4f}'.format(np.mean(aps)))
        
        if self.dataset == 'camocrops':
            camocrops_save(all_boxes=all_boxes,
                     dataset=dataset,
                     results_path=results_path,
                     output_txt=self.output_txt)

            aps, mAP = do_camocrops_eval(results_path=results_path,
                                   dataset=dataset,
                                   output_txt=self.output_txt,
                                   mode='test',
                                   iou_threshold=self.iou_threshold,
                                   use_07_metric=self.use_07_metric)

            write_print(self.output_txt, '\nResults:')
            for ap in aps:
                write_print(self.output_txt, '{:.4f}'.format(ap))
            write_print(self.output_txt, '{:.4f}'.format(np.mean(aps)))

        if self.dataset == 'tomatod': # TOMATOD
            detection_list = tomatod_save(all_boxes=all_boxes,
                                       dataset=dataset,
                                       results_path=results_path,
                                       output_txt=self.output_txt)

            detection_list = dataset.pycoco.loadRes(detection_list)
            tomatod_eval = do_coco_eval(dataset.pycoco,
                                     detection_list,
                                     'bbox')
            tomatod_eval.evaluate()
            tomatod_eval.accumulate()
            tomatod_eval.summarize()

            stats = ['AP--IoU=0.50:0.95--all--100',
                     'AP--IoU=0.50--all--100',
                     'AP--IoU=0.75--all-100',
                     'AP--IoU=0.50:0.95--small--100',
                     'AP--IoU=0.50:0.95--medium--100',
                     'AP--IoU=0.50:0.95--large--100',
                     'AR--IoU=0.50:0.95--all--1',
                     'AR--IoU=0.50:0.95--all--10',
                     'AR--IoU=0.50:0.95--all--100',
                     'AR--IoU=0.50:0.95--small--100',
                     'AR--IoU=0.50:0.95--medium--100',
                     'AR--IoU=0.50:0.95--large--100']

            for stat, val in zip(stats, tomatod_eval.stats):
                str_out = '{:s}: {:.3f}'.format(stat, val)
                write_print(self.output_txt, str_out)

            write_print(self.output_txt, '\nResults:')
            for val in tomatod_eval.stats:
                write_print(self.output_txt, '{:.3f}'.format(val))
            
            write_print(self.output_txt, '\n--- Per-Class Evaluation Metrics ---')
            
            category_ids_str = f"Category IDs: {dataset.pycoco.getCatIds()}"
            write_print(self.output_txt, category_ids_str)

            # Retrieve all category IDs
            cat_ids = dataset.pycoco.getCatIds()
            class_names = ["unripe", "semi-ripe", "fully-ripe"]  

            for catId, class_name in zip(cat_ids, class_names):
                per_class_eval = do_coco_eval(dataset.pycoco, detection_list, 'bbox')
                per_class_eval.params.catIds = [catId]  
                
                with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                    per_class_eval.evaluate()
                    per_class_eval.accumulate()
                    per_class_eval.summarize()
                    per_class_output = buf.getvalue()  

                # print(per_class_output)  # print to console
                write_print(self.output_txt, per_class_output)  # save to output file

                # Write specific AP and AR for each class
                ap = per_class_eval.stats[0]  # AP at IoU=0.50:0.95
                ar = per_class_eval.stats[8]  # AR at IoU=0.50:0.95 for maxDets=100
                per_class_str = f"{class_name}: AP={ap:.3f}, AR={ar:.3f}"
                
                write_print(self.output_txt, per_class_str)
                write_print(self.output_txt, '\n')



        if self.dataset == 'coco':
            detection_list = coco_save(all_boxes=all_boxes,
                                       dataset=dataset,
                                       results_path=results_path,
                                       output_txt=self.output_txt)

            detection_list = dataset.pycoco.loadRes(detection_list)
            coco_eval = do_coco_eval(dataset.pycoco,
                                     detection_list,
                                     'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            stats = ['AP--IoU=0.50:0.95--all--100',
                     'AP--IoU=0.50--all--100',
                     'AP--IoU=0.75--all-100',
                     'AP--IoU=0.50:0.95--small--100',
                     'AP--IoU=0.50:0.95--medium--100',
                     'AP--IoU=0.50:0.95--large--100',
                     'AR--IoU=0.50:0.95--all--1',
                     'AR--IoU=0.50:0.95--all--10',
                     'AR--IoU=0.50:0.95--all--100',
                     'AR--IoU=0.50:0.95--small--100',
                     'AR--IoU=0.50:0.95--medium--100',
                     'AR--IoU=0.50:0.95--large--100']

            for stat, val in zip(stats, coco_eval.stats):
                str_out = '{:s}: {:.3f}'.format(stat, val)
                write_print(self.output_txt, str_out)

            write_print(self.output_txt, '\nResults:')
            for val in coco_eval.stats:
                write_print(self.output_txt, '{:.3f}'.format(val))

        detect_times = np.asarray(detect_times)
        nms_times = np.asarray(nms_times)
        total_times = np.add(detect_times, nms_times)
        write_print(self.output_txt, str(1 / np.mean(detect_times[1:])))
        write_print(self.output_txt, str(1 / np.mean(nms_times[1:])))
        write_print(self.output_txt, str(1 / np.mean(total_times[1:])))

    def test(self):
        """
        Testing process.
        """
        self.model.eval()
        self.eval(dataset=self.data_loader.dataset,
                  max_per_image=self.max_per_image,
                  score_threshold=self.score_threshold)


    def print_network(self, model):
        """
        Prints the structure of the network and the total number of parameters + GFLOPs.
        """
        num_params = 0

        # Compute FLOPs with PyTorch profiler (batch_size=1) 12/03/2025 (5:44am)
        # changed back to torchinfo 12/03/2025 (6:47am)
        model_summary = summary(model, input_size=(1, self.input_channels, self.new_size, self.new_size), verbose=0)

        total_flops = (model_summary.total_mult_adds * 2) / 1e9 

        for p in model.parameters():
            num_params += p.numel()
        write_print(self.output_txt, str(model))
        write_print(self.output_txt,
                    'The number of parameters: {}'.format(num_params))
        write_print(self.output_txt,
                    'The number of FLOPs: {:.3f} GFLOPs'.format(total_flops))

    # def load_pretrained_model(self,
    #                           model,
    #                           model_save_path,
    #                           pretrained_model):
    #     """
    #     loads a pre-trained model from a .pth file
    #     """
    #     model.load_state_dict(torch.load(os.path.join(
    #         model_save_path, '{}.pth'.format(pretrained_model))))
    #     write_print(self.output_txt,
    #                 'loaded trained model {}'.format(pretrained_model))

    def adjust_learning_rate(self,
                             optimizer,
                             gamma,
                             step):
        """Sets the learning rate to the initial LR decayed by 10 at every
            specified step.
        """
        lr = self.lr * (gamma ** (step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def print_loss_log(self,
                       start_time,
                       iters_per_epoch,
                       e,
                       i,
                       class_loss,
                       loc_loss,
                       loss):
        """
        Prints the loss and elapsed time for each epoch.
        """
        total_iter = self.num_epochs * iters_per_epoch
        cur_iter = e * iters_per_epoch + i

        elapsed = time.time() - start_time
        total_time = (total_iter - cur_iter) * elapsed / (cur_iter + 1)
        epoch_time = (iters_per_epoch - i) * elapsed / (cur_iter + 1)

        epoch_time = str(datetime.timedelta(seconds=epoch_time))
        total_time = str(datetime.timedelta(seconds=total_time))
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed {}/{} -- {}, Epoch [{}/{}], Iter [{}/{}], " \
              "class_loss: {:.4f}, loc_loss: {:.4f}, " \
              "loss: {:.4f}".format(elapsed,
                                    epoch_time,
                                    total_time,
                                    e + 1,
                                    self.num_epochs,
                                    i + 1,
                                    iters_per_epoch,
                                    class_loss.item(),
                                    loc_loss.item(),
                                    loss.item())

        write_print(self.output_txt, log)

    def save_model(self, e):
        """
        Saves a model per epoch.
        """
        path = os.path.join(
            self.model_save_path,
            '{}/{}.pth'.format(self.version, e + 1)
        )

        torch.save(self.model.state_dict(), path)

    # def model_step(self,
    #                images,
    #                targets,
    #                count):
        """
        A step for each iteration
        """

        """
        if count == 0:
            # update parameters
            # self.optimizer.step()
            self.scaler.step(self.optimizer)

            # empty the gradients of the model through the optimizer
            self.optimizer.zero_grad()

            self.scaler.update()

            count = self.batch_multiplier

        with autocast():

            # forward pass
            class_preds, loc_preds = self.model(images)

            # compute loss
            class_targets = [target[:, -1] for target in targets]
            loc_targets = [target[:, :-1] for target in targets]
            losses = self.criterion(class_preds=class_preds,
                                    class_targets=class_targets,
                                    loc_preds=loc_preds,
                                    loc_targets=loc_targets,
                                    anchors=self.anchor_boxes)

            class_loss, loc_loss, loss = losses

        # compute gradients using back propagation
        # loss.backward()
        class_loss = class_loss / self.batch_multiplier
        loc_loss = loc_loss / self.batch_multiplier
        loss = loss / self.batch_multiplier
        self.scaler.scale(loss).backward()
        count = count - 1

        # return loss
        return class_loss, loc_loss, loss, count
    """

    def model_step(self,
                   images,
                   targets,
                   count):
        """
        Signifies a step for each iteration.
        """

        # empty the gradients of the model through the optimizer
        self.optimizer.zero_grad()

        # with autocast():

        # forward pass
        class_preds, loc_preds = self.model(images)

        # compute loss
        class_targets = [target[:, -1] for target in targets]
        loc_targets = [target[:, :-1] for target in targets]
        losses = self.criterion(class_preds=class_preds,
                                class_targets=class_targets,
                                loc_preds=loc_preds,
                                loc_targets=loc_targets,
                                anchors=self.anchor_boxes)

        class_loss, loc_loss, loss = losses

        # compute gradients using back propagation
        loss.backward()
        # self.scaler.scale(loss).backward()

        # update parameters
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        self.optimizer.step()
        # self.scaler.step(self.optimizer)

        # self.scaler.update()

        # return loss
        return class_loss, loc_loss, loss, count

    def train(self):
        """
        Training process for SSD.
        """

        # set model in training mode
        self.model.train()

        self.losses = []
        count = self.batch_multiplier

        iters_per_epoch = len(self.data_loader)

        # start with a trained model if exists
        if self.pretrained_model:
            start = 0 #int(self.pretrained_model.split('/')[-1])
        else:
            start = 0

        sched = 0

        if self.warmup_epoch != 0:
            self.lr /= 10
            write_print(self.output_txt,
                        'Learning rate reduced to ' + str(self.lr))
            self.adjust_learning_rate(optimizer=self.optimizer,
                                      gamma=self.sched_gamma,
                                      step=sched)

        # start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            for i, (images, targets) in enumerate(tqdm(self.data_loader)):
                images = to_var(images, self.use_gpu)
                targets = [to_var(target, self.use_gpu) for target in targets]

                class_loss, loc_loss, loss, count = self.model_step(images,
                                                                    targets,
                                                                    count)

            self.scheduler.step()

            # print out loss log
            if (e + 1) % self.loss_log_step == 0:
                self.print_loss_log(start_time=start_time,
                                    iters_per_epoch=iters_per_epoch,
                                    e=e,
                                    i=i,
                                    class_loss=class_loss,
                                    loc_loss=loc_loss,
                                    loss=loss)

                self.losses.append([e, class_loss, loc_loss, loss])

            # save model
            if (e + 1) % self.model_save_step == 0:
                self.save_model(e)

            if self.warmup_epoch != 0 and (e + 1) == self.warmup_epoch:
                self.lr *= 10
                write_print(self.output_txt,
                            'Learning rate increased to ' + str(self.lr))
                self.adjust_learning_rate(optimizer=self.optimizer,
                                          gamma=self.sched_gamma,
                                          step=sched)

            num_sched = len(self.learning_sched)
            if num_sched != 0 and sched < num_sched:
                if (e + 1) == self.learning_sched[sched]:

                    self.lr /= 10
                    write_print(self.output_txt,
                                'Learning rate reduced to ' + str(self.lr))
                    sched += 1
                    # self.adjust_learning_rate(optimizer=self.optimizer,
                    #                           gamma=self.sched_gamma,
                    #                           step=sched)

        # print losses
        write_print(self.output_txt, '\n--Losses--')
        for e, class_loss, loc_loss, loss in self.losses:
            loss_string = ' {:.4f} {:.4f} {:.4f}'.format(class_loss,
                                                         loc_loss,
                                                         loss)
            write_print(self.output_txt, str(e) + loss_string)

    def eval(self,
             dataset,
             max_per_image,
             score_threshold):

        num_images = len(dataset)
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(self.class_count)]

        # prepare timers, paths, and files
        timer = {'detection': Timer(), 'nms': Timer()}

        results_path = osp.join(self.model_test_path,
                                self.pretrained_model)
        mkdir(results_path)
        detection_file = osp.join(results_path,
                                  'detections.pkl')

        detect_times = []
        nms_times = []

        with torch.no_grad():

            # for each image
            for i in range(num_images):

                # get image
                image, target, h, w = dataset.pull_item(i)
                image = to_var(image.unsqueeze(0), self.use_gpu)

                # get and time detection
                timer['detection'].tic()
                bboxes, scores = self.model(image)
                detect_time = timer['detection'].toc(average=False)
                detect_times.append(detect_time)

                # convert to CPU tensors
                bboxes = bboxes[0]
                scores = scores[0]
                # bboxes = bboxes.cpu().numpy()
                # scores = scores.cpu().numpy()

                # scale each detection back up to the image
                # scale = torch.Tensor([w, h, w, h]).cpu().numpy()
                scale = torch.Tensor([w, h, w, h])
                bboxes *= scale

                # perform and time NMS
                timer['nms'].tic()

                for j in range(1, self.class_count):

                    # get scores greater than score_threshold
                    selected_i = np.where(scores[:, j] > score_threshold)[0]

                    # if there are scores greater than score_threshold
                    if len(selected_i) > 0:
                        bboxes_i = bboxes[selected_i]
                        scores_i = scores[selected_i, j]
                        detections_i = (bboxes_i, scores_i[:, np.newaxis])
                        detections_i = np.hstack(detections_i)
                        # detections_i = detections_i.astype(np.float32,
                        #                                    copy=False)

                        # keep = nms(detections=detections_i,
                        #            threshold=self.nms_threshold,
                        #            force_cpu=True)
                        # print(type(bboxes_i), type(scores_io: object))
                        keep = nms(boxes=bboxes_i,
                                   scores=scores_i,
                                   iou_threshold=self.nms_threshold)

                        keep = keep[:50]
                        detections_i = detections_i[keep, :]
                        if len(detections_i.shape) == 1:
                            all_boxes[j][i] = np.expand_dims(detections_i, 0)
                        else:
                            all_boxes[j][i] = detections_i

                    elif len(selected_i) == 0:
                        all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)

                # if we need to limit the maximum per image
                if max_per_image > 0:

                    # get all the scores for the image across all classes
                    scores_i = np.hstack([all_boxes[j][i][:, -1]
                                          for j in range(1, self.class_count)])

                    # if the number of detections is greater than max_per_image
                    if len(scores_i) > max_per_image:

                        # get the score of the max_per_image-th image
                        threshold_i = np.sort(scores_i)[-max_per_image]

                        # keep detections with score greater than threshold_i
                        for j in range(1, self.class_count):
                            keep = np.where(all_boxes[j][i][:, -1]
                                            >= threshold_i)[0]
                            all_boxes[j][i] = all_boxes[j][i][keep, :]

                nms_time = timer['nms'].toc(average=False)
                nms_times.append(nms_time)

                temp_string = 'detection: {:d}/{:d} {:.4f}s {:.4f}s'
                temp_string = temp_string.format(i + 1,
                                                 num_images,
                                                 detect_time,
                                                 nms_time)

                write_print(self.output_txt, temp_string)

        with open(detection_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        write_print(self.output_txt, '\nEvaluating detections')

        # perform evaluation
        if self.dataset == 'voc':
            voc_save(all_boxes=all_boxes,
                     dataset=dataset,
                     results_path=results_path,
                     output_txt=self.output_txt)

            aps, mAP = do_voc_eval(results_path=results_path,
                                   dataset=dataset,
                                   output_txt=self.output_txt,
                                   mode='test',
                                   iou_threshold=self.iou_threshold,
                                   use_07_metric=self.use_07_metric)

            write_print(self.output_txt, '\nResults:')
            for ap in aps:
                write_print(self.output_txt, '{:.4f}'.format(ap))
            write_print(self.output_txt, '{:.4f}'.format(np.mean(aps)))

        if self.dataset == 'ccrop':
            ccrop_save(all_boxes=all_boxes,
                     dataset=dataset,
                     results_path=results_path,
                     output_txt=self.output_txt)

            aps, mAP = do_ccrop_eval(results_path=results_path,
                                   dataset=dataset,
                                   output_txt=self.output_txt,
                                   mode='test',
                                   iou_threshold=self.iou_threshold,
                                   use_07_metric=self.use_07_metric)

            write_print(self.output_txt, '\nResults:')
            for ap in aps:
                write_print(self.output_txt, '{:.4f}'.format(ap))
            write_print(self.output_txt, '{:.4f}'.format(np.mean(aps)))
        
        if self.dataset == 'camocrops':
            camocrops_save(all_boxes=all_boxes,
                     dataset=dataset,
                     results_path=results_path,
                     output_txt=self.output_txt)

            aps, mAP = do_camocrops_eval(results_path=results_path,
                                   dataset=dataset,
                                   output_txt=self.output_txt,
                                   mode='test',
                                   iou_threshold=self.iou_threshold,
                                   use_07_metric=self.use_07_metric)

            write_print(self.output_txt, '\nResults:')
            for ap in aps:
                write_print(self.output_txt, '{:.4f}'.format(ap))
            write_print(self.output_txt, '{:.4f}'.format(np.mean(aps)))

        if self.dataset == 'tomatod': # TOMATOD
            detection_list = tomatod_save(all_boxes=all_boxes,
                                       dataset=dataset,
                                       results_path=results_path,
                                       output_txt=self.output_txt)

            detection_list = dataset.pycoco.loadRes(detection_list)
            tomatod_eval = do_coco_eval(dataset.pycoco,
                                     detection_list,
                                     'bbox')
            tomatod_eval.evaluate()
            tomatod_eval.accumulate()
            tomatod_eval.summarize()

            stats = ['AP--IoU=0.50:0.95--all--100',
                     'AP--IoU=0.50--all--100',
                     'AP--IoU=0.75--all-100',
                     'AP--IoU=0.50:0.95--small--100',
                     'AP--IoU=0.50:0.95--medium--100',
                     'AP--IoU=0.50:0.95--large--100',
                     'AR--IoU=0.50:0.95--all--1',
                     'AR--IoU=0.50:0.95--all--10',
                     'AR--IoU=0.50:0.95--all--100',
                     'AR--IoU=0.50:0.95--small--100',
                     'AR--IoU=0.50:0.95--medium--100',
                     'AR--IoU=0.50:0.95--large--100']

            for stat, val in zip(stats, tomatod_eval.stats):
                str_out = '{:s}: {:.3f}'.format(stat, val)
                write_print(self.output_txt, str_out)

            write_print(self.output_txt, '\nResults:')
            for val in tomatod_eval.stats:
                write_print(self.output_txt, '{:.3f}'.format(val))
            
            write_print(self.output_txt, '\n--- Per-Class Evaluation Metrics ---')
            
            category_ids_str = f"Category IDs: {dataset.pycoco.getCatIds()}"
            write_print(self.output_txt, category_ids_str)

            # Retrieve all category IDs
            cat_ids = dataset.pycoco.getCatIds()
            class_names = ["unripe", "semi-ripe", "fully-ripe"]  # Adjust based on your dataset

            for catId, class_name in zip(cat_ids, class_names):
                per_class_eval = do_coco_eval(dataset.pycoco, detection_list, 'bbox')
                per_class_eval.params.catIds = [catId]  # Set category to evaluate
                
                # Capture printed output
                with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                    per_class_eval.evaluate()
                    per_class_eval.accumulate()
                    per_class_eval.summarize()
                    per_class_output = buf.getvalue()  # Store output

                # print(per_class_output)  # Print to console
                write_print(self.output_txt, per_class_output)  # Save to output file

                # Write specific AP and AR for each class
                ap = per_class_eval.stats[0]  # AP at IoU=0.50:0.95
                ar = per_class_eval.stats[8]  # AR at IoU=0.50:0.95 for maxDets=100
                per_class_str = f"{class_name}: AP={ap:.3f}, AR={ar:.3f}"
                
                write_print(self.output_txt, per_class_str)
                write_print(self.output_txt, '\n')



        if self.dataset == 'coco':
            detection_list = coco_save(all_boxes=all_boxes,
                                       dataset=dataset,
                                       results_path=results_path,
                                       output_txt=self.output_txt)

            detection_list = dataset.pycoco.loadRes(detection_list)
            coco_eval = do_coco_eval(dataset.pycoco,
                                     detection_list,
                                     'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            stats = ['AP--IoU=0.50:0.95--all--100',
                     'AP--IoU=0.50--all--100',
                     'AP--IoU=0.75--all-100',
                     'AP--IoU=0.50:0.95--small--100',
                     'AP--IoU=0.50:0.95--medium--100',
                     'AP--IoU=0.50:0.95--large--100',
                     'AR--IoU=0.50:0.95--all--1',
                     'AR--IoU=0.50:0.95--all--10',
                     'AR--IoU=0.50:0.95--all--100',
                     'AR--IoU=0.50:0.95--small--100',
                     'AR--IoU=0.50:0.95--medium--100',
                     'AR--IoU=0.50:0.95--large--100']

            for stat, val in zip(stats, coco_eval.stats):
                str_out = '{:s}: {:.3f}'.format(stat, val)
                write_print(self.output_txt, str_out)

            write_print(self.output_txt, '\nResults:')
            for val in coco_eval.stats:
                write_print(self.output_txt, '{:.3f}'.format(val))

        detect_times = np.asarray(detect_times)
        nms_times = np.asarray(nms_times)
        total_times = np.add(detect_times, nms_times)
        write_print(self.output_txt, str(1 / np.mean(detect_times[1:])))
        write_print(self.output_txt, str(1 / np.mean(nms_times[1:])))
        write_print(self.output_txt, str(1 / np.mean(total_times[1:])))

    def test(self):
        """
        Testing process for SSD.
        """
        self.model.eval()
        self.eval(dataset=self.data_loader.dataset,
                  max_per_image=self.max_per_image,
                  score_threshold=self.score_threshold)

"""
YOLOv8 Knowledge Distillation Trainer
------------------------------------
Custom trainer implementing feature-based and logit distillation for YOLO models.

Functions

- **YOLOv8DistillationTrainer:** Extended DetectionTrainer with teacher-student learning.
- **_compute_distillation_loss:** Hybrid loss combining feature alignment and KL-divergence.
- **_extract_outputs:** Extract logits and features from model predictions.
----
"""

import torch
import torch.nn.functional as F
import pandas as pd
import time
import warnings
import math
import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK, TQDM, colorstr
from ultralytics.utils.torch_utils import autocast
from ultralytics.cfg import get_cfg
from pathlib import Path
from torch import distributed as dist

try:
    from ultralytics.utils.torch_utils import unset_deterministic
except ImportError:
    unset_deterministic = lambda: None  # fallback if not available



class YOLOv8DistillationTrainer(DetectionTrainer):
    """Extended Detection Trainer with Knowledge Distillation.
    
    Implements teacher-student training with feature alignment and logit distillation.
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if cfg is None:
            cfg = DEFAULT_CFG

        if overrides is None:
            overrides = {}

        self.distill_weight = overrides.pop('distill_weight', 0.5)
        self.temperature = overrides.pop('temperature', 4.0)
        self.teacher_weights_path = overrides.pop('teacher_weights', None)
        self.feature_loss_weight = overrides.pop('feature_loss_weight', 0.5)
        
        # New scaling factors to balance the two distillation loss components
        self.feature_loss_scale = overrides.pop('feature_loss_scale', 1.0)
        self.logit_loss_scale = overrides.pop('logit_loss_scale', 1.0)

        self.debug_distillation = True
        self.distill_step_count = 0

        self.loss_items = None
        self.loss = None

        # Cleaned up custom_distill_args dictionary
        self.custom_distill_args = {
            'distill_weight': self.distill_weight,
            'temperature': self.temperature,
            'teacher_weights': self.teacher_weights_path,
            'feature_loss_weight': self.feature_loss_weight,
            'feature_loss_scale': self.feature_loss_scale,
            'logit_loss_scale': self.logit_loss_scale
        }

        super().__init__(cfg, overrides, _callbacks)

        self.teacher_model = None
        self._setup_teacher_model()

    def _setup_teacher_model(self):
        if self.teacher_weights_path is None:
            LOGGER.warning("No teacher_weights specified. Training without knowledge distillation.")
            return

        if not Path(self.teacher_weights_path).exists():
            LOGGER.error(f"Teacher weights not found: {self.teacher_weights_path}")
            LOGGER.warning("Training without knowledge distillation.")
            return

        LOGGER.info(f"Loading teacher model from {self.teacher_weights_path}")
        try:
            self.teacher_model = YOLO(self.teacher_weights_path)
            self.teacher_model.model.eval()

            for param in self.teacher_model.model.parameters():
                param.requires_grad = False

            LOGGER.info("Teacher model loaded and frozen successfully")
            LOGGER.info(f"Distillation weight: {self.distill_weight}")
            LOGGER.info(f"Temperature: {self.temperature}")
            LOGGER.info(f"Feature Loss Weight: {self.feature_loss_weight}")
            LOGGER.info(f"Feature Loss Scale: {self.feature_loss_scale}")
            LOGGER.info(f"Logit Loss Scale: {self.logit_loss_scale}")
            LOGGER.info(f"Debug mode: {self.debug_distillation}")
        except Exception as e:
            LOGGER.error(f"Failed to load teacher model: {e}")
            LOGGER.warning("Training without knowledge distillation.")
            self.teacher_model = None


    def _setup_train(self, world_size):
        super()._setup_train(world_size)

        if self.teacher_model is not None:
            self.teacher_model.model.to(self.device)
            LOGGER.info(f"Teacher model moved to device: {self.device}")

    def _model_train(self):
        self.model.train()

    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")

        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )

        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        epoch = self.start_epoch
        self.optimizer.zero_grad()

        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)

            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)

            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)

                    if self.model.training:
                        model_output = self.model(batch)

                        if isinstance(model_output, tuple) and len(model_output) == 2:
                            loss, loss_items = model_output
                            preds = None
                        else:
                            preds = model_output
                            loss, loss_items = self.model.loss(preds, batch)

                        self.loss_items = loss_items
                        self.loss = loss

                        with torch.no_grad():
                            self.model.eval()
                            student_preds = self.model(batch['img'])
                            self.model.train()

                        if self.teacher_model is not None:
                            try:
                                with torch.no_grad():
                                    teacher_preds = self.teacher_model.model(batch['img'])

                                standard_loss_val = loss.item()  # Store the original standard loss for logging
                                distill_loss = self._compute_distillation_loss(student_preds, teacher_preds)
                                total_loss = (1 - self.distill_weight) * loss + self.distill_weight * distill_loss

                                if isinstance(loss_items, torch.Tensor):
                                    distill_loss_item = distill_loss.detach()
                                    loss_items = torch.cat([loss_items, distill_loss_item.unsqueeze(0)])
                                
                                loss = total_loss # Now update the loss for backpropagation
                                self.loss = loss

                                if self.debug_distillation and self.distill_step_count < 5:
                                    # Log the original standard loss and the new combined loss
                                    LOGGER.info(f"KD Step {self.distill_step_count}: Standard={standard_loss_val:.4f}, Distill={distill_loss.item():.4f}, Combined={total_loss.item():.4f}")
                                
                                self.distill_step_count += 1

                            except Exception as e:
                                LOGGER.warning(f"Distillation error: {e}")
                                if self.debug_distillation:
                                    import traceback
                                    LOGGER.warning(f"Full traceback: {traceback.format_exc()}")
                    else:
                        preds = self.model(batch)
                        loss = torch.tensor(0.0, device=self.device)
                        loss_items = torch.zeros(3, device=self.device)
                        self.loss_items = loss_items
                        self.loss = loss

                    if RANK != -1:
                        loss *= world_size
                    self.tloss = (
                        (self.tloss * i + loss_items) / (i + 1) if self.tloss is not None else loss_items
                    )

                self.scaler.scale(loss).backward()

                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break

                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")

            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            self.run_callbacks("on_fit_epoch_end")
            if self._get_memory() > 0.5 * torch.cuda.get_device_properties(self.device).total_memory / 1e9:
                self._clear_memory()

            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
            epoch += 1

        if RANK in {-1, 0}:
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def _compute_distillation_loss(self, student_preds, teacher_preds):
        """Compute Hybrid Distillation Loss.
        
        Combines feature alignment and KL-divergence with scaling factors.

        Args:
            student_preds: Student model predictions (logits, features).
            teacher_preds: Teacher model predictions (logits, features).

        Returns:
            torch.Tensor: Combined distillation loss.
        """
        student_outputs = self._extract_outputs(student_preds, "student")
        teacher_outputs = self._extract_outputs(teacher_preds, "teacher")

        if student_outputs is None or teacher_outputs is None:
            return torch.tensor(0.0, device=self.device)
        
        # 1. Calculate the raw losses
        logit_loss_raw = self._kl_divergence_loss(student_outputs['logits'], teacher_outputs['logits'])
        feature_loss_raw = self._feature_alignment_loss(student_outputs['features'], teacher_outputs['features'])

        # 2. Apply scaling factors
        logit_loss = logit_loss_raw * self.logit_loss_scale
        feature_loss = feature_loss_raw * self.feature_loss_scale
        
        # 3. Combine the two scaled distillation losses using the weight
        total_distill_loss = (self.feature_loss_weight * feature_loss) + \
                             ((1 - self.feature_loss_weight) * logit_loss)

        if self.debug_distillation and self.distill_step_count < 5:
            # Updated log to show scaled losses
            LOGGER.info(f"DEBUG Distill: Scaled Feature Loss={feature_loss.item():.4f}, Scaled Logit Loss={logit_loss.item():.4f}, Combined Distill Loss={total_distill_loss.item():.4f}")

        return total_distill_loss

    def _extract_outputs(self, preds, model_type):
        """
        Extracts both logits (for KL-divergence) and features (for alignment)
        from the model's output.
        """
        # Based on logs, YOLO output is a tuple: (prediction_tensor, feature_map_list)
        if isinstance(preds, (list, tuple)) and len(preds) == 2:
            logits = preds[0]
            features = preds[1]

            # Basic validation of the expected structure
            if torch.is_tensor(logits) and isinstance(features, list) and torch.is_tensor(features[0]):
                if self.debug_distillation and self.distill_step_count < 5:
                    LOGGER.info(f"DEBUG Extract: Successfully extracted logits {logits.shape} and {len(features)} feature maps for {model_type}.")
                return {'logits': logits, 'features': features}

        LOGGER.warning(f"DEBUG Extract: Failed to extract outputs for {model_type}. Check model output structure.")
        return None

    def _kl_divergence_loss(self, student_logits, teacher_logits):
        try:
            if student_logits.device != teacher_logits.device:
                teacher_logits = teacher_logits.to(student_logits.device)

            if student_logits.shape != teacher_logits.shape:
                LOGGER.warning(f"Shape mismatch: student {student_logits.shape} vs teacher {teacher_logits.shape}")
                return torch.tensor(0.0, device=student_logits.device)

            original_shape = student_logits.shape
            if len(student_logits.shape) > 2:
                student_logits = student_logits.view(-1, student_logits.shape[-1])
                teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])

            student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

            kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

            return kl_loss * (self.temperature ** 2)

        except Exception as e:
            LOGGER.warning(f"Error in KL divergence computation: {e}")
            return torch.tensor(0.0, device=student_logits.device if torch.is_tensor(student_logits) else self.device)


    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/box_loss", f"{prefix}/cls_loss", f"{prefix}/dfl_loss"]
        if self.teacher_model is not None:
            keys.append(f"{prefix}/distill_loss")

        if loss_items is not None:
            if not torch.is_tensor(loss_items):
                loss_items = torch.tensor(loss_items, device=self.device)
            loss_items = loss_items[:len(keys)]
            return dict(zip(keys, loss_items))
        else:
            return keys

    def _feature_alignment_loss(self, student_features_list, teacher_features_list):
        """
        Compute alignment loss between lists of student and teacher features.
        """
        total_alignment_loss = torch.tensor(0.0, device=self.device)
        if len(student_features_list) != len(teacher_features_list):
            LOGGER.warning("Feature map lists have different lengths, skipping feature alignment.")
            return total_alignment_loss

        for s_feat, t_feat in zip(student_features_list, teacher_features_list):
            # Basic MSE loss on flattened and normalized features
            s_feat_flat = s_feat.view(s_feat.shape[0], -1)
            t_feat_flat = t_feat.view(t_feat.shape[0], -1)

            s_norm = F.normalize(s_feat_flat, p=2, dim=1)
            t_norm = F.normalize(t_feat_flat, p=2, dim=1)

            loss = F.mse_loss(s_norm, t_norm)
            total_alignment_loss += loss

        return total_alignment_loss

    def read_results_csv(self):
        try:
            return pd.read_csv(self.csv, on_bad_lines='skip').to_dict(orient="list")
        except Exception as e:
            LOGGER.warning(f"[read_results_csv] Failed to parse results CSV: {e}")
            return {}
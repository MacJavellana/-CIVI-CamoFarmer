import torch
import torch.nn.functional as F
import time
import warnings
import math
import numpy as np
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.model import RTDETR
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK, TQDM, colorstr
from ultralytics.utils.torch_utils import autocast, unset_deterministic
from ultralytics.cfg import get_cfg
from pathlib import Path
from torch import distributed as dist

class RTDETRDistillationTrainer(RTDETRTrainer):
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if cfg is None:
            cfg = DEFAULT_CFG
        
        if overrides is None:
            overrides = {}
        
        self.distill_weight = overrides.pop('distill_weight', 0.5)
        self.temperature = overrides.pop('temperature', 4.0)
        self.teacher_weights_path = overrides.pop('teacher_weights', None)
        
        self.debug_distillation = True
        self.distill_step_count = 0

        self.loss_items = None
        self.loss = None  # Add this line

        self.custom_distill_args = {
            'distill_weight': self.distill_weight,
            'temperature': self.temperature,
            'teacher_weights': self.teacher_weights_path
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
            self.teacher_model = RTDETR(self.teacher_weights_path)
            self.teacher_model.model.eval()
            
            for param in self.teacher_model.model.parameters():
                param.requires_grad = False
            
            LOGGER.info("Teacher model loaded and frozen successfully")
            LOGGER.info(f"Distillation weight: {self.distill_weight}")
            LOGGER.info(f"Temperature: {self.temperature}")
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
                                
                                distill_loss = self._compute_distillation_loss(student_preds, teacher_preds)
                                
                                total_loss = (1 - self.distill_weight) * loss + self.distill_weight * distill_loss
                                
                                if isinstance(loss_items, torch.Tensor):
                                    distill_loss_item = distill_loss.detach()
                                    loss_items = torch.cat([loss_items, distill_loss_item.unsqueeze(0)])
                                
                                loss = total_loss
                                self.loss = loss
                                if self.debug_distillation and self.distill_step_count < 5:
                                    LOGGER.info(f"🔥 KD Step {self.distill_step_count}: Standard={loss.item():.4f}, Distill={distill_loss.item():.4f}, Combined={total_loss.item():.4f}")
                                
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
            if self._get_memory(fraction=True) > 0.5:
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
        """
        Compute feature-level knowledge distillation loss.
        Since student and teacher have different architectures, we'll use feature alignment.
        """
        distill_loss = torch.tensor(0.0, device=self.device)
        
        try:
            if self.debug_distillation and self.distill_step_count < 5:
                LOGGER.info(f"DEBUG Distill: Student type = {type(student_preds)}, Teacher type = {type(teacher_preds)}")
            
            # Extract meaningful features for distillation
            student_features = self._extract_features(student_preds, "student")
            teacher_features = self._extract_features(teacher_preds, "teacher")
            
            if student_features is not None and teacher_features is not None:
                # Compute feature alignment loss
                distill_loss = self._feature_alignment_loss(student_features, teacher_features)
                
                if self.debug_distillation and self.distill_step_count < 5:
                    LOGGER.info(f"DEBUG Distill: Feature alignment loss = {distill_loss.item():.4f}")
            else:
                if self.debug_distillation and self.distill_step_count < 5:
                    LOGGER.warning("DEBUG Distill: Could not extract comparable features")
        
        except Exception as e:
            LOGGER.warning(f"Error in distillation loss computation: {e}")
            if self.debug_distillation:
                import traceback
                LOGGER.warning(f"Full traceback: {traceback.format_exc()}")
            distill_loss = torch.tensor(0.0, device=self.device)
        
        if self.debug_distillation and self.distill_step_count < 5:
            LOGGER.info(f"DEBUG Distill: Final distillation loss = {distill_loss.item():.4f}")
        
        return distill_loss

    def _extract_features(self, preds, model_type):
        """Extract comparable features from model predictions."""
        try:
            if self.debug_distillation and self.distill_step_count < 5:
                LOGGER.info(f"DEBUG Extract: {model_type} preds type = {type(preds)}")
                if isinstance(preds, (list, tuple)):
                    LOGGER.info(f"DEBUG Extract: {model_type} preds length = {len(preds)}")
                    for i, item in enumerate(preds):
                        LOGGER.info(f"DEBUG Extract: {model_type} item {i} type = {type(item)}")
                        if torch.is_tensor(item):
                            LOGGER.info(f"DEBUG Extract: {model_type} item {i} shape = {item.shape}")
            
            if isinstance(preds, (list, tuple)) and len(preds) > 0:
                # Strategy 1: Look for main prediction tensor with query dimension
                for i, pred in enumerate(preds):
                    if torch.is_tensor(pred):
                        if len(pred.shape) == 3:  # [batch, queries, features]
                            if model_type == "student" and pred.shape[1] >= 100:  # Student has ~300 queries
                                if self.debug_distillation and self.distill_step_count < 5:
                                    LOGGER.info(f"DEBUG Extract: Found {model_type} main tensor at index {i}, shape = {pred.shape}")
                                return pred
                            elif model_type == "teacher" and pred.shape[1] >= 3:  # Teacher might have fewer queries
                                if self.debug_distillation and self.distill_step_count < 5:
                                    LOGGER.info(f"DEBUG Extract: Found {model_type} main tensor at index {i}, shape = {pred.shape}")
                                return pred
                    
                    # Strategy 2: Handle nested structures recursively
                    elif isinstance(pred, (list, tuple)):
                        if self.debug_distillation and self.distill_step_count < 5:
                            LOGGER.info(f"DEBUG Extract: Exploring nested structure at index {i}")
                        nested_result = self._extract_features(pred, model_type)
                        if nested_result is not None:
                            return nested_result
                
                # Strategy 3: If no 3D tensor found, look for any suitable tensor
                for i, pred in enumerate(preds):
                    if torch.is_tensor(pred):
                        # For teacher, accept any tensor with reasonable size
                        if model_type == "teacher" and len(pred.shape) >= 2 and pred.numel() > 100:
                            if self.debug_distillation and self.distill_step_count < 5:
                                LOGGER.info(f"DEBUG Extract: Using fallback {model_type} tensor at index {i}, shape = {pred.shape}")
                            
                            # Reshape to [batch, queries, features] format if needed
                            if len(pred.shape) == 2:
                                # Assume [batch*queries, features] -> [batch, queries, features]
                                batch_size = 4  # Your batch size
                                queries = pred.shape[0] // batch_size
                                features = pred.shape[1]
                                reshaped = pred.view(batch_size, queries, features)
                                if self.debug_distillation and self.distill_step_count < 5:
                                    LOGGER.info(f"DEBUG Extract: Reshaped {model_type} from {pred.shape} to {reshaped.shape}")
                                return reshaped
                            elif len(pred.shape) == 4:
                                # Flatten spatial dimensions: [batch, channels, h, w] -> [batch, h*w, channels]
                                b, c, h, w = pred.shape
                                flattened = pred.permute(0, 2, 3, 1).contiguous().view(b, h*w, c)
                                if self.debug_distillation and self.distill_step_count < 5:
                                    LOGGER.info(f"DEBUG Extract: Flattened {model_type} from {pred.shape} to {flattened.shape}")
                                return flattened
                            else:
                                return pred
            
            if self.debug_distillation and self.distill_step_count < 5:
                LOGGER.warning(f"DEBUG Extract: No suitable tensor found for {model_type}")
            return None
            
        except Exception as e:
            if self.debug_distillation and self.distill_step_count < 5:
                LOGGER.warning(f"DEBUG Extract: Error extracting {model_type} features: {e}")
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
    def _feature_alignment_loss(self, student_features, teacher_features):
        """
        Compute alignment loss between student and teacher features.
        Uses adaptive pooling to handle different dimensions.
        """
        try:
            if self.debug_distillation and self.distill_step_count < 5:
                LOGGER.info(f"DEBUG Align: Input shapes - Student: {student_features.shape}, Teacher: {teacher_features.shape}")
            
            # Ensure both are 3D tensors [batch, queries, features]
            if len(student_features.shape) != 3 or len(teacher_features.shape) != 3:
                if self.debug_distillation and self.distill_step_count < 5:
                    LOGGER.warning(f"DEBUG Align: Invalid tensor dimensions")
                return torch.tensor(0.0, device=self.device)
            
            s_batch, s_queries, s_dim = student_features.shape
            t_batch, t_queries, t_dim = teacher_features.shape
            
            # Ensure same batch size
            min_batch = min(s_batch, t_batch)
            student_features = student_features[:min_batch]
            teacher_features = teacher_features[:min_batch]
            
            # Align query dimensions using adaptive pooling
            target_queries = min(s_queries, t_queries, 100)  # Cap at 100 for efficiency
            
            # Transpose for pooling: [batch, features, queries]
            student_pooled = student_features.transpose(1, 2)
            teacher_pooled = teacher_features.transpose(1, 2)
            
            # Pool queries dimension
            if s_queries != target_queries:
                student_pooled = F.adaptive_avg_pool1d(student_pooled, target_queries)
            if t_queries != target_queries:
                teacher_pooled = F.adaptive_avg_pool1d(teacher_pooled, target_queries)
            
            # Transpose back: [batch, queries, features]
            student_pooled = student_pooled.transpose(1, 2)
            teacher_pooled = teacher_pooled.transpose(1, 2)
            
            # Align feature dimensions
            target_dim = min(s_dim, t_dim, 64)  # Cap at 64 for efficiency
            student_projected = student_pooled[..., :target_dim]
            teacher_projected = teacher_pooled[..., :target_dim]
            
            if self.debug_distillation and self.distill_step_count < 5:
                LOGGER.info(f"DEBUG Align: Final aligned shapes - Student: {student_projected.shape}, Teacher: {teacher_projected.shape}")
            
            # Normalize features before computing loss
            student_norm = F.normalize(student_projected, p=2, dim=-1)
            teacher_norm = F.normalize(teacher_projected, p=2, dim=-1)
            
            # Compute MSE loss between normalized features
            alignment_loss = F.mse_loss(student_norm, teacher_norm)
            
            if self.debug_distillation and self.distill_step_count < 5:
                LOGGER.info(f"DEBUG Align: Computed alignment loss = {alignment_loss.item():.4f}")
            
            return alignment_loss
            
        except Exception as e:
            if self.debug_distillation and self.distill_step_count < 5:
                LOGGER.warning(f"DEBUG Align: Error in feature alignment: {e}")
                import traceback
                LOGGER.warning(f"DEBUG Align: Traceback: {traceback.format_exc()}")
            return torch.tensor(0.0, device=self.device)
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if cfg is None:
            cfg = DEFAULT_CFG
        
        if overrides is None:
            overrides = {}
        
        self.distill_weight = overrides.pop('distill_weight', 0.5)
        self.temperature = overrides.pop('temperature', 4.0)
        self.teacher_weights_path = overrides.pop('teacher_weights', None)
        
        self.debug_distillation = True
        self.distill_step_count = 0
        
        # Initialize loss_items attribute
        self.loss_items = None
        
        self.custom_distill_args = {
            'distill_weight': self.distill_weight,
            'temperature': self.temperature,
            'teacher_weights': self.teacher_weights_path
        }
        
        super().__init__(cfg, overrides, _callbacks)
        
        self.teacher_model = None
        self._setup_teacher_model()

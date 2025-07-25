import torch
import torch.nn.functional as F
import time
import warnings
import math
import numpy as np
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.model import RTDETR
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK, TQDM, colorstr
from ultralytics.utils.torch_utils import autocast, unset_deterministic
from ultralytics.cfg import get_cfg
from pathlib import Path
from torch import distributed as dist



class RTDETRDistillationTrainer(RTDETRTrainer):
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the distillation trainer with teacher model configuration."""
        # Ensure cfg is not None
        if cfg is None:
            cfg = DEFAULT_CFG
        
        # Ensure overrides is a dictionary
        if overrides is None:
            overrides = {}
        
        # Extract custom distillation arguments before parent initialization
        self.distill_weight = overrides.pop('distill_weight', 0.5)
        self.temperature = overrides.pop('temperature', 4.0)
        self.teacher_weights_path = overrides.pop('teacher_weights', None)
        
        # Debug flags
        self.debug_distillation = True
        self.distill_step_count = 0
        
        # ADD THIS: Training mode flag to control teacher inference
        self.training_mode = True
        
        # Store custom arguments for later use
        self.custom_distill_args = {
            'distill_weight': self.distill_weight,
            'temperature': self.temperature,
            'teacher_weights': self.teacher_weights_path
        }
        
        # Initialize parent trainer with cleaned overrides
        super().__init__(cfg, overrides, _callbacks)
        
        # Initialize teacher model after parent initialization
        self.teacher_model = None
        self._setup_teacher_model()

    def criterion(self, preds, batch):
        """
        Compute combined loss: detection loss + distillation loss.
        
        Args:
            preds: Student model predictions
            batch: Training batch data
            
        Returns:
            Combined loss and loss items
        """
        # Get standard detection loss from parent
        loss, loss_items = super().criterion(preds, batch)
        
        # Debug logging
        if self.debug_distillation and self.distill_step_count < 5:
            LOGGER.info(f"DEBUG Step {self.distill_step_count}: Standard loss = {loss.item():.4f}")
            LOGGER.info(f"DEBUG Step {self.distill_step_count}: Training mode = {self.training_mode}")
            LOGGER.info(f"DEBUG Step {self.distill_step_count}: Model training = {self.model.training}")
        
        # MODIFIED: Only use teacher during training, not validation/testing
        if self.teacher_model is None or not self.training_mode or not self.model.training:
            if self.debug_distillation and self.distill_step_count < 5:
                LOGGER.info(f"DEBUG Step {self.distill_step_count}: Skipping teacher inference (validation/testing mode)")
            self.distill_step_count += 1
            return loss, loss_items
        
        # Get teacher predictions (no gradient computation) - ONLY DURING TRAINING
        try:
            with torch.no_grad():
                # Ensure teacher model is in eval mode
                self.teacher_model.model.eval()
                teacher_preds = self.teacher_model.model(batch['img'])
            
            # Debug teacher predictions
            if self.debug_distillation and self.distill_step_count < 5:
                LOGGER.info(f"DEBUG Step {self.distill_step_count}: Teacher inference executed")
            
            # Compute distillation loss
            distill_loss = self._compute_distillation_loss(preds, teacher_preds)
            
            # Combine losses
            total_loss = (1 - self.distill_weight) * loss + self.distill_weight * distill_loss
            
            # Add distillation loss to loss_items for logging
            if isinstance(loss_items, torch.Tensor):
                distill_loss_item = distill_loss.detach()
                loss_items = torch.cat([loss_items, distill_loss_item.unsqueeze(0)])
            
            self.distill_step_count += 1
            return total_loss, loss_items
        
        except Exception as e:
            LOGGER.warning(f"Error computing distillation loss: {e}")
            if self.debug_distillation:
                import traceback
                LOGGER.warning(f"Full traceback: {traceback.format_exc()}")
            self.distill_step_count += 1
            return loss, loss_items

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
    

    def validate(self):
        """
        Override to disable teacher during validation to reduce FLOPS.
        
        Returns:
            metrics (dict): Dictionary of validation metrics.
            fitness (float): Fitness score for the validation.
        """
        # Disable teacher inference during validation
        original_training_mode = self.training_mode
        self.training_mode = False
        
        try:
            # Run parent validation (this will call criterion without teacher)
            metrics, fitness = super().validate()
            return metrics, fitness
        finally:
            # Always restore original training mode
            self.training_mode = original_training_mode


    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        
        if self.teacher_model is not None:
            self.teacher_model.model.to(self.device)
            LOGGER.info(f"Teacher model moved to device: {self.device}")
    
    
    
    


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
                                batch_size = 16  # Your batch size
                                queries = pred.shape[0] // batch_size
                                features = pred.shape[1]
                                if queries > 0:  # Add safety check
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
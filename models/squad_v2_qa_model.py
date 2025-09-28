"""
SQuAD v2 Model Extension for Existing Codebase

This extends the standard AutoModelForQuestionAnswering with an answerability head
for proper SQuAD v2 support, integrating with the existing experiment framework.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering
import logging

logger = logging.getLogger(__name__)


class SquadV2QuestionAnsweringModel(nn.Module):
    """
    SQuAD v2 model that extends standard QA model with answerability head.
    
    ARCHITECTURAL FIX: Use direct pretrained QA model + add answerability head
    """
    
    def __init__(self, model_name: str, answerability_weight: float = 1.0):
        super().__init__()
        
        # ARCHITECTURAL FIX: Load CausalLM and manually add QA head (preserves pretrained weights)
        from transformers import AutoModelForCausalLM
        import torch.nn as nn
        
        logger.info(f"Loading PRETRAINED CausalLM model: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Extract the base transformer (preserves all pretrained weights)
        self.transformer = base_model.model  # LlamaModel with pretrained weights
        
        # Manually add QA head (like LlamaForQuestionAnswering)
        hidden_size = base_model.config.hidden_size
        self.qa_outputs = nn.Linear(hidden_size, 2)  # start/end positions
        
        logger.info("✅ Transformer loaded with pretrained weights")
        logger.info("✅ QA head (qa_outputs) initialized randomly for fine-tuning")
        
        # Add answerability head  
        hidden_size = base_model.config.hidden_size
        self.answerability_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 2)  # [unanswerable, answerable]
        )
        
        self.answerability_weight = answerability_weight
        
        # Expose config and other attributes for compatibility
        self.config = base_model.config
        self.config.num_labels = 2  # For QA positions
        self.num_labels = 2  # For answerability
        
        logger.info(f"Initialized SQuAD v2 model with answerability head (weight: {answerability_weight})")
        logger.info("🎯 FIXED: Pretrained transformer weights preserved, only QA + answerability heads are new")
    
    @property
    def device(self):
        """Get the device of the underlying transformer model."""
        return self.transformer.device
    
    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing for the underlying transformer."""
        if hasattr(self.transformer, 'gradient_checkpointing_enable'):
            self.transformer.gradient_checkpointing_enable(**kwargs)
    
    def gradient_checkpointing_disable(self, **kwargs):
        """Disable gradient checkpointing for the underlying transformer."""
        if hasattr(self.transformer, 'gradient_checkpointing_disable'):
            self.transformer.gradient_checkpointing_disable(**kwargs)
    
    def get_input_embeddings(self):
        """Get input embeddings from the underlying transformer."""
        return self.transformer.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """Set input embeddings for the underlying transformer."""
        return self.transformer.set_input_embeddings(value)
    
    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings for the underlying transformer."""
        return self.transformer.resize_token_embeddings(new_num_tokens)
    
    def forward(self, input_ids, attention_mask=None, start_positions=None, 
                end_positions=None, answerability_labels=None, **kwargs):
        """
        Forward pass compatible with Trainer.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            start_positions: Start positions for span (training only)
            end_positions: End positions for span (training only)  
            answerability_labels: Answerability labels (training only)
            
        Returns:
            Dictionary with logits and losses
        """
        
        # Get transformer outputs (manually implement QA forward pass)
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        
        # Extract last hidden state and apply QA head
        last_hidden_state = transformer_outputs.last_hidden_state
        qa_logits = self.qa_outputs(last_hidden_state)
        
        # Split into start and end logits
        start_logits = qa_logits[:, :, 0]
        end_logits = qa_logits[:, :, 1]
        
        # Create QA outputs structure (compatible with original)
        qa_outputs = type('QAOutputs', (), {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'hidden_states': transformer_outputs.hidden_states,
            'attentions': getattr(transformer_outputs, 'attentions', None),
            'loss': None  # Will be calculated below
        })()
        
        # Calculate QA loss if positions provided
        if start_positions is not None and end_positions is not None:
            import torch.nn as nn
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            qa_outputs.loss = (start_loss + end_loss) / 2
        
        # Get hidden states for answerability classification
        hidden_states = qa_outputs.hidden_states[-1]  # Last layer
        
        # Use mean pooling for answerability (captures full context)
        # Apply attention mask to avoid pooling over padding tokens
        if attention_mask is not None:
            # Expand attention mask to match hidden_states dimensions
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            # Zero out hidden states for padded tokens
            hidden_states_masked = hidden_states * attention_mask_expanded
            # Sum and normalize by actual sequence length
            sum_embeddings = torch.sum(hidden_states_masked, dim=1)
            sum_mask = torch.sum(attention_mask_expanded, dim=1)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_size]
            
        answerability_logits = self.answerability_classifier(pooled_output)
        
        # ROOT CAUSE FIX: Ensure all output values are valid tensors, never None
        # This prevents _pad_across_processes errors during evaluation
        outputs = {
            "start_logits": qa_outputs.start_logits,
            "end_logits": qa_outputs.end_logits,
            "answerability_logits": answerability_logits,
            "hidden_states": qa_outputs.hidden_states,
        }
        
        # Only include attentions if they exist and are not None
        if hasattr(qa_outputs, 'attentions') and qa_outputs.attentions is not None:
            outputs["attentions"] = qa_outputs.attentions
        
        # ROOT CAUSE FIX: Extract answerability_labels from kwargs if not provided directly
        # This handles PEFT/LoRA case where custom parameters get filtered out
        if answerability_labels is None:
            answerability_labels = kwargs.get('answerability_labels', None)
        
        # Calculate losses during training
        if answerability_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            
            # 1. Span extraction loss (only for answerable questions)
            if start_positions is not None and end_positions is not None:
                # Mask out unanswerable questions for span loss
                answerable_mask = (answerability_labels == 1)
                
                if answerable_mask.sum() > 0:
                    # Use QA model's loss for answerable questions
                    if qa_outputs.loss is not None:
                        span_loss = qa_outputs.loss
                    else:
                        # Create tensor with matching dtype and device
                        span_loss = torch.tensor(0.0, device=input_ids.device, 
                                               dtype=torch.float32, requires_grad=True)
                else:
                    # Create tensor with matching dtype and device  
                    span_loss = torch.tensor(0.0, device=input_ids.device,
                                           dtype=torch.float32, requires_grad=True)
            else:
                # Create tensor with matching dtype and device
                span_loss = torch.tensor(0.0, device=input_ids.device,
                                       dtype=torch.float32, requires_grad=True)
            
            # 2. Answerability classification loss
            answerability_loss = loss_fct(answerability_logits, answerability_labels)
            
            # 3. Combined loss with numerical stability checks
            total_loss = span_loss + self.answerability_weight * answerability_loss
            
            # NUMERICAL STABILITY FIX: Check for NaN/Inf and clamp if needed
            if not torch.isfinite(total_loss).all():
                logger.warning(f"⚠️ Non-finite loss detected! span_loss={span_loss.item():.4f}, "
                             f"answerability_loss={answerability_loss.item():.4f}")
                # Clamp to prevent NaN/Inf propagation
                total_loss = torch.clamp(total_loss, min=-100.0, max=100.0)
                
            outputs.update({
                "loss": total_loss,
                "span_loss": span_loss,
                "answerability_loss": answerability_loss
            })
        
        return outputs
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save model for compatibility with Trainer."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the transformer (preserving pretrained structure)
        from transformers import LlamaForCausalLM
        
        # Reconstruct CausalLM for saving
        causal_model = LlamaForCausalLM(self.config)
        causal_model.model = self.transformer
        causal_model.save_pretrained(save_directory, **kwargs)
        
        # Save QA and answerability heads separately
        heads_path = f"{save_directory}/qa_heads.pt"
        torch.save({
            'qa_outputs': self.qa_outputs.state_dict(),
            'answerability_classifier': self.answerability_classifier.state_dict(),
            'answerability_weight': self.answerability_weight
        }, heads_path)
    
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """Load model for compatibility with Trainer."""
        from transformers import AutoModelForCausalLM
        import torch
        
        # Load the base CausalLM model (preserves pretrained weights)
        base_model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        
        # Create new instance
        instance = cls.__new__(cls)
        super(SquadV2QuestionAnsweringModel, instance).__init__()
        
        # Extract transformer and create QA head
        instance.transformer = base_model.model
        hidden_size = base_model.config.hidden_size
        instance.qa_outputs = nn.Linear(hidden_size, 2)
        
        # Try to load QA and answerability heads
        try:
            heads_path = f"{model_path}/qa_heads.pt"
            checkpoint = torch.load(heads_path, map_location='cpu')
            
            # Load QA head
            instance.qa_outputs.load_state_dict(checkpoint['qa_outputs'])
            
            # Load answerability head
            instance.answerability_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 4, 2)
            )
            instance.answerability_classifier.load_state_dict(checkpoint['answerability_classifier'])
            instance.answerability_weight = checkpoint.get('answerability_weight', 1.0)
        except:
            # Initialize new heads if not found
            instance.answerability_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 4, 2)
            )
            instance.answerability_weight = 1.0
        
        instance.config = base_model.config
        instance.config.num_labels = 2
        instance.num_labels = 2
        
        return instance


def create_squad_v2_model(model_name: str, **kwargs):
    """Factory function to create SQuAD v2 model."""
    return SquadV2QuestionAnsweringModel(model_name, **kwargs)

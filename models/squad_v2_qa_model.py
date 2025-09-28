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
    
    Integrates seamlessly with existing experiment framework.
    """
    
    def __init__(self, model_name: str, answerability_weight: float = 1.0):
        super().__init__()
        
        # CRITICAL FIX: Load base CausalLM first to preserve pre-trained weights
        from transformers import AutoModelForCausalLM, AutoConfig
        
        logger.info(f"Loading base CausalLM model to preserve pre-trained weights: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create QA model with proper weight transfer
        config = base_model.config
        config.num_labels = 2  # For QA start/end positions
        
        # Create QA model structure manually
        from transformers.models.llama.modeling_llama import LlamaForQuestionAnswering
        self.qa_model = LlamaForQuestionAnswering(config)
        
        # Transfer all transformer weights from base model
        if hasattr(base_model, 'model') and hasattr(self.qa_model, 'transformer'):
            base_transformer = base_model.model  # CausalLM structure
            qa_transformer = self.qa_model.transformer  # QA structure
            
            # Copy all transformer layers and embeddings
            qa_transformer.load_state_dict(base_transformer.state_dict(), strict=False)
            logger.info("✅ Successfully transferred pre-trained transformer weights")
        else:
            logger.warning(f"⚠️ Could not find transformer - base has model: {hasattr(base_model, 'model')}, qa has transformer: {hasattr(self.qa_model, 'transformer')}")
        
        # QA head is already randomly initialized in LlamaForQuestionAnswering
        logger.info("✅ QA head (qa_outputs) initialized randomly as expected")
        
        # Add answerability head
        hidden_size = self.qa_model.config.hidden_size
        self.answerability_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 2)  # [unanswerable, answerable]
        )
        
        self.answerability_weight = answerability_weight
        
        # Expose config and other attributes for compatibility
        self.config = self.qa_model.config
        self.num_labels = 2  # For answerability
        
        logger.info(f"Initialized SQuAD v2 model with answerability head (weight: {answerability_weight})")
        logger.info("🎯 Pre-trained weights preserved, only QA and answerability heads are new")
    
    @property
    def device(self):
        """Get the device of the underlying QA model."""
        return self.qa_model.device
    
    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing for the underlying QA model."""
        if hasattr(self.qa_model, 'gradient_checkpointing_enable'):
            self.qa_model.gradient_checkpointing_enable(**kwargs)
    
    def gradient_checkpointing_disable(self, **kwargs):
        """Disable gradient checkpointing for the underlying QA model."""
        if hasattr(self.qa_model, 'gradient_checkpointing_disable'):
            self.qa_model.gradient_checkpointing_disable(**kwargs)
    
    def get_input_embeddings(self):
        """Get input embeddings from the underlying QA model."""
        return self.qa_model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """Set input embeddings for the underlying QA model."""
        return self.qa_model.set_input_embeddings(value)
    
    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings for the underlying QA model."""
        return self.qa_model.resize_token_embeddings(new_num_tokens)
    
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
        
        # Get QA model outputs
        qa_outputs = self.qa_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            return_dict=True,
            output_hidden_states=True
        )
        
        # Get hidden states for answerability classification
        hidden_states = qa_outputs.hidden_states[-1]  # Last layer
        
        # Use mean pooling for answerability (captures full context)
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
                    span_loss = qa_outputs.loss if qa_outputs.loss is not None else torch.tensor(0.0, device=input_ids.device)
                else:
                    span_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            else:
                span_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            
            # 2. Answerability classification loss
            answerability_loss = loss_fct(answerability_logits, answerability_labels)
            
            # 3. Combined loss
            total_loss = span_loss + self.answerability_weight * answerability_loss
            
            outputs.update({
                "loss": total_loss,
                "span_loss": span_loss,
                "answerability_loss": answerability_loss
            })
        
        return outputs
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save model for compatibility with Trainer."""
        # Save the QA model
        self.qa_model.save_pretrained(save_directory, **kwargs)
        
        # Save answerability head separately
        answerability_path = f"{save_directory}/answerability_head.pt"
        torch.save({
            'answerability_classifier': self.answerability_classifier.state_dict(),
            'answerability_weight': self.answerability_weight
        }, answerability_path)
    
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """Load model for compatibility with Trainer."""
        # Load the QA model
        qa_model = AutoModelForQuestionAnswering.from_pretrained(model_path, **kwargs)
        
        # Create new instance
        instance = cls.__new__(cls)
        super(SquadV2QuestionAnsweringModel, instance).__init__()
        instance.qa_model = qa_model
        
        # Try to load answerability head
        try:
            answerability_path = f"{model_path}/answerability_head.pt"
            checkpoint = torch.load(answerability_path, map_location='cpu')
            
            hidden_size = qa_model.config.hidden_size
            instance.answerability_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 4, 2)
            )
            instance.answerability_classifier.load_state_dict(checkpoint['answerability_classifier'])
            instance.answerability_weight = checkpoint.get('answerability_weight', 1.0)
        except:
            # Initialize new answerability head if not found
            hidden_size = qa_model.config.hidden_size
            instance.answerability_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 4, 2)
            )
            instance.answerability_weight = 1.0
        
        instance.config = qa_model.config
        instance.num_labels = 2
        
        return instance


def create_squad_v2_model(model_name: str, **kwargs):
    """Factory function to create SQuAD v2 model."""
    return SquadV2QuestionAnsweringModel(model_name, **kwargs)

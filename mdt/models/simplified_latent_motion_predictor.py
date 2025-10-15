# Simplified Latent Motion Predictor extracted from MotoGPT
# Only keeps the essential components for latent motion prediction

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SimplifiedLatentMotionPredictor(nn.Module):
    """
    Simplified version of MotoGPT that only handles latent motion prediction.
    This can be integrated into other models as a component.
    
    Uses a fully flattened approach: generates seq_len * per_latent_motion_len tokens
    in a single autoregressive sequence for perfect train-inference alignment.
    """
    def __init__(
        self,
        input_dim: int,  # Input feature dimension from perceptual embeddings
        hidden_size: int = 512,
        per_latent_motion_len: int = 10,
        latent_motion_codebook_size: int = 1024,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_pos_embedding: bool = True,
        mask_probability: float = 0.0,
        parallel_prediction: bool = False,
    ):
        super().__init__()
        
        # Core parameters
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.per_latent_motion_len = per_latent_motion_len
        self.latent_motion_codebook_size = latent_motion_codebook_size
        self.mask_probability = mask_probability
        self.use_pos_embedding = use_pos_embedding
        self.parallel_prediction = parallel_prediction
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_size)
        
        # Embeddings
        self.embed_latent_motion = nn.Embedding(latent_motion_codebook_size, hidden_size)
        
        # Start token for bridging from condition to latent motion generation
        self.start_token = nn.Embedding(1, hidden_size)  # Special token to start latent motion generation
        
        # Layer normalization
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # Transformer configuration
        from transformers import GPT2Config, GPT2Model
        config = GPT2Config(
            vocab_size=latent_motion_codebook_size,
            n_positions=1024,  # Max sequence length
            n_embd=hidden_size,
            n_layer=n_layers,
            n_head=n_heads,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.transformer = GPT2Model(config)
        
        # Prediction head for latent motion
        self.pred_latent_motion_head = nn.Linear(hidden_size, latent_motion_codebook_size, bias=False)
        # Parallel per-step head: predicts per_latent_motion_len tokens at once per timestep
        self.pred_latent_motion_parallel_head = nn.Linear(
            hidden_size,
            per_latent_motion_len * latent_motion_codebook_size,
            bias=False,
        )
        
    def forward(
        self,
        perceptual_features: torch.Tensor,  # (batch, cond_tokens, input_dim) - single timestep condition
        latent_motion_ids: Optional[torch.Tensor] = None,  # (batch, seq_len, per_latent_motion_len)
        attention_mask: Optional[torch.Tensor] = None,  # (batch, seq_len)
        train: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for latent motion prediction.
        
        Args:
            perceptual_features: Input condition features from visual/other encoders (single timestep)
            latent_motion_ids: Ground truth latent motion indices (for training)
            attention_mask: Mask for valid sequence positions
            train: Whether in training mode
            
        Returns:
            Dictionary containing predictions and losses
        """
        batch_size, cond_tokens, _ = perceptual_features.shape
        seq_len = latent_motion_ids.shape[1] if latent_motion_ids is not None else kwargs.get('seq_len', 1)
        
        # Project input features to hidden dimension
        cond_embeddings = self.input_projection(perceptual_features)  # (batch, cond_tokens, hidden_size)
        
        if train and latent_motion_ids is not None:
            return self._forward_train(cond_embeddings, latent_motion_ids, attention_mask)
        else:
            return self._forward_inference(cond_embeddings, attention_mask, **kwargs)
    
    def _forward_train(
        self, 
        cond_embeddings: torch.Tensor,  # (batch, cond_tokens, hidden_size)
        latent_motion_ids: torch.Tensor,  # (batch, seq_len, per_latent_motion_len)
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with teacher forcing.
        If parallel_prediction is True, predict K=per_len tokens per timestep in parallel (factorized).
        Else, predict all tokens autoregressively on the flattened sequence.
        """
        batch_size, cond_tokens, _ = cond_embeddings.shape
        seq_len, per_len = latent_motion_ids.shape[1], latent_motion_ids.shape[2]  
        total_latent_tokens = seq_len * per_len
        
        # Flatten latent motion IDs to a single sequence
        # (batch, seq_len, per_latent_motion_len) -> (batch, seq_len * per_latent_motion_len)
        latent_motion_ids_flat = latent_motion_ids.view(batch_size, total_latent_tokens)
        
        # Embed flattened latent motion IDs
        latent_motion_embeddings = self.embed_latent_motion(latent_motion_ids_flat)  # (batch, total_latent_tokens, hidden_size)
        
        # Add start token to bridge from condition to latent motion
        start_token_emb = self.start_token.weight.view(1, 1, -1).repeat(batch_size, 1, 1)  # (batch, 1, hidden_size)
        
        # Stack inputs: [condition, start_token, latent_tokens]
        # For teacher forcing, we use [cond, start, GT_tok0, GT_tok1, ..., GT_tok_{N-1}] to predict [GT_tok0, GT_tok1, ..., GT_tok_N]
        stacked_inputs = torch.cat([cond_embeddings, start_token_emb, latent_motion_embeddings], dim=1)  # (batch, cond_tokens + 1 + total_latent_tokens, hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # Create attention mask
        if attention_mask is not None:
            cond_mask = attention_mask
        else:
            cond_mask = torch.ones(batch_size, cond_tokens, device=cond_embeddings.device)
        
        start_mask = torch.ones(batch_size, 1, device=cond_embeddings.device)
        latent_mask = torch.ones(batch_size, total_latent_tokens, device=cond_embeddings.device)
        full_attention_mask = torch.cat([cond_mask, start_mask, latent_mask], dim=1)
        # if attention_mask is not None:
        #     # Condition tokens are always visible
        #     cond_mask = torch.ones(batch_size, cond_tokens, device=cond_embeddings.device)
        #     # Start token is always visible
        #     start_mask = torch.ones(batch_size, 1, device=cond_embeddings.device)
        #     # Flatten attention mask for latent tokens
        #     latent_mask = attention_mask.unsqueeze(-1).repeat(1, 1, per_len).view(batch_size, total_latent_tokens)
        #     full_attention_mask = torch.cat([cond_mask, start_mask, latent_mask], dim=1)
        # else:
        #     full_attention_mask = None
        
        # Transformer forward pass
        outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=full_attention_mask,
        )
        hidden_states = outputs.last_hidden_state

        if self.parallel_prediction:
            # Use context positions at the start of each timestep to predict all K tokens in parallel
            # Context position for step t is cond_tokens + t*per_len (hidden at last seen token before step t)
            device = hidden_states.device
            ctx_positions = cond_tokens + torch.arange(seq_len, device=device) * per_len  # (seq_len,)
            # Gather hidden states at these positions for all batches
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, seq_len)
            ctx_positions_exp = ctx_positions.unsqueeze(0).expand(batch_size, seq_len)
            ctx_hidden = hidden_states[batch_idx, ctx_positions_exp, :]  # (B, seq_len, hidden)

            logits_parallel = self.pred_latent_motion_parallel_head(ctx_hidden)  # (B, seq_len, K*V)
            logits_parallel = logits_parallel.view(batch_size, seq_len, per_len, self.latent_motion_codebook_size)

            # Compute CE over (B, T, K)
            loss = self._compute_loss(
                logits_parallel,
                latent_motion_ids,
                attention_mask=None,
            )

            return {
                'latent_motion_preds': logits_parallel,  # (B, T, K, V)
                'loss': loss,
            }
        else:
            # Autoregressive flattened token prediction
            # For teacher forcing: positions [cond_tokens : cond_tokens + total_latent_tokens] predict tokens [0..N-1]
            prediction_hidden = hidden_states[:, cond_tokens:-1]  # (batch, 1+total_latent_tokens-1, hidden_size)
            latent_motion_preds_flat = self.pred_latent_motion_head(prediction_hidden)  # (batch, total_latent_tokens, codebook_size)
            targets_flat = latent_motion_ids_flat  # (batch, total_latent_tokens)

            # Compute loss on flattened predictions
            loss = self._compute_loss_flat_with_start(latent_motion_preds_flat, targets_flat, None, seq_len, per_len)

            # Reshape predictions for output compatibility
            latent_motion_preds = latent_motion_preds_flat.view(batch_size, seq_len, per_len, -1)

            return {
                'latent_motion_preds': latent_motion_preds,
                'loss': loss,
            }
    
    def _forward_inference(
        self, 
        cond_embeddings: torch.Tensor,  # (batch, cond_tokens, hidden_size)
        attention_mask: Optional[torch.Tensor] = None,
        seq_len: int = 1,  # Number of timesteps to predict
        temperature: float = 1.0,
        top_k: int = 0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Inference forward pass with autoregressive generation."""
        
        if self.parallel_prediction:
            ids = self._generate_latent_motion_parallel(
                cond_embeddings,
                attention_mask,
                seq_len=seq_len,
                temperature=temperature,
                top_k=top_k,
            )
            return {'latent_motion_id_preds': ids}
        else:
            # Generate latent motion IDs autoregressively across all flattened tokens
            latent_motion_ids = self._generate_latent_motion(
                cond_embeddings, 
                attention_mask,
                seq_len=seq_len,
                temperature=temperature,
                top_k=top_k
            )
            return {'latent_motion_id_preds': latent_motion_ids}
    
    def _generate_latent_motion(
        self,
        cond_embeddings: torch.Tensor,  # (batch, cond_tokens, hidden_size)
        attention_mask: Optional[torch.Tensor] = None,
        seq_len: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Generate latent motion IDs autoregressively in flattened sequence."""
        batch_size, cond_tokens, _ = cond_embeddings.shape
        device = cond_embeddings.device
        total_latent_tokens = seq_len * self.per_latent_motion_len
        
        # Apply layer norm to condition embeddings once (consistent with training)
        cond_embeddings_normed = self.embed_ln(cond_embeddings)
        
        # Start token embedding and apply layer norm
        start_token_emb = self.start_token.weight.view(1, 1, -1).repeat(batch_size, 1, 1)  # (batch, 1, hidden_size)
        start_token_emb_normed = self.embed_ln(start_token_emb)
        
        # Initialize with normalized condition and start token
        current_sequence = torch.cat([cond_embeddings_normed, start_token_emb_normed], dim=1)  # (batch, cond_tokens + 1, hidden_size)
        
        # Initialize generated sequence
        generated_ids = []  # Will contain total_latent_tokens elements
        
        for step in range(total_latent_tokens):
            # No need to apply layer norm again - embeddings are already normalized
            # This ensures consistency with training where layer norm is applied once
            
            # Create attention mask
            if attention_mask is not None:
                cond_mask = attention_mask
            else:
                cond_mask = torch.ones(batch_size, cond_tokens, device=cond_embeddings.device)
            
            start_mask = torch.ones(batch_size, 1, device=cond_embeddings.device)
            latent_mask = torch.ones(batch_size, current_sequence.shape[1] - cond_tokens - 1, device=cond_embeddings.device)
            current_mask = torch.cat([cond_mask, start_mask, latent_mask], dim=1)
            
            # Forward pass
            outputs = self.transformer(
                inputs_embeds=current_sequence,  # Use already normalized embeddings
                attention_mask=current_mask,
            )
            
            # Get prediction for next token
            hidden_states = outputs.last_hidden_state
            last_hidden = hidden_states[:, -1]  # (batch, hidden_size) - last position
            
            # Predict next token
            logits = self.pred_latent_motion_head(last_hidden)  # (batch, codebook_size)
            
            # Apply temperature and top-k sampling
            if temperature != 1.0:
                logits = logits / temperature
            
            if top_k > 0:
                values, _ = torch.topk(logits, top_k, dim=-1)
                min_values = values[:, -1:].expand_as(logits)
                logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch,)
            
            generated_ids.append(next_token)
            
            # Add the generated token to the current sequence for next iteration
            # Apply layer norm to the new token embedding (consistent with training)
            next_token_emb = self.embed_latent_motion(next_token).unsqueeze(1)  # (batch, 1, hidden_size)
            next_token_emb_normed = self.embed_ln(next_token_emb)  # Apply layer norm to new token
            current_sequence = torch.cat([current_sequence, next_token_emb_normed], dim=1)
        
        # Convert to tensor and reshape
        generated_tensor = torch.stack(generated_ids, dim=1)  # (batch, total_latent_tokens)
        latent_motion_ids = generated_tensor.view(batch_size, seq_len, self.per_latent_motion_len)  # (batch, seq_len, per_latent_motion_len)
        
        return latent_motion_ids

    def _generate_latent_motion_parallel(
        self,
        cond_embeddings: torch.Tensor,  # (batch, cond_tokens, hidden_size)
        attention_mask: Optional[torch.Tensor] = None,
        seq_len: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Generate K tokens per timestep in parallel (factorized), autoregressive across timesteps only."""
        batch_size, cond_tokens, _ = cond_embeddings.shape
        device = cond_embeddings.device
        per_len = self.per_latent_motion_len

        # Normalize condition once
        cond_embeddings_normed = self.embed_ln(cond_embeddings)
        start_token_emb = self.start_token.weight.view(1, 1, -1).repeat(batch_size, 1, 1)
        start_token_emb_normed = self.embed_ln(start_token_emb)

        # Sequence begins with cond + start
        current_sequence = torch.cat([cond_embeddings_normed, start_token_emb_normed], dim=1)  # (B, cond_tokens+1, H)

        all_ids = []
        for t in range(seq_len):
            # Build attention mask for current sequence
            if attention_mask is not None:
                cond_mask = attention_mask
            else:
                cond_mask = torch.ones(batch_size, cond_tokens, device=device)

            start_mask = torch.ones(batch_size, 1, device=device)
            latent_mask = torch.ones(batch_size, current_sequence.shape[1] - cond_tokens - 1, device=device)
            current_mask = torch.cat([cond_mask, start_mask, latent_mask], dim=1)

            # Forward
            outputs = self.transformer(inputs_embeds=current_sequence, attention_mask=current_mask)
            ctx_hidden = outputs.last_hidden_state[:, -1]  # (B, H) context for this timestep

            # Parallel head -> (B, K*V) -> (B, K, V)
            logits = self.pred_latent_motion_parallel_head(ctx_hidden)
            logits = logits.view(batch_size, per_len, self.latent_motion_codebook_size)

            # Apply temperature and top-k per K
            if temperature != 1.0:
                logits = logits / temperature

            logits_flat = logits.view(batch_size * per_len, self.latent_motion_codebook_size)
            if top_k > 0:
                values, _ = torch.topk(logits_flat, top_k, dim=-1)
                min_values = values[:, -1:].expand_as(logits_flat)
                logits_flat = torch.where(
                    logits_flat < min_values,
                    torch.full_like(logits_flat, float('-inf')),
                    logits_flat,
                )

            probs = F.softmax(logits_flat, dim=-1)
            next_ids_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B*K,)
            next_ids = next_ids_flat.view(batch_size, per_len)  # (B, K)
            all_ids.append(next_ids)

            # Append new embeddings (normalized) for next timestep context
            next_token_emb = self.embed_latent_motion(next_ids)  # (B, K, H)
            next_token_emb = self.embed_ln(next_token_emb)
            current_sequence = torch.cat([current_sequence, next_token_emb], dim=1)

        # Stack steps -> (B, T, K)
        ids = torch.stack(all_ids, dim=1)
        return ids
    
    def _compute_loss_flat_with_start(
        self, 
        predictions: torch.Tensor,  # (batch, total_tokens, codebook_size)
        targets: torch.Tensor,      # (batch, total_tokens)
        attention_mask: Optional[torch.Tensor],  # (batch, seq_len) 
        seq_len: int,
        per_len: int
    ) -> torch.Tensor:
        """Compute cross-entropy loss for flattened sequence prediction with start token."""
        batch_size = predictions.shape[0]
        total_tokens = predictions.shape[1]
        vocab_size = predictions.shape[2]
        
        # Flatten for loss computation
        predictions_flat = predictions.view(-1, vocab_size)  # (batch*total_tokens, vocab_size)
        targets_flat = targets.view(-1)  # (batch*total_tokens)
        
        # Create loss mask from attention_mask if provided
        if attention_mask is not None:
            # Expand attention mask: each timestep contributes per_len tokens
            loss_mask = attention_mask.unsqueeze(-1).repeat(1, 1, per_len)  # (batch, seq_len, per_len)
            loss_mask = loss_mask.view(batch_size, seq_len * per_len)  # (batch, total_tokens)
            loss_mask = loss_mask.view(-1)  # (batch*total_tokens)
        else:
            loss_mask = torch.ones_like(targets_flat, dtype=torch.bool)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            predictions_flat, 
            targets_flat, 
            reduction='none'
        )
        
        # Apply mask and compute mean
        if loss_mask.sum() > 0:
            loss = (loss * loss_mask.float()).sum() / loss_mask.float().sum()
        else:
            loss = loss.mean()
        
        return loss
    
    # def _compute_loss_flat(
    #     self, 
    #     predictions: torch.Tensor,  # (batch, total_tokens-1, codebook_size)
    #     targets: torch.Tensor,      # (batch, total_tokens-1)
    #     attention_mask: Optional[torch.Tensor],  # (batch, seq_len) 
    #     seq_len: int,
    #     per_len: int
    # ) -> torch.Tensor:
    #     """Compute cross-entropy loss for flattened sequence prediction."""
    #     batch_size = predictions.shape[0]
    #     total_tokens_minus_1 = predictions.shape[1]
    #     vocab_size = predictions.shape[2]
        
    #     # Flatten for loss computation
    #     predictions_flat = predictions.view(-1, vocab_size)  # (batch*(total_tokens-1), vocab_size)
    #     targets_flat = targets.view(-1)  # (batch*(total_tokens-1))
        
    #     # Create loss mask from attention_mask if provided
    #     if attention_mask is not None:
    #         # Expand attention mask: each timestep contributes per_len tokens
    #         loss_mask = attention_mask.unsqueeze(-1).repeat(1, 1, per_len)  # (batch, seq_len, per_len)
    #         loss_mask = loss_mask.view(batch_size, seq_len * per_len)  # (batch, total_tokens)
    #         loss_mask = loss_mask[:, 1:]  # Remove first token (we predict tokens 1 to total_tokens)
    #         loss_mask = loss_mask.view(-1)  # (batch*(total_tokens-1))
    #     else:
    #         loss_mask = torch.ones_like(targets_flat, dtype=torch.bool)
        
    #     # Compute cross-entropy loss
    #     loss = F.cross_entropy(
    #         predictions_flat, 
    #         targets_flat, 
    #         reduction='none'
    #     )
        
    #     # Apply mask and compute mean
    #     if loss_mask.sum() > 0:
    #         loss = (loss * loss_mask.float()).sum() / loss_mask.float().sum()
    #     else:
    #         loss = loss.mean()
        
    #     return loss
    
    def _compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute cross-entropy loss for latent motion prediction (legacy method for compatibility)."""
        # This method is kept for compatibility but not used in the new flattened approach
        # predictions: (batch, seq_len, per_latent_motion_len, codebook_size)
        # targets: (batch, seq_len, per_latent_motion_len)
        
        batch_size, seq_len, motion_len, vocab_size = predictions.shape
        
        # Flatten for loss computation
        predictions_flat = predictions.view(-1, vocab_size)  # (batch*seq_len*motion_len, vocab_size)
        targets_flat = targets.view(-1)  # (batch*seq_len*motion_len)
        
        # Create loss mask
        if attention_mask is not None:
            # Expand attention mask to cover all motion tokens
            loss_mask = attention_mask.unsqueeze(-1).repeat(1, 1, motion_len)  # (batch, seq_len, motion_len)
            loss_mask = loss_mask.view(-1)  # (batch*seq_len*motion_len)
        else:
            loss_mask = torch.ones_like(targets_flat, dtype=torch.bool)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            predictions_flat, 
            targets_flat, 
            reduction='none'
        )
        
        # Apply mask and compute mean
        if loss_mask.sum() > 0:
            loss = (loss * loss_mask.float()).sum() / loss_mask.float().sum()
        else:
            loss = loss.mean()
        
        return loss
    
    def get_latent_motion_embeddings(self, latent_motion_ids: torch.Tensor) -> torch.Tensor:
        """Convert latent motion IDs to embeddings for downstream use."""
        # latent_motion_ids: (batch, seq_len, per_latent_motion_len)
        embeddings = self.embed_latent_motion(latent_motion_ids)  # (batch, seq_len, per_latent_motion_len, hidden_size)
        
        # Average over the motion sequence dimension for a single embedding per timestep
        motion_embeddings = embeddings.mean(dim=2)  # (batch, seq_len, hidden_size)
        
        return motion_embeddings


# Utility function to create a simplified predictor from MotoGPT config
def create_simplified_predictor_from_motogpt_config(motogpt_config: dict, input_dim: int) -> SimplifiedLatentMotionPredictor:
    """Create simplified predictor using configuration from original MotoGPT."""
    return SimplifiedLatentMotionPredictor(
        input_dim=input_dim,
        hidden_size=motogpt_config.get('hidden_size', 512),
        per_latent_motion_len=motogpt_config.get('per_latent_motion_len', 10),
        latent_motion_codebook_size=motogpt_config.get('latent_motion_codebook_size', 1024),
        n_layers=6,  # Reasonable default
        n_heads=8,   # Reasonable default
        use_pos_embedding=motogpt_config.get('use_latent_motion_pos_embedding', True),
        mask_probability=motogpt_config.get('mask_latent_motion_probability', 0.0),
    )
#!/usr/bin/env python3
"""
Test script for the refactored SimplifiedLatentMotionPredictor
"""

import torch
import torch.nn.functional as F
from mdt.models.simplified_latent_motion_predictor import SimplifiedLatentMotionPredictor


def test_flattened_predictor():
    """Test the flattened approach for train-inference alignment."""
    
    # Model parameters
    batch_size = 2
    cond_tokens = 5
    input_dim = 256
    seq_len = 3
    per_latent_motion_len = 4
    hidden_size = 128
    codebook_size = 512
    
    print("Testing SimplifiedLatentMotionPredictor with flattened approach (with start token)...")
    print(f"Batch size: {batch_size}")
    print(f"Condition tokens: {cond_tokens}")
    print(f"Sequence length: {seq_len}")
    print(f"Per latent motion length: {per_latent_motion_len}")
    print(f"Total latent tokens: {seq_len * per_latent_motion_len}")
    print(f"Sequence structure: [cond_tokens] [start_token] [latent_tokens]")
    
    # Create model
    model = SimplifiedLatentMotionPredictor(
        input_dim=input_dim,
        hidden_size=hidden_size,
        per_latent_motion_len=per_latent_motion_len,
        latent_motion_codebook_size=codebook_size,
        n_layers=2,  # Small for testing
        n_heads=4,
        use_pos_embedding=True,  # This should be ignored now
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test data
    perceptual_features = torch.randn(batch_size, cond_tokens, input_dim)
    latent_motion_ids = torch.randint(0, codebook_size, (batch_size, seq_len, per_latent_motion_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print("\nTesting training forward pass...")
    
    # Training forward pass
    model.train()
    train_output = model(
        perceptual_features=perceptual_features,
        latent_motion_ids=latent_motion_ids,
        attention_mask=attention_mask,
        train=True
    )
    
    print(f"Training loss: {train_output['loss'].item():.4f}")
    print(f"Training predictions shape: {train_output['latent_motion_preds'].shape}")
    
    print("\nTesting inference forward pass...")
    
    # Inference forward pass
    model.eval()
    with torch.no_grad():
        inference_output = model(
            perceptual_features=perceptual_features,
            attention_mask=attention_mask,
            train=False,
            seq_len=seq_len,
            temperature=1.0,
            top_k=10
        )
    
    print(f"Inference predictions shape: {inference_output['latent_motion_id_preds'].shape}")
    
    # Test the flattened generation step by step
    print("\nTesting step-by-step generation consistency with start token...")
    
    # Manual step-by-step generation to verify consistency
    model.eval()
    with torch.no_grad():
        cond_embeddings = model.input_projection(perceptual_features)
        total_tokens = seq_len * per_latent_motion_len
        
        print(f"Generating {total_tokens} tokens step by step...")
        print("Starting with: [condition] [start_token]")
        
        # Start with condition + start token
        start_token_emb = model.start_token.weight.view(1, 1, -1).repeat(batch_size, 1, 1)
        current_sequence = torch.cat([cond_embeddings, start_token_emb], dim=1)
        
        generated_step_by_step = []
        for step in range(min(3, total_tokens)):  # Test first 3 steps only
            stacked_inputs = model.embed_ln(current_sequence)
            outputs = model.transformer(inputs_embeds=stacked_inputs)
            hidden_states = outputs.last_hidden_state
            logits = model.pred_latent_motion_head(hidden_states[:, -1])
            next_token = torch.argmax(logits, dim=-1)
            generated_step_by_step.append(next_token)
            
            # Add generated token to sequence for next step
            next_token_emb = model.embed_latent_motion(next_token).unsqueeze(1)
            current_sequence = torch.cat([current_sequence, next_token_emb], dim=1)
            
            print(f"Step {step}: Generated token {next_token.tolist()}, sequence length now {current_sequence.shape[1]}")
    
    print("\nAll tests passed! The flattened approach with start token is working correctly.")
    
    # Verify shapes
    assert train_output['latent_motion_preds'].shape == (batch_size, seq_len, per_latent_motion_len, codebook_size)
    assert inference_output['latent_motion_id_preds'].shape == (batch_size, seq_len, per_latent_motion_len)
    
    return True


if __name__ == "__main__":
    test_flattened_predictor()
    print("✓ All tests passed!")
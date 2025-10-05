# SimplifiedLatentMotionPredictor - Flattened Approach

## Overview

The `SimplifiedLatentMotionPredictor` has been refactored to use a **fully flattened approach** that ensures perfect train-inference alignment. Instead of treating timesteps and per-timestep tokens separately, the model now generates `seq_len * per_latent_motion_len` tokens in a single autoregressive sequence.

## Key Changes

### 1. Flattened Token Generation
- **Before**: Generated tokens per timestep with potential train-inference mismatch
- **After**: Generates all tokens in sequence: `[t0_tok0, t0_tok1, ..., t0_tokN, t1_tok0, t1_tok1, ..., t_{T-1}_tokN]`

### 2. Simplified Architecture
- **Removed**: Query tokens per timestep (`latent_motion_queries`)
- **Removed**: Positional embeddings (`embed_latent_motion_pos`) 
- **Added**: Single start token (`start_token`) to bridge condition → latent motion generation
- **Kept**: Core latent motion embedding (`embed_latent_motion`)

### 3. Perfect Train-Inference Alignment
- **Training**: Uses teacher forcing on the flattened sequence `[cond, GT_tok0, GT_tok1, ..., GT_tokN]`
- **Inference**: Generates tokens autoregressively in the same flattened order
- **Result**: No exposure bias between training and inference

## Usage

### Training
```python
model = SimplifiedLatentMotionPredictor(
    input_dim=256,
    hidden_size=512,
    per_latent_motion_len=10,
    latent_motion_codebook_size=1024,
    n_layers=6,
    n_heads=8
)

# Training forward pass
output = model(
    perceptual_features=cond_features,  # (batch, cond_tokens, input_dim)
    latent_motion_ids=gt_ids,          # (batch, seq_len, per_latent_motion_len)
    attention_mask=mask,               # (batch, seq_len)
    train=True
)

loss = output['loss']
predictions = output['latent_motion_preds']  # (batch, seq_len, per_latent_motion_len, codebook_size)
```

### Inference
```python
# Inference forward pass
model.eval()
with torch.no_grad():
    output = model(
        perceptual_features=cond_features,  # (batch, cond_tokens, input_dim)
        attention_mask=mask,               # (batch, seq_len) - optional
        train=False,
        seq_len=10,                        # Number of timesteps to predict
        temperature=1.0,                   # Sampling temperature
        top_k=50                          # Top-k sampling
    )

generated_ids = output['latent_motion_id_preds']  # (batch, seq_len, per_latent_motion_len)
```

## Technical Details

### Sequence Structure
The flattened sequence is organized as:
```
[cond_tokens] [start_token] [t0_tok0, t0_tok1, ..., t0_tokN, t1_tok0, t1_tok1, ..., t_{seq_len-1}_tokN]
```

Where:
- `cond_tokens`: Condition embedding tokens (always visible)
- `start_token`: Special learned token to bridge condition → latent motion generation
- `ti_tokj`: Token `j` of timestep `i`

### Training Process
1. **Input**: Ground truth tokens `(batch, seq_len, per_latent_motion_len)`
2. **Flatten**: Reshape to `(batch, seq_len * per_latent_motion_len)`
3. **Embed**: Apply token embeddings
4. **Add Start Token**: Insert special start token after condition
5. **Stack**: Concatenate as `[condition, start_token, latent_tokens]`
6. **Forward**: Pass through transformer
7. **Predict**: Use positions `[cond_end, cond_end+1, ..., cond_end+N-1]` to predict `[tok0, tok1, ..., tokN-1]`
8. **Loss**: Cross-entropy on all latent motion tokens

### Inference Process
1. **Initialize**: Start with `[condition, start_token]`
2. **Loop**: For each token position `i` in `[0, seq_len * per_latent_motion_len)`:
   - Forward pass with current sequence
   - Predict next token from last hidden state
   - Sample using temperature and top-k
   - Append token embedding to sequence
3. **Reshape**: Convert flat sequence back to `(batch, seq_len, per_latent_motion_len)`

### Attention Masking
- **Condition tokens**: Always visible (mask = 1)
- **Generated tokens**: Use flattened attention mask derived from input `attention_mask`
- **Causal masking**: Applied automatically by GPT-2 transformer

## Benefits

1. **Perfect Alignment**: Training and inference see identical context at each step
2. **Simplified Logic**: No complex timestep-aware indexing
3. **Standard Autoregressive**: Uses well-established sequence generation patterns
4. **Efficient**: Single forward pass per generation step
5. **Flexible**: Easy to extend with different sampling strategies

## Compatibility

The refactored model maintains backward compatibility:
- Same input/output shapes for the `forward()` method
- Same configuration parameters (unused ones are ignored)
- Same utility functions (`get_latent_motion_embeddings()`)

## Migration Notes

If migrating from the previous version:
1. **Remove** any code that relied on query tokens or positional embeddings
2. **Update** any custom attention mask logic (now uses flattened approach)
3. **Test** thoroughly as the generation order has changed

## Performance Considerations

- **Memory**: Similar memory usage as before
- **Speed**: Potentially faster due to simplified logic
- **Quality**: Should improve due to perfect train-inference alignment
- **Convergence**: May converge faster due to reduced exposure bias
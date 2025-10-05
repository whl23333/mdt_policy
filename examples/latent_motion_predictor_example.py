# Example usage of the simplified latent motion predictor

import torch
from mdt.models.simplified_latent_motion_predictor import SimplifiedLatentMotionPredictor, create_simplified_predictor_from_motogpt_config

def example_usage():
    """示例：如何使用简化的 latent motion 预测器"""
    
    # 1. 创建预测器实例
    predictor = SimplifiedLatentMotionPredictor(
        input_dim=1024,  # 感知特征的维度 (例如：static + gripper features)
        hidden_size=512,
        per_latent_motion_len=10,  # 每个时间步的 latent motion 序列长度
        latent_motion_codebook_size=1024,  # 码书大小
        n_layers=6,
        n_heads=8,
        use_pos_embedding=True,
    )
    
    # 2. 准备输入数据
    batch_size = 4
    seq_len = 8
    input_dim = 1024
    
    # 感知特征 (例如：来自视觉编码器的特征)
    perceptual_features = torch.randn(batch_size, seq_len, input_dim)
    
    # 训练模式：提供 ground truth latent motion IDs
    gt_latent_motion_ids = torch.randint(0, 1024, (batch_size, seq_len, 10))  # (B, T, per_latent_motion_len)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)  # 所有位置都有效
    
    # 3. 训练前向传播
    print("=== Training Mode ===")
    predictor.train()
    train_results = predictor(
        perceptual_features=perceptual_features,
        latent_motion_ids=gt_latent_motion_ids,
        attention_mask=attention_mask,
        train=True
    )
    
    print(f"Training loss: {train_results['loss']:.4f}")
    print(f"Predictions shape: {train_results['latent_motion_preds'].shape}")  # (B, T, per_latent_motion_len, codebook_size)
    
    # 4. 推理模式：生成 latent motion IDs
    print("\n=== Inference Mode ===")
    predictor.eval()
    with torch.no_grad():
        inference_results = predictor(
            perceptual_features=perceptual_features,
            attention_mask=attention_mask,
            train=False,
            temperature=1.0,
            top_k=50
        )
    
    print(f"Generated motion IDs shape: {inference_results['latent_motion_id_preds'].shape}")  # (B, T, per_latent_motion_len)
    
    # 5. 获取 latent motion 嵌入用于下游任务
    print("\n=== Getting Embeddings ===")
    generated_ids = inference_results['latent_motion_id_preds']
    motion_embeddings = predictor.get_latent_motion_embeddings(generated_ids)
    print(f"Motion embeddings shape: {motion_embeddings.shape}")  # (B, T, hidden_size)
    
    return predictor, train_results, inference_results

def example_integration_with_mdt():
    """示例：如何将简化预测器集成到 MDT 模型中"""
    
    # 模拟 MDT 中的感知嵌入
    class MockPerceptualEmbedding:
        def __init__(self):
            self.static_dim = 512
            self.gripper_dim = 512
    
    # 创建预测器
    perceptual_embed_dim = 512 * 2  # static + gripper
    predictor = SimplifiedLatentMotionPredictor(
        input_dim=perceptual_embed_dim,
        hidden_size=512,
        per_latent_motion_len=10,
        latent_motion_codebook_size=1024,
    )
    
    # 模拟来自 MDT 的感知嵌入
    batch_size, seq_len = 2, 5
    static_features = torch.randn(batch_size, seq_len, 512)
    gripper_features = torch.randn(batch_size, seq_len, 512)
    
    # 合并特征
    combined_features = torch.cat([static_features, gripper_features], dim=-1)
    
    # 训练时使用 ground truth
    gt_motion_ids = torch.randint(0, 1024, (batch_size, seq_len, 10))
    
    # 前向传播
    results = predictor(
        perceptual_features=combined_features,
        latent_motion_ids=gt_motion_ids,
        train=True
    )
    
    print(f"Integration example - Loss: {results['loss']:.4f}")
    
    # 获取条件嵌入用于 diffusion 模型
    predicted_ids = torch.argmax(results['latent_motion_preds'], dim=-1)
    condition_embeddings = predictor.get_latent_motion_embeddings(predicted_ids)
    condition_embeddings = condition_embeddings.mean(dim=1)  # 时间维度平均
    
    print(f"Condition embeddings for diffusion: {condition_embeddings.shape}")  # (B, hidden_size)
    
    return condition_embeddings

def example_from_motogpt_config():
    """示例：从 MotoGPT 配置创建预测器"""
    
    # 模拟 MotoGPT 的配置
    motogpt_config = {
        'hidden_size': 768,
        'per_latent_motion_len': 12,
        'latent_motion_codebook_size': 2048,
        'use_latent_motion_pos_embedding': True,
        'mask_latent_motion_probability': 0.1,
    }
    
    # 创建预测器
    predictor = create_simplified_predictor_from_motogpt_config(
        motogpt_config=motogpt_config,
        input_dim=1536  # 你的输入特征维度
    )
    
    print(f"Created predictor with config: {motogpt_config}")
    print(f"Predictor hidden size: {predictor.hidden_size}")
    print(f"Codebook size: {predictor.latent_motion_codebook_size}")
    
    return predictor

if __name__ == "__main__":
    print("Running simplified latent motion predictor examples...")
    
    # 基本使用示例
    predictor, train_results, inference_results = example_usage()
    
    # 与 MDT 集成示例
    condition_embeddings = example_integration_with_mdt()
    
    # 从 MotoGPT 配置创建示例
    config_predictor = example_from_motogpt_config()
    
    print("\nAll examples completed successfully!")
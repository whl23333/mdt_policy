import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import argparse
import json
from torch.utils.data import DataLoader
import omegaconf
import hydra
from functools import partial
from transformers import AutoTokenizer
from common.models.model_utils import load_model
from common.processors.preprocessor_utils import get_rgb_preprocessor
from latent_motion_tokenizer.src.trainers.latent_motion_tokenizer_trainer import LatentMotionTokenizer_Trainer, LatentMotionTokenizer_Trainer_Multiview
from torch.utils.data import DataLoader
from functools import partial
from common.data.data_utils import load_dataset
import wandb

def main(cfg):
    # Prepare Latent Motion Tokenizer
    wandb.init(
        project="Moto_training",  # 替换为你的项目名称
        name="3dlatent",         # 替换为你的运行名称（可选）
    )
    if cfg['paired_loss']:
        if cfg['training_config']['paired_method'] == "3d":
            latent_motion_tokenizer_config_path = cfg['latent_motion_tokenizer_3d_config_path']
        else:
            latent_motion_tokenizer_config_path = cfg['paired_latent_motion_tokenizer_config_path']
    else:
        latent_motion_tokenizer_config_path = cfg['not_paired_latent_motion_tokenizer_config_path']
    print(f"initializing Latent Motion Tokenizer from {latent_motion_tokenizer_config_path} ...")
    latent_motion_tokenizer_config = omegaconf.OmegaConf.load(latent_motion_tokenizer_config_path)
    latent_motion_tokenizer = hydra.utils.instantiate(latent_motion_tokenizer_config)
    latent_motion_tokenizer.config = latent_motion_tokenizer_config

    # Prepare rgb_processor
    rgb_preprocessor = get_rgb_preprocessor(**cfg['rgb_preprocessor_config'])

    # Preprepare Dataloaders
    dataset_config_path = cfg['dataset_config_path']
    extra_data_config = {
        'sequence_length': 1,
        'do_extract_future_frames': True,
        'do_extract_action': False
    }
    train_dataset, eval_dataset = load_dataset(dataset_config_path, extra_data_config)
    dataloader_cls = partial(
        DataLoader, 
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        persistent_workers=True,
        num_workers=cfg['dataloader_config']['workers_per_gpu'],
        batch_size=cfg['dataloader_config']['bs_per_gpu'],
        prefetch_factor= cfg['dataloader_config']['prefetch_factor']
    )
    train_dataloader = dataloader_cls(train_dataset)
    eval_dataloader = dataloader_cls(eval_dataset)
    
    # Prepare Trainer
    trainer = LatentMotionTokenizer_Trainer_Multiview(
        latent_motion_tokenizer=latent_motion_tokenizer,
        rgb_preprocessor=rgb_preprocessor,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        bs_per_gpu=cfg['dataloader_config']['bs_per_gpu'],
        paired_loss=cfg['paired_loss'],
        **cfg['training_config']
    )

    # Start Training
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="/data/250010208/whl/code/Moto/latent_motion_tokenizer/configs/train/data_calvin_3d.yaml")
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config_path)
    main(cfg)

    



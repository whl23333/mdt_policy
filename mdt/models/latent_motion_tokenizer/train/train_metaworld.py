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
from latent_motion_tokenizer.src.trainers.latent_motion_tokenizer_trainer import LatentMotionTokenizer_Trainer, LatentMotionTokenizer_Trainer_Metaworld
from torch.utils.data import DataLoader
from functools import partial
from common.data.data_utils import load_dataset
from common.data.datasets import MetaworldMultiView
from torch.utils.data import random_split
import torch

def main(cfg):
    # Prepare Latent Motion Tokenizer
    latent_motion_tokenizer_config_path = cfg['latent_motion_tokenizer_config_path']
    print(f"initializing Latent Motion Tokenizer from {latent_motion_tokenizer_config_path} ...")
    latent_motion_tokenizer_config = omegaconf.OmegaConf.load(latent_motion_tokenizer_config_path)
    latent_motion_tokenizer = hydra.utils.instantiate(latent_motion_tokenizer_config)
    latent_motion_tokenizer.config = latent_motion_tokenizer_config

    # Prepare rgb_processor
    rgb_preprocessor = get_rgb_preprocessor(**cfg['rgb_preprocessor_config'])

    # Preprepare Dataloaders
    
    
    # dataset_config_path = cfg['dataset_config_path']
    # extra_data_config = {
    #     'sequence_length': 1,
    #     'do_extract_future_frames': True,
    #     'do_extract_action': False
    # }
    # train_dataset, eval_dataset = load_dataset(dataset_config_path, extra_data_config)
    # dataloader_cls = partial(
    #     DataLoader, 
    #     pin_memory=True, # Accelerate data reading
    #     shuffle=True,
    #     persistent_workers=True,
    #     num_workers=cfg['dataloader_config']['workers_per_gpu'],
    #     batch_size=cfg['dataloader_config']['bs_per_gpu'],
    #     prefetch_factor= cfg['dataloader_config']['prefetch_factor']
    # )
    # train_dataloader = dataloader_cls(train_dataset)
    # eval_dataloader = dataloader_cls(eval_dataset)
    
    metaworld_dataset = MetaworldMultiView(
        path="/home/yyang-infobai/metaworld2",
        sample_per_seq=8,
        frameskip=1,
        use_video=False
    )
    
    train_ratio=0.8
    train_size = int(train_ratio * len(metaworld_dataset))
    eval_size = len(metaworld_dataset) - train_size
    train_dataset, eval_dataset = random_split(metaworld_dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42))
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Prepare Trainer
    trainer = LatentMotionTokenizer_Trainer_Metaworld(
        latent_motion_tokenizer=latent_motion_tokenizer,
        rgb_preprocessor=rgb_preprocessor,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        bs_per_gpu=cfg['dataloader_config']['bs_per_gpu'],
        save_path = "/home/yyang-infobai/Moto/latent_motion_tokenizer/outputs/latent_motion_tokenizer_trained_on_metaworld/paired_N2",
        save_steps = 10000,
        print_steps = 100,
        max_steps=100000,
        num_warmup_steps=5000,
        eval_steps=1000,
        resume_ckpt_path = None,
        paired_loss = True,
    )

    # Start Training
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="/home/yyang-infobai/Moto/latent_motion_tokenizer/configs/train/data_calvin-vq_size128_dim32_num8_legacyTrue-vision_MaeLarge-decoder_queryFusionModeAdd_Patch196_useMaskFalse-mformer_legacyTrue-train_lr0.0001_bs256-aug_shiftTrue_resizedCropFalse.yaml")
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config_path)
    main(cfg)

    



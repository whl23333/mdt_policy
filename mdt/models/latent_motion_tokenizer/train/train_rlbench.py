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
from latent_motion_tokenizer.src.trainers.latent_motion_tokenizer_trainer import LatentMotionTokenizer_Trainer, LatentMotionTokenizer_Trainer_RLBench
from torch.utils.data import DataLoader
from functools import partial
from common.data.data_utils import load_dataset, load_instructions, traj_collate_fn
from common.data.datasets import RLBenchDataset_Moto
from torch.utils.data import random_split
import torch
import os

def main(cfg):
    # Prepare Latent Motion Tokenizer
    if cfg['paired_loss']:
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
    rlbench_dataset_config = omegaconf.OmegaConf.load(dataset_config_path)
    # extra_data_config = {
    #     'sequence_length': 1,
    #     'do_extract_future_frames': True,
    #     'do_extract_action': False
    # }
    instruction = load_instructions(
        os.path.join("/group/ycyang/yyang-infobai/instructions/peract/", "instructions.pkl"),
        tasks=("place_cups", "close_jar", "insert_onto_square_peg", "light_bulb_in", "meat_off_grill", "open_drawer", "place_shape_in_shape_sorter", "place_wine_at_rack_location", "push_buttons", "put_groceries_in_cupboard", "put_item_in_drawer", "put_money_in_safe", "reach_and_drag", "slide_block_to_color_target", "stack_blocks", "stack_cups", "sweep_to_dustpan_of_size", "turn_tap"),
        variations=tuple(range(200))
    )
    if instruction is None:
        raise NotImplementedError()
    else:
        taskvar = [
            (task, var)
            for task, var_instr in instruction.items()
            for var in var_instr.keys()
        ]
    image_rescale = "0.75,1.25"
    rlbench_dataset = RLBenchDataset_Moto(
        root="/group/ycyang/yyang-infobai/rlbench_train/Peract_packaged/train",
        instructions=instruction,
        taskvar=taskvar,
        cache_size=600,
        max_episodes_per_task=-1,
        num_iters=600000,
        cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        training=True,
        image_rescale=tuple(
            float(x) for x in image_rescale.split(",")
        ),
        return_low_lvl_trajectory=True,
        dense_interpolation=bool(1),
        interpolation_length=2,
    )
    train_ratio=0.8
    train_size = int(train_ratio * len(rlbench_dataset))
    eval_size = len(rlbench_dataset) - train_size
    train_dataset, eval_dataset = random_split(rlbench_dataset, [train_size, eval_size])
    
    g = torch.Generator()
    g.manual_seed(0)
    
    train_dataloader = DataLoader(
        train_dataset,
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        num_workers=cfg['dataloader_config']['workers_per_gpu'],
        batch_size=cfg['dataloader_config']['bs_per_gpu'],
        prefetch_factor= cfg['dataloader_config']['prefetch_factor'],
        collate_fn=traj_collate_fn,
        generator=g
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        pin_memory=True, # Accelerate data reading
        shuffle=False,
        num_workers=cfg['dataloader_config']['workers_per_gpu'],
        batch_size=cfg['dataloader_config']['bs_per_gpu'],
        prefetch_factor= cfg['dataloader_config']['prefetch_factor'],
        collate_fn=traj_collate_fn,
        generator=g
    )
    # Prepare Trainer
    trainer = LatentMotionTokenizer_Trainer_RLBench(
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
    parser.add_argument('--config_path', type=str, default="/home/yyang-infobai/Moto/latent_motion_tokenizer/configs/train/data_rlbench.yaml")
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config_path)
    main(cfg)

    



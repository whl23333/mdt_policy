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

def cycle(dl):
    while True:
        for data in dl:
            yield data

metaworld_dataset = MetaworldMultiView(
    path="/home/yyang-infobai/metaworld2",
    sample_per_seq=8,
    frameskip=1,
    use_video=False
)
metaworld_dataloader = DataLoader(metaworld_dataset, batch_size=1, shuffle=False, num_workers=0)
latent_motion_config_path = "/home/yyang-infobai/Moto/latent_motion_tokenizer/outputs/ckpts/paired/temp_step_50000/config.yaml"
latent_motion_tokenizer_config = omegaconf.OmegaConf.load(latent_motion_config_path)
latent_motion_tokenizer = hydra.utils.instantiate(latent_motion_tokenizer_config)
latent_motion_tokenizer.config = latent_motion_tokenizer_config
ckpt_path = "/home/yyang-infobai/Moto/latent_motion_tokenizer/outputs/ckpts/paired/temp_step_50000/pytorch_model.bin"
missing_keys, unexpected_keys = latent_motion_tokenizer.load_state_dict(torch.load(ckpt_path), strict=False)
missing_root_keys = set([k.split(".")[0] for k in missing_keys])
print('load ', ckpt_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys)
cfg = omegaconf.OmegaConf.load("/home/yyang-infobai/Moto/latent_motion_tokenizer/configs/train/data_calvin-vq_size128_dim32_num8_legacyTrue-vision_MaeLarge-decoder_queryFusionModeAdd_Patch196_useMaskFalse-mformer_legacyTrue-train_lr0.0001_bs256-aug_shiftTrue_resizedCropFalse.yaml")
rgb_preprocessor = get_rgb_preprocessor(**cfg['rgb_preprocessor_config'])
metaworld_dataloader = cycle(metaworld_dataloader)
for i in range(100):
    x_target, x_cond, task, actions = next(metaworld_dataloader)
    print(f"task: {task}")
    batch_list = [{'rgb_initial':x_cond[:, i], 'rgb_future':x_target[:, i]} for i in range(x_target.shape[1])]
    orig_rgb_seq_list = [torch.stack([batch['rgb_initial'], batch['rgb_future']], dim=1) for batch in batch_list]
    rgb_seq_list = [rgb_preprocessor(orig_rgb_seq, train=True) for orig_rgb_seq in orig_rgb_seq_list]
    latent_motion_tokenizer.eval()
    N = len(rgb_seq_list)
    for i in range(N):
        output = latent_motion_tokenizer(
            cond_pixel_values1 = rgb_seq_list[i][:, 0],
            target_pixel_values1 = rgb_seq_list[i][:, 1],
            cond_pixel_values2 = rgb_seq_list[i][:, 0],
            target_pixel_values2 = rgb_seq_list[i][:, 1],
            return_recons_only = True
        )
        indices=output["indices"]
        print(f"corner{i}: ", indices)
    
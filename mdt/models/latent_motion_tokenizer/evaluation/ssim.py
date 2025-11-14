from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch
from torch import Tensor
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
import os
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from common.data.datasets import DataPrefetcher
import numpy as np
import torch.nn.functional as F

class SSIM_EVAL:
    def __init__(
        self,
        latent_motion_tokenizer,
        rgb_preprocessor,
        eval_dataloader,
        bs_per_gpu=32,
        paired_loss=False,
        eval_step=200,
        resume_ckpt_path=None,
        **kwargs
    ):
        self.rgb_preprocessor = rgb_preprocessor
        self.bs_per_gpu = bs_per_gpu
        self.paired_loss = paired_loss
        self.eval_step = eval_step
        
        assert resume_ckpt_path is not None, "Please provide a checkpoint path to resume from."
        print(f"loading checkpoint from {resume_ckpt_path} ...")
        missing_keys, unexpected_keys = latent_motion_tokenizer.load_state_dict(torch.load(os.path.join(resume_ckpt_path, 'pytorch_model.bin'), map_location='cpu'), strict=False)
        missing_root_keys = set([k.split(".")[0] for k in missing_keys])
        print('load ', resume_ckpt_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
        )
        latent_motion_tokenizer, eval_dataloader = accelerator.prepare(latent_motion_tokenizer, eval_dataloader)
        self.accelerator = accelerator
        self.latent_motion_tokenizer = latent_motion_tokenizer
        self.eval_prefetcher = DataPrefetcher(eval_dataloader, self.accelerator.device)
        self.rgb_preprocessor = rgb_preprocessor.to(self.accelerator.device)
    
    def is_main(self):
        return self.accelerator.is_main_process
    
    def evaluate(self):
        eval_step = self.eval_step
        eval_static_cond = []
        eval_static_target = []
        eval_gripper_cond = []
        eval_gripper_target = []
        eval_diff1_cond = []
        eval_diff2_cond = []
        eval_diff1_target = []
        eval_diff2_target = []
        eval_exact_match = []
        eval_cosine_sim = []
        self.latent_motion_tokenizer.eval()
        for step in range(eval_step):
            with torch.no_grad():
                batch, load_time = self.eval_prefetcher.next_without_none()
                orig_rgb_seq_static = torch.cat([batch['rgb_initial_static'], batch['rgb_future_static']], dim=1) # (b, 2, c, h, w)
                rgb_seq_static = self.rgb_preprocessor(orig_rgb_seq_static, train=True)
                orig_rgb_seq_gripper = torch.cat([batch['rgb_initial_gripper'], batch['rgb_future_gripper']], dim=1) # (b, 2, c, h, w)
                rgb_seq_gripper = self.rgb_preprocessor(orig_rgb_seq_gripper, train=True)
                
                outputs_static = self.latent_motion_tokenizer(
                    cond_pixel_values1=rgb_seq_static[:, 0],
                    target_pixel_values1=rgb_seq_static[:, 1],
                    cond_pixel_values2=rgb_seq_static[:, 0],
                    target_pixel_values2=rgb_seq_static[:, 1],
                    return_recons_only=True,
                    corner=['static']
                )
                outputs_gripper = self.latent_motion_tokenizer(
                    cond_pixel_values1=rgb_seq_gripper[:, 0],
                    target_pixel_values1=rgb_seq_gripper[:, 1],
                    cond_pixel_values2=rgb_seq_gripper[:, 0],
                    target_pixel_values2=rgb_seq_gripper[:, 1],
                    return_recons_only=True,
                    corner=['gripper']
                )
                outputs_diff1 = self.latent_motion_tokenizer(
                    cond_pixel_values1=rgb_seq_static[:, 0],
                    target_pixel_values1=rgb_seq_static[:, 1],
                    cond_pixel_values2=rgb_seq_gripper[:, 0],
                    target_pixel_values2=rgb_seq_gripper[:, 1],
                    return_recons_only=True,
                    corner=['gripper']
                )
                outputs_diff2 = self.latent_motion_tokenizer(
                    cond_pixel_values1=rgb_seq_gripper[:, 0],
                    target_pixel_values1=rgb_seq_gripper[:, 1],
                    cond_pixel_values2=rgb_seq_static[:, 0],
                    target_pixel_values2=rgb_seq_static[:, 1],
                    return_recons_only=True,
                    corner=['static']
                )
                    
                    
                recons_rgb_future_static = self.rgb_preprocessor.post_process(outputs_static["recons_pixel_values"])
                recons_rgb_future_gripper = self.rgb_preprocessor.post_process(outputs_gripper["recons_pixel_values"])
                recons_rgb_future_diff1 = self.rgb_preprocessor.post_process(outputs_diff1["recons_pixel_values"])
                recons_rgb_future_diff2 = self.rgb_preprocessor.post_process(outputs_diff2["recons_pixel_values"])
                gt_latent_motion_ids_static = outputs_static["indices"]
                gt_latent_motion_ids_gripper = outputs_gripper["indices"]
                exact_match = (gt_latent_motion_ids_static == gt_latent_motion_ids_gripper).float().mean()
                cosine_sim = F.cosine_similarity(gt_latent_motion_ids_static.float(), gt_latent_motion_ids_gripper.float(), dim=1).mean()
                orig_rgb_seq_static = self.rgb_preprocessor.post_process(rgb_seq_static)
                orig_rgb_seq_gripper = self.rgb_preprocessor.post_process(rgb_seq_gripper)
                
                ssim_static_cond = ssim(
                    recons_rgb_future_static,
                    orig_rgb_seq_static[:, 0],
                    data_range=1.0,
                    size_average=True
                )
                ssim_static_target = ssim(
                    recons_rgb_future_static,
                    orig_rgb_seq_static[:, 1],
                    data_range=1.0,
                    size_average=True
                )
                ssim_gripper_cond = ssim(
                    recons_rgb_future_gripper,
                    orig_rgb_seq_gripper[:, 0],
                    data_range=1.0,
                    size_average=True
                )
                ssim_gripper_target = ssim(
                    recons_rgb_future_gripper,
                    orig_rgb_seq_gripper[:, 1],
                    data_range=1.0,
                    size_average=True
                )
                ssim_diff1_cond = ssim(
                    recons_rgb_future_diff1,
                    orig_rgb_seq_gripper[:, 0],
                    data_range=1.0,
                    size_average=True
                )
                ssim_diff2_cond = ssim(
                    recons_rgb_future_diff2,
                    orig_rgb_seq_static[:, 0],
                    data_range=1.0,
                    size_average=True
                )
                ssim_diff1_target = ssim(
                    recons_rgb_future_diff1,
                    orig_rgb_seq_gripper[:, 1],
                    data_range=1.0,
                    size_average=True
                )
                ssim_diff2_target = ssim(
                    recons_rgb_future_diff2,
                    orig_rgb_seq_static[:, 1],
                    data_range=1.0,
                    size_average=True
                )
                eval_diff1_cond.append(ssim_diff1_cond)
                eval_diff2_cond.append(ssim_diff2_cond)
                eval_diff1_target.append(ssim_diff1_target)
                eval_diff2_target.append(ssim_diff2_target)
                eval_static_cond.append(ssim_static_cond)
                eval_static_target.append(ssim_static_target)
                eval_gripper_cond.append(ssim_gripper_cond)
                eval_gripper_target.append(ssim_gripper_target)
                eval_exact_match.append(exact_match)
                eval_cosine_sim.append(cosine_sim)
                print(f"Step {step}: ssim_static_cond: {ssim_static_cond}, ssim_static_target: {ssim_static_target}, ssim_gripper_cond: {ssim_gripper_cond}, ssim_gripper_target: {ssim_gripper_target},\
                      ssim_diff1_cond: {ssim_diff1_cond}, ssim_diff1_target: {ssim_diff1_target}, ssim_diff2_cond: {ssim_diff2_cond}, ssim_diff2_target: {ssim_diff2_target},\
                          exact_match: {exact_match}, cosine_sim: {cosine_sim}")
                
        eval_static_cond_mean = np.mean([x.cpu().numpy() for x in eval_static_cond])
        eval_static_target_mean = np.mean([x.cpu().numpy() for x in eval_static_target])
        eval_gripper_cond_mean = np.mean([x.cpu().numpy() for x in eval_gripper_cond])
        eval_gripper_target_mean = np.mean([x.cpu().numpy() for x in eval_gripper_target])
        eval_exact_match_mean = np.mean([x.cpu().numpy() for x in eval_exact_match])
        eval_cosine_sim_mean = np.mean([x.cpu().numpy() for x in eval_cosine_sim])
        eval_diff1_cond_mean = np.mean([x.cpu().numpy() for x in eval_diff1_cond])
        eval_diff2_cond_mean = np.mean([x.cpu().numpy() for x in eval_diff2_cond])
        eval_diff1_target_mean = np.mean([x.cpu().numpy() for x in eval_diff1_target])
        eval_diff2_target_mean = np.mean([x.cpu().numpy() for x in eval_diff2_target])
        return eval_static_cond_mean, eval_static_target_mean, eval_gripper_cond_mean, eval_gripper_target_mean, eval_diff1_cond_mean, eval_diff1_target_mean, eval_diff2_cond_mean, eval_diff2_target_mean, eval_exact_match_mean, eval_cosine_sim_mean
                

                
            

            
        

        

def main(cfg):
    # Prepare Latent Motion Tokenizer
    latent_motion_tokenizer_config_path = cfg['paired_latent_motion_tokenizer_config_path']
    print(f"initializing Latent Motion Tokenizer from {latent_motion_tokenizer_config_path} ...")
    latent_motion_tokenizer_config = omegaconf.OmegaConf.load(latent_motion_tokenizer_config_path)
    latent_motion_tokenizer = hydra.utils.instantiate(latent_motion_tokenizer_config)
    latent_motion_tokenizer.config = latent_motion_tokenizer_config
        
    # Prepare data loader
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
    
    # Prepare rgb_processor
    rgb_preprocessor = get_rgb_preprocessor(**cfg['rgb_preprocessor_config'])
    
    # evaluate
    eval_step = 3000
    
    ssim = SSIM_EVAL(
        latent_motion_tokenizer=latent_motion_tokenizer,
        rgb_preprocessor=rgb_preprocessor,
        eval_dataloader=eval_dataloader,
        bs_per_gpu=cfg['dataloader_config']['bs_per_gpu'],
        paired_loss=cfg['paired_loss'],
        eval_step=eval_step,
        **cfg['training_config']
    )
    
    ssim_static_cond, ssim_static_target, ssim_gripper_cond, ssim_gripper_target, ssim_diff1_cond, ssim_diff1_target, ssim_diff2_cond, ssim_diff2_target, exact_match, cosine_sim = ssim.evaluate()
    output_file = f"{cfg['training_config']['save_path']}/ssim.txt"
    with open(output_file, "w") as f:
        f.write(f"ssim_static_cond: {ssim_static_cond}\n")
        f.write(f"ssim_static_target: {ssim_static_target}\n")
        f.write(f"ssim_gripper_cond: {ssim_gripper_cond}\n")
        f.write(f"ssim_gripper_target: {ssim_gripper_target}\n")
        f.write(f"ssim_diff1_cond: {ssim_diff1_cond}\n")
        f.write(f"ssim_diff1_target: {ssim_diff1_target}\n")
        f.write(f"ssim_diff2_cond: {ssim_diff2_cond}\n")
        f.write(f"ssim_diff2_target: {ssim_diff2_target}\n")
        f.write(f"eval_exact_match: {exact_match}\n")
        f.write(f"eval_cosine_sim: {cosine_sim}\n")
    print(f"Results saved to {output_file}")
    print(f"ssim_static_cond: {ssim_static_cond}")
    print(f"ssim_static_target: {ssim_static_target}")
    print(f"ssim_gripper_cond: {ssim_gripper_cond}")
    print(f"ssim_gripper_target: {ssim_gripper_target}")
    print(f"ssim_diff1_cond: {ssim_diff1_cond}")
    print(f"ssim_diff1_target: {ssim_diff1_target}")
    print(f"ssim_diff2_cond: {ssim_diff2_cond}")
    print(f"ssim_diff2_target: {ssim_diff2_target}")    
    print(f"eval_exact_match: {exact_match}")
    print(f"eval_cosine_sim: {cosine_sim}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/home/yyang-infobai/Moto/latent_motion_tokenizer/configs/train/data_calvin.yaml')
    args=parser.parse_args()
    cfg=omegaconf.OmegaConf.load(args.config_path)
    main(cfg)
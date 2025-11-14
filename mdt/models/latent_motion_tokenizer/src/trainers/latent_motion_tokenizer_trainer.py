import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import os
from time import time
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
import torch
from common.data.datasets import DataPrefetcher
from latent_motion_tokenizer.src.trainers.trainer_utils import visualize_latent_motion_reconstruction
import omegaconf
from glob import glob
import shutil
from collections import defaultdict
from latent_motion_tokenizer.src.trainers.optimizer import get_optimizer, LinearWarmup_CosineAnnealing
from contextlib import contextmanager
import wandb

def cycle(dl):
    while True:
        for data in dl:
            yield data
class LatentMotionTokenizer_Trainer:
    def __init__(
        self,
        latent_motion_tokenizer,
        rgb_preprocessor,
        train_dataloader,
        eval_dataloader,
        save_path,
        save_epochs=1,
        save_steps=10000,
        num_epochs=20,
        print_steps=100,
        lr_max=0.0001,
        weight_decay=0.,
        num_warmup_epochs=1,
        gradient_accumulation_steps=4,
        resume_ckpt_path=None,
        bs_per_gpu=32,
        max_epoch=None,
    ):
        if resume_ckpt_path is not None:
            print(f"resuming Latent Motion Tokenizer from {resume_ckpt_path} ...")
            missing_keys, unexpected_keys = latent_motion_tokenizer.load_state_dict(torch.load(os.path.join(resume_ckpt_path, 'pytorch_model.bin'), map_location='cpu'), strict=False)
            missing_root_keys = set([k.split(".")[0] for k in missing_keys])
            print('load ', resume_ckpt_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys)
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        accelerator= Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs]
        )
        self.gradient_accumulation_steps = gradient_accumulation_steps

        total_prints_per_epoch = len(train_dataloader.dataset) // (print_steps * bs_per_gpu * accelerator.num_processes)

        optimizer = get_optimizer(
                        [p for n, p in latent_motion_tokenizer.named_parameters() if p.requires_grad], 
                        lr=lr_max, 
                        wd=weight_decay
                    )
        
        linear_warmup_total_iters = min(num_warmup_epochs*total_prints_per_epoch, 5000000 // (print_steps * bs_per_gpu * accelerator.num_processes))
        scheduler = LinearWarmup_CosineAnnealing(
                        optimizer=optimizer,
                        linear_warmup_start_factor=0.5,
                        linear_warmup_total_iters=linear_warmup_total_iters,
                        cosine_annealing_T_max=num_epochs*total_prints_per_epoch-linear_warmup_total_iters,
                        cosine_annealing_eta_min=5e-5
                    )

        latent_motion_tokenizer, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            latent_motion_tokenizer, optimizer, train_dataloader, eval_dataloader, 
            device_placement=[True, True, False, False]
        )
        
        self.writer = SummaryWriter(os.path.join(save_path, 'logs'))
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_prints_per_epoch = total_prints_per_epoch
        self.latent_motion_tokenizer = latent_motion_tokenizer
        self.optimizer = optimizer
        self.train_prefetcher = DataPrefetcher(train_dataloader, self.device)
        self.eval_prefetcher = DataPrefetcher(eval_dataloader, self.device)
        self.rgb_preprocessor = rgb_preprocessor.to(self.device)
        self.save_path = save_path
        self.save_epochs = save_epochs
        self.save_steps = save_steps
        self.max_epoch = max_epoch
        self.num_epochs = num_epochs
        self.print_steps = print_steps
        self.bs_per_gpu = bs_per_gpu


    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def process_index(self):
        return self.accelerator.process_index

    def print(self, msg):
        self.accelerator.print(msg)

    def save_checkpoint(self, save_dir):
        unwrapped_latent_motion_tokenizer = self.accelerator.unwrap_model(self.latent_motion_tokenizer)
        state_dict = unwrapped_latent_motion_tokenizer.get_state_dict_to_save()
        
        torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
        omegaconf.OmegaConf.save(unwrapped_latent_motion_tokenizer.config, os.path.join(save_dir, "config.yaml"))

        self.print(f"A new model checkpoint is saved to {save_dir}!!!")
        
    def train(self):
        eval_loss_steps = len(self.train_prefetcher) // len(self.eval_prefetcher)
        step = 0
        
        for epoch in range(self.num_epochs+1):
            if epoch != 0:
                self.accelerator.wait_for_everyone()
                save_dir = os.path.join(self.save_path, f'saved_epoch_{epoch}_step_{step}')

                if self.is_main:
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_checkpoint(save_dir)
                    
                visualization_dir = os.path.join(save_dir, 'visualization')
                self.eval_latent_motion_reconstruction(visualization_dir)

                if epoch == self.num_epochs:
                    break
                if (self.max_epoch is not None) and (epoch >= self.max_epoch):
                    break

            log_loss = {}
            eval_log_loss = {}
            
            cum_load_time = 0 
            clock = time()
            batch_idx = 0
            batch, load_time = self.train_prefetcher.next()
            
            while batch is not None:
                with self.accelerator.accumulate(self.latent_motion_tokenizer):

                    self.latent_motion_tokenizer.train()
                    self.optimizer.zero_grad()
                    loss = self.calculate_loss(batch, train=True)
                    self.accelerator.backward(loss['loss'])
                    self.optimizer.step()

                    for key in loss:
                        if key not in log_loss:
                            log_loss[key] = 0.0
                        log_loss[key] += loss[key].detach() / self.print_steps

                    cum_load_time += load_time / self.print_steps

                if (batch_idx+1) % self.print_steps == 0:

                    with torch.no_grad():
                        batch, _ = self.eval_prefetcher.next_without_none()
                        self.latent_motion_tokenizer.eval()
                        loss = self.calculate_loss(batch, train=True)

                        for key in loss:
                            eval_log_loss[key] = loss[key].detach()

                    self.log(log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step)
                    log_loss = {}
                    eval_log_loss = {}

                    cum_load_time = 0
                    clock = time()
                    self.scheduler.step()

                if step % self.save_steps == 0:
                    self.accelerator.wait_for_everyone()
                    save_dir = os.path.join(self.save_path, f'temp_epoch_{epoch}_step_{step}')
                    if self.is_main:
                        existing_ckpt_dirs = glob(os.path.join(self.save_path, f'temp_epoch_*_step_*'))
                        for existing_ckpt_dir in existing_ckpt_dirs:
                            if existing_ckpt_dir != save_dir:
                                shutil.rmtree(existing_ckpt_dir)
                        os.makedirs(save_dir, exist_ok=True)
                        self.save_checkpoint(save_dir)

                    visualization_dir = os.path.join(save_dir, 'visualization')
                    self.eval_latent_motion_reconstruction(visualization_dir)

                batch_idx += 1
                step += 1
                batch, load_time = self.train_prefetcher.next()



    @torch.no_grad()
    def eval_latent_motion_reconstruction(self, visualization_dir):
        os.makedirs(visualization_dir, exist_ok=True)
        self.print(f"Saving visualization results to {visualization_dir} ...")
        batch, _ = self.eval_prefetcher.next_without_none()

        orig_rgb_seq = torch.cat([batch['rgb_initial'], batch['rgb_future']], dim=1) # (b, 2, c, h, w)
        rgb_seq = self.rgb_preprocessor(orig_rgb_seq, train=True)

        self.latent_motion_tokenizer.eval()
        outputs = self.latent_motion_tokenizer(
            cond_pixel_values=rgb_seq[:,0],
            target_pixel_values=rgb_seq[:,1],
            return_recons_only=True
        )
            
        recons_rgb_future = self.rgb_preprocessor.post_process(outputs["recons_pixel_values"]).detach().cpu()  # (b, c, h, w)
        gt_latent_motion_ids = outputs["indices"].detach().cpu() # (b, per_latent_motion_len)
        # orig_rgb_seq = orig_rgb_seq.detach().cpu()
        orig_rgb_seq = self.rgb_preprocessor.post_process(rgb_seq).detach().cpu()

        for i in range(orig_rgb_seq.shape[0]):
            visualize_latent_motion_reconstruction(
                initial_frame=orig_rgb_seq[i,0],
                next_frame=orig_rgb_seq[i,1],
                recons_next_frame=recons_rgb_future[i],
                latent_motion_ids=gt_latent_motion_ids[i],
                path=os.path.join(visualization_dir, f"{self.process_index}-{i}.png")
            )


    def calculate_loss(self, batch, train):
        # image preprocessing
        rgb_seq = torch.cat([batch['rgb_initial'], batch['rgb_future']], dim=1)
        rgb_seq = self.rgb_preprocessor(rgb_seq, train=train)

        # compute loss
        loss = self.latent_motion_tokenizer(
            cond_pixel_values=rgb_seq[:,0],
            target_pixel_values=rgb_seq[:,1]
        )

        return loss


    def log(self, log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step):
        for key in log_loss:
            log_loss[key] = self.accelerator.gather_for_metrics(log_loss[key]).mean()
        for key in eval_log_loss:
            eval_log_loss[key] = self.accelerator.gather_for_metrics(eval_log_loss[key]).mean()
        load_pecnt = torch.tensor(cum_load_time / (time()-clock)).to(self.device)
        load_pecnt = self.accelerator.gather_for_metrics(load_pecnt).mean()
        fps = (self.bs_per_gpu*self.print_steps*2) / (time()-clock)
        fps = self.accelerator.gather_for_metrics(torch.tensor(fps).to(self.device)).sum()

        text = 'Train Epoch: {} [{}/{} ({:.0f}%)] FPS:{:.5f} Load Pertentage:{:.5f} LR:{}'.format(
            epoch, 
            batch_idx * self.bs_per_gpu * self.accelerator.num_processes, 
            len(self.train_prefetcher), 
            100. * batch_idx * self.bs_per_gpu * self.accelerator.num_processes / len(self.train_prefetcher),
            fps,
            load_pecnt,
            self.scheduler.get_last_lr()[0],
        )
        for key in log_loss:
            text = text + ' {}: {:.5f}'.format(key, log_loss[key])
        for key in eval_log_loss:
            text = text + ' eval_{}: {:.5f}'.format(key, eval_log_loss[key])
        self.print(text)
        if self.is_main:
            for key in log_loss:
                self.writer.add_scalar(key, log_loss[key], step)
            for key in eval_log_loss:
                self.writer.add_scalar('eval_'+key, eval_log_loss[key], step)
            self.writer.add_scalar("learning rate", self.scheduler.get_last_lr()[0], step)
            self.writer.add_scalar("FPS", fps, step)
            self.writer.add_scalar("loading time in total time", load_pecnt, step)

class LatentMotionTokenizer_Trainer_Metaworld:
    def __init__(
        self,
        latent_motion_tokenizer,
        rgb_preprocessor,
        train_dataloader,
        eval_dataloader,
        save_path,
        # save_epochs=1,
        save_steps=10000,
        # num_epochs=20,
        print_steps=100,
        lr_max=0.0001,
        weight_decay=0.,
        gradient_accumulation_steps=4,
        resume_ckpt_path=None,
        bs_per_gpu=32,
        max_steps=50000,
        num_warmup_steps=5000,
        eval_steps=1000,
        paired_loss=True
    ):
        if resume_ckpt_path is not None:
            print(f"resuming Latent Motion Tokenizer from {resume_ckpt_path} ...")
            missing_keys, unexpected_keys = latent_motion_tokenizer.load_state_dict(torch.load(os.path.join(resume_ckpt_path, 'pytorch_model.bin'), map_location='cpu'), strict=False)
            missing_root_keys = set([k.split(".")[0] for k in missing_keys])
            print('load ', resume_ckpt_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys)
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        accelerator= Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs]
        )
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # total_prints_per_epoch = len(train_dataloader.dataset) // (print_steps * bs_per_gpu * accelerator.num_processes) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        optimizer = get_optimizer(
                        [p for n, p in latent_motion_tokenizer.named_parameters() if p.requires_grad], 
                        lr=lr_max, 
                        wd=weight_decay
                    )
        
        # linear_warmup_total_iters = min(num_warmup_epochs*total_prints_per_epoch, 5000000 // (print_steps * bs_per_gpu * accelerator.num_processes)) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        scheduler = LinearWarmup_CosineAnnealing(
                        optimizer=optimizer,
                        linear_warmup_start_factor=0.5,
                        linear_warmup_total_iters=num_warmup_steps,
                        cosine_annealing_T_max=max_steps-num_warmup_steps,
                        cosine_annealing_eta_min=5e-5
                    )

        latent_motion_tokenizer, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            latent_motion_tokenizer, optimizer, train_dataloader, eval_dataloader
        )
        self.train_dataloader = cycle(train_dataloader)
        self.eval_dataloader = cycle(eval_dataloader)
        self.writer = SummaryWriter(os.path.join(save_path, 'logs'))
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.scheduler = scheduler
        # self.total_prints_per_epoch = total_prints_per_epoch
        self.latent_motion_tokenizer = latent_motion_tokenizer
        # self.optimizer = optimizer
        # self.train_prefetcher = DataPrefetcher(train_dataloader, self.device)
        # self.eval_prefetcher = DataPrefetcher(eval_dataloader, self.device)
        self.rgb_preprocessor = rgb_preprocessor.to(self.device)
        self.save_path = save_path
        # self.save_epochs = save_epochs
        self.save_steps = save_steps
        self.max_steps = max_steps
        # self.num_epochs = num_epochs
        self.eval_steps = eval_steps
        self.print_steps = print_steps
        self.bs_per_gpu = bs_per_gpu
        self.paired_loss = paired_loss
        


    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def process_index(self):
        return self.accelerator.process_index

    def print(self, msg):
        self.accelerator.print(msg)

    def save_checkpoint(self, save_dir):
        unwrapped_latent_motion_tokenizer = self.accelerator.unwrap_model(self.latent_motion_tokenizer)
        state_dict = unwrapped_latent_motion_tokenizer.get_state_dict_to_save()
        
        torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
        omegaconf.OmegaConf.save(unwrapped_latent_motion_tokenizer.config, os.path.join(save_dir, "config.yaml"))

        self.print(f"A new model checkpoint is saved to {save_dir}!!!")
        
    def train(self):
        # eval_loss_steps = len(self.train_prefetcher) // len(self.eval_prefetcher)
        step = 0
        
        while step < self.max_steps:
            # batch, load_time = self.train_prefetcher.next()
            x_target, x_cond, task, actions = next(self.train_dataloader)
            x_target = x_target.to(self.device)
            x_cond = x_cond.to(self.device)
            batch_list = [{'rgb_initial':x_cond[:, i], 'rgb_future':x_target[:, i]} for i in range(x_target.shape[1])]
            # if batch is None:
            #     self.train_prefetcher.reset()
            #     batch, load_time = self.train_prefetcher.next()
            
            with self.accelerator.accumulate(self.latent_motion_tokenizer):
                self.latent_motion_tokenizer.train()
                self.optimizer.zero_grad()
                if self.paired_loss:
                    loss = self.calculate_paired_loss(batch_list, train=True)
                else:
                    loss = 0
                    for i in range(len(batch_list)):
                        loss += self.calculate_loss(batch_list[i], train=True)['loss']
                    loss = loss / len(batch_list)
                # loss0 = self.calculate_loss(batch0, train=True)
                # loss1 = self.calculate_loss(batch1, train=True)
                # loss2 = self.calculate_loss(batch2, train=True)
                # loss={'loss':loss0['loss']+loss1['loss']+loss2['loss']}
                self.accelerator.backward(loss)
                self.optimizer.step()
            
            if step % self.print_steps == 0:
                with torch.no_grad():
                    x_target, x_cond, task, actions = next(self.eval_dataloader)
                    x_target = x_target.to(self.device)
                    x_cond = x_cond.to(self.device)
                    batch_list = [{'rgb_initial':x_cond[:, i], 'rgb_future':x_target[:, i]} for i in range(x_target.shape[1])]
                    if self.paired_loss:
                        loss = self.calculate_paired_loss(batch_list, train=True)
                    else:
                        loss = 0
                        for i in range(len(batch_list)):
                            loss += self.calculate_loss(batch_list[i], train=True)['loss']
                        loss = loss / len(batch_list)
                    
                    # batch0={'rgb_initial':x_cond[:, 0], 'rgb_future':x_target[:, 0]}
                    # batch1={'rgb_initial':x_cond[:, 1], 'rgb_future':x_target[:, 1]}
                    # batch2={'rgb_initial':x_cond[:, 2], 'rgb_future':x_target[:, 2]}
                    # loss0 = self.calculate_loss(batch0, train=True)
                    # loss1 = self.calculate_loss(batch1, train=True)
                    # loss2 = self.calculate_loss(batch2, train=True)
                    # loss={'loss':loss0['loss']+loss1['loss']+loss2['loss']}
                    print(f"step {step} loss {loss}")
                # self.log(log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step)
                # log_loss = {}
                # eval_log_loss = {}

                # cum_load_time = 0
                # clock = time()
                # self.scheduler.step()
            
            if step % self.save_steps == 0:
                self.accelerator.wait_for_everyone()
                save_dir = os.path.join(self.save_path, f'temp_step_{step}')
                if self.is_main:
                    existing_ckpt_dirs = glob(os.path.join(self.save_path, f'temp_step_*'))
                    # for existing_ckpt_dir in existing_ckpt_dirs:
                    #     if existing_ckpt_dir != save_dir:
                    #         shutil.rmtree(existing_ckpt_dir)
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_checkpoint(save_dir)

                visualization_dir = os.path.join(save_dir, 'visualization')
                self.eval_latent_motion_reconstruction(visualization_dir)
            
            self.scheduler.step()
            step += 1
                        
                
            
        # for epoch in range(self.num_epochs+1):
        #     if epoch != 0:
        #         self.accelerator.wait_for_everyone()
        #         save_dir = os.path.join(self.save_path, f'saved_epoch_{epoch}_step_{step}')

        #         if self.is_main:
        #             os.makedirs(save_dir, exist_ok=True)
        #             self.save_checkpoint(save_dir)
                    
        #         visualization_dir = os.path.join(save_dir, 'visualization')
        #         self.eval_latent_motion_reconstruction(visualization_dir)

        #         if epoch == self.num_epochs:
        #             break
        #         if (self.max_epoch is not None) and (epoch >= self.max_epoch):
        #             break

        #     log_loss = {}
        #     eval_log_loss = {}
            
        #     cum_load_time = 0 
        #     clock = time()
        #     batch_idx = 0
        #     batch, load_time = self.train_prefetcher.next()
            
        #     while batch is not None:
        #         with self.accelerator.accumulate(self.latent_motion_tokenizer):

        #             self.latent_motion_tokenizer.train()
        #             self.optimizer.zero_grad()
        #             loss = self.calculate_loss(batch, train=True)
        #             self.accelerator.backward(loss['loss'])
        #             self.optimizer.step()

        #             for key in loss:
        #                 if key not in log_loss:
        #                     log_loss[key] = 0.0
        #                 log_loss[key] += loss[key].detach() / self.print_steps

        #             cum_load_time += load_time / self.print_steps

        #         if (batch_idx+1) % self.print_steps == 0:

        #             with torch.no_grad():
        #                 batch, _ = self.eval_prefetcher.next_without_none()
        #                 self.latent_motion_tokenizer.eval()
        #                 loss = self.calculate_loss(batch, train=True)

        #                 for key in loss:
        #                     eval_log_loss[key] = loss[key].detach()

        #             self.log(log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step)
        #             log_loss = {}
        #             eval_log_loss = {}

        #             cum_load_time = 0
        #             clock = time()
        #             self.scheduler.step()

        #         if step % self.save_steps == 0:
        #             self.accelerator.wait_for_everyone()
        #             save_dir = os.path.join(self.save_path, f'temp_epoch_{epoch}_step_{step}')
        #             if self.is_main:
        #                 existing_ckpt_dirs = glob(os.path.join(self.save_path, f'temp_epoch_*_step_*'))
        #                 for existing_ckpt_dir in existing_ckpt_dirs:
        #                     if existing_ckpt_dir != save_dir:
        #                         shutil.rmtree(existing_ckpt_dir)
        #                 os.makedirs(save_dir, exist_ok=True)
        #                 self.save_checkpoint(save_dir)

        #             visualization_dir = os.path.join(save_dir, 'visualization')
        #             self.eval_latent_motion_reconstruction(visualization_dir)

        #         batch_idx += 1
        #         step += 1
        #         batch, load_time = self.train_prefetcher.next()



    @torch.no_grad()
    def eval_latent_motion_reconstruction(self, visualization_dir):
        os.makedirs(visualization_dir, exist_ok=True)
        self.print(f"Saving visualization results to {visualization_dir} ...")
        x_target, x_cond, task, actions = next(self.eval_dataloader)
        x_target = x_target.to(self.device)
        x_cond = x_cond.to(self.device)
        batch_list = [{'rgb_initial':x_cond[:, i], 'rgb_future':x_target[:, i]} for i in range(x_target.shape[1])]
        
        # batch0={'rgb_initial':x_cond[:, 0], 'rgb_future':x_target[:, 0]}
        # batch1={'rgb_initial':x_cond[:, 1], 'rgb_future':x_target[:, 1]}
        # batch2={'rgb_initial':x_cond[:, 2], 'rgb_future':x_target[:, 2]}
        
        orig_rgb_seq_list = [torch.stack([batch['rgb_initial'], batch['rgb_future']], dim=1) for batch in batch_list] # (b, 2, c, h, w)
        rgb_seq_list = [self.rgb_preprocessor(rgb_seq, train=True) for rgb_seq in orig_rgb_seq_list]
        # orig_rgb_seq = torch.stack([batch_list[0]['rgb_initial'], batch_list[0]['rgb_future']], dim=1) # (b, 2, c, h, w)
        # rgb_seq = self.rgb_preprocessor(orig_rgb_seq, train=True)

        self.latent_motion_tokenizer.eval()
        if self.paired_loss:
            outputs_same_corner = self.latent_motion_tokenizer(
                cond_pixel_values1=rgb_seq_list[0][:,0],
                target_pixel_values1=rgb_seq_list[0][:,1],
                cond_pixel_values2=rgb_seq_list[0][:,0],
                target_pixel_values2=rgb_seq_list[0][:,1],
                return_recons_only=True
            )
            outputs_diff_corner = self.latent_motion_tokenizer(
                cond_pixel_values1=rgb_seq_list[0][:,0],
                target_pixel_values1=rgb_seq_list[0][:,1],
                cond_pixel_values2=rgb_seq_list[1][:,0],
                target_pixel_values2=rgb_seq_list[1][:,1],
                return_recons_only=True
            )
            recons_rgb_future_same_corner = self.rgb_preprocessor.post_process(outputs_same_corner["recons_pixel_values"]).detach().cpu()  # (b, c, h, w)
            recons_rgb_future_diff_corner = self.rgb_preprocessor.post_process(outputs_diff_corner["recons_pixel_values"]).detach().cpu()  # (b, c, h, w)
            gt_latent_motion_ids_same_corner = outputs_same_corner["indices"].detach().cpu() # (b, per_latent_motion_len)
            gt_latent_motion_ids_diff_corner = outputs_diff_corner["indices"].detach().cpu() # (b, per_latent_motion_len)
            orig_rgb_seq_list = [self.rgb_preprocessor.post_process(rgb_seq).detach().cpu() for rgb_seq in rgb_seq_list]
            for i in range(orig_rgb_seq_list[0].shape[0]):
                visualize_latent_motion_reconstruction(
                    initial_frame=orig_rgb_seq_list[0][i,0],
                    next_frame=orig_rgb_seq_list[0][i,1],
                    recons_next_frame=recons_rgb_future_same_corner[i],
                    latent_motion_ids=gt_latent_motion_ids_same_corner[i],
                    path=os.path.join(visualization_dir, f"{self.process_index}-{i}-samecorner.png")
                )
                visualize_latent_motion_reconstruction(
                    initial_frame=orig_rgb_seq_list[1][i,0],
                    next_frame=orig_rgb_seq_list[1][i,1],
                    recons_next_frame=recons_rgb_future_diff_corner[i],
                    latent_motion_ids=gt_latent_motion_ids_diff_corner[i],
                    path=os.path.join(visualization_dir, f"{self.process_index}-{i}-diffcorner.png")
                )
        else:
            outputs = self.latent_motion_tokenizer(
                cond_pixel_values=rgb_seq_list[0][:,0],
                target_pixel_values=rgb_seq_list[0][:,1],
                return_recons_only=True
            )
                
            recons_rgb_future = self.rgb_preprocessor.post_process(outputs["recons_pixel_values"]).detach().cpu()  # (b, c, h, w)
            gt_latent_motion_ids = outputs["indices"].detach().cpu() # (b, per_latent_motion_len)
            # orig_rgb_seq = orig_rgb_seq.detach().cpu()
            orig_rgb_seq_list = [self.rgb_preprocessor.post_process(rgb_seq).detach().cpu() for rgb_seq in rgb_seq_list]

            for i in range(orig_rgb_seq_list[0].shape[0]):
                visualize_latent_motion_reconstruction(
                    initial_frame=orig_rgb_seq_list[0][i,0],
                    next_frame=orig_rgb_seq_list[0][i,1],
                    recons_next_frame=recons_rgb_future[i],
                    latent_motion_ids=gt_latent_motion_ids[i],
                    path=os.path.join(visualization_dir, f"{self.process_index}-{i}.png")
                )


    def calculate_loss(self, batch, train):
        # image preprocessing
        rgb_seq = torch.stack([batch['rgb_initial'], batch['rgb_future']], dim=1)
        rgb_seq = self.rgb_preprocessor(rgb_seq, train=train)
        # compute loss
        loss = self.latent_motion_tokenizer(
            cond_pixel_values=rgb_seq[:,0],
            target_pixel_values=rgb_seq[:,1]
        )

        return loss
    
    def calculate_paired_loss(self, batch_list, train):
        N = len(batch_list)
        # image preprocessing
        rgb_seq_list = [torch.stack([batch['rgb_initial'], batch['rgb_future']], dim=1) for batch in batch_list]
        rgb_seq_list = [self.rgb_preprocessor(rgb_seq, train=train) for rgb_seq in rgb_seq_list]
        # compute paired loss
        loss = 0
        for i in range(N):
            for j in range(N):
                pred = self.latent_motion_tokenizer(
                    cond_pixel_values1=rgb_seq_list[i][:,0],
                    target_pixel_values1=rgb_seq_list[i][:,1],
                    cond_pixel_values2=rgb_seq_list[j][:,0],
                    target_pixel_values2=rgb_seq_list[j][:,1]
                )
                print(f"active code num: {pred['active_code_num']}")
                loss += pred['loss']
        loss = loss / (N*N)
        return loss
        
    
    def log(self, log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step):
        for key in log_loss:
            log_loss[key] = self.accelerator.gather_for_metrics(log_loss[key]).mean()
        for key in eval_log_loss:
            eval_log_loss[key] = self.accelerator.gather_for_metrics(eval_log_loss[key]).mean()
        load_pecnt = torch.tensor(cum_load_time / (time()-clock)).to(self.device)
        load_pecnt = self.accelerator.gather_for_metrics(load_pecnt).mean()
        fps = (self.bs_per_gpu*self.print_steps*2) / (time()-clock)
        fps = self.accelerator.gather_for_metrics(torch.tensor(fps).to(self.device)).sum()

        text = 'Train Epoch: {} [{}/{} ({:.0f}%)] FPS:{:.5f} Load Pertentage:{:.5f} LR:{}'.format(
            epoch, 
            batch_idx * self.bs_per_gpu * self.accelerator.num_processes, 
            len(self.train_prefetcher), 
            100. * batch_idx * self.bs_per_gpu * self.accelerator.num_processes / len(self.train_prefetcher),
            fps,
            load_pecnt,
            self.scheduler.get_last_lr()[0],
        )
        for key in log_loss:
            text = text + ' {}: {:.5f}'.format(key, log_loss[key])
        for key in eval_log_loss:
            text = text + ' eval_{}: {:.5f}'.format(key, eval_log_loss[key])
        self.print(text)
        if self.is_main:
            for key in log_loss:
                self.writer.add_scalar(key, log_loss[key], step)
            for key in eval_log_loss:
                self.writer.add_scalar('eval_'+key, eval_log_loss[key], step)
            self.writer.add_scalar("learning rate", self.scheduler.get_last_lr()[0], step)
            self.writer.add_scalar("FPS", fps, step)
            self.writer.add_scalar("loading time in total time", load_pecnt, step)


class LatentMotionTokenizer_Trainer_RLBench:
    def __init__(
        self,
        latent_motion_tokenizer,
        rgb_preprocessor,
        train_dataloader,
        eval_dataloader,
        save_path,
        save_epochs=1,
        save_steps=10000,
        num_epochs=20,
        print_steps=100,
        lr_max=0.0001,
        weight_decay=0.,
        num_warmup_epochs=1,
        gradient_accumulation_steps=4,
        resume_ckpt_path=None,
        bs_per_gpu=32,
        max_epoch=None,
        paired_loss = False
    ):
        if resume_ckpt_path is not None:
            print(f"resuming Latent Motion Tokenizer from {resume_ckpt_path} ...")
            missing_keys, unexpected_keys = latent_motion_tokenizer.load_state_dict(torch.load(os.path.join(resume_ckpt_path, 'pytorch_model.bin'), map_location='cpu'), strict=False)
            missing_root_keys = set([k.split(".")[0] for k in missing_keys])
            print('load ', resume_ckpt_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys)
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        accelerator= Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs]
        )
        self.gradient_accumulation_steps = gradient_accumulation_steps

        total_prints_per_epoch = len(train_dataloader.dataset) // (print_steps * bs_per_gpu * accelerator.num_processes)

        optimizer = get_optimizer(
                        [p for n, p in latent_motion_tokenizer.named_parameters() if p.requires_grad], 
                        lr=lr_max, 
                        wd=weight_decay
                    )
        
        linear_warmup_total_iters = min(num_warmup_epochs*total_prints_per_epoch, 5000000 // (print_steps * bs_per_gpu * accelerator.num_processes))
        scheduler = LinearWarmup_CosineAnnealing(
                        optimizer=optimizer,
                        linear_warmup_start_factor=0.5,
                        linear_warmup_total_iters=linear_warmup_total_iters,
                        cosine_annealing_T_max=num_epochs*total_prints_per_epoch-linear_warmup_total_iters,
                        cosine_annealing_eta_min=5e-5
                    )

        latent_motion_tokenizer, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            latent_motion_tokenizer, optimizer, train_dataloader, eval_dataloader, 
            device_placement=[True, True, False, False]
        )
        
        self.writer = SummaryWriter(os.path.join(save_path, 'logs'))
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_prints_per_epoch = total_prints_per_epoch
        self.latent_motion_tokenizer = latent_motion_tokenizer
        self.optimizer = optimizer
        # self.train_prefetcher = DataPrefetcher(train_dataloader, self.device)
        # self.eval_prefetcher = DataPrefetcher(eval_dataloader, self.device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = cycle(eval_dataloader)
        self.rgb_preprocessor = rgb_preprocessor.to(self.device)
        self.save_path = save_path
        self.save_epochs = save_epochs
        self.save_steps = save_steps
        self.max_epoch = max_epoch
        self.num_epochs = num_epochs
        self.print_steps = print_steps
        self.bs_per_gpu = bs_per_gpu
        self.paired_loss = paired_loss


    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def process_index(self):
        return self.accelerator.process_index

    def print(self, msg):
        self.accelerator.print(msg)

    def save_checkpoint(self, save_dir):
        unwrapped_latent_motion_tokenizer = self.accelerator.unwrap_model(self.latent_motion_tokenizer)
        state_dict = unwrapped_latent_motion_tokenizer.get_state_dict_to_save()
        
        torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
        omegaconf.OmegaConf.save(unwrapped_latent_motion_tokenizer.config, os.path.join(save_dir, "config.yaml"))

        self.print(f"A new model checkpoint is saved to {save_dir}!!!")
        
    def train(self):
        # eval_loss_steps = len(self.train_prefetcher) // len(self.eval_prefetcher)
        step = 0
        use_paired_loss = False
        for epoch in range(self.num_epochs+1):
            if epoch != 0:
                self.accelerator.wait_for_everyone()
                save_dir = os.path.join(self.save_path, f'saved_epoch_{epoch}_step_{step}')

                if self.is_main:
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_checkpoint(save_dir)
                    
                visualization_dir = os.path.join(save_dir, 'visualization')
                self.eval_latent_motion_reconstruction(visualization_dir)

                if epoch == self.num_epochs:
                    break
                if (self.max_epoch is not None) and (epoch >= self.max_epoch):
                    break

            log_loss = {}
            eval_log_loss = {}
            
            # cum_load_time = 0 
            # clock = time()
            batch_idx = 0
            # batch, load_time = self.train_prefetcher.next()
            # batch = self.train_dataloader.next()
            for batch in self.train_dataloader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                with self.accelerator.accumulate(self.latent_motion_tokenizer):
                    
                    self.latent_motion_tokenizer.train()
                    self.optimizer.zero_grad()
                    if self.paired_loss:
                        loss = self.calculate_paired_loss(batch, train=True, use_paired_loss=use_paired_loss)
                    else:
                        loss = self.calculate_loss(batch, train=True)
                    self.accelerator.backward(loss['loss'])
                    self.optimizer.step()
                    
                    for key in loss:
                        if key not in log_loss:
                            log_loss[key] = 0.0
                        log_loss[key] += loss[key].detach() / self.print_steps
                        
                if (batch_idx+1) % self.print_steps == 0:
                    with torch.no_grad():
                        eval_batch = next(self.eval_dataloader)
                        for k, v in eval_batch.items():
                            if isinstance(v, torch.Tensor):
                                eval_batch[k] = v.to(self.device)
                        self.latent_motion_tokenizer.eval()
                        if self.paired_loss:
                            eval_log_loss = self.calculate_paired_loss(eval_batch, train=True)
                        else:
                            eval_log_loss = self.calculate_loss(eval_batch, train=True)
                        
                        for key in loss:
                            eval_log_loss[key] = loss[key].detach()
                    
                    self.log(log_loss, eval_log_loss, epoch, batch_idx, step)
                    log_loss = {}
                    eval_log_loss = {}
                    
                    self.scheduler.step()
                
                if step % self.save_steps == 0:
                    self.accelerator.wait_for_everyone()
                    save_dir = os.path.join(self.save_path, f'temp_epoch_{epoch}_step_{step}')
                    if self.is_main:
                        existing_ckpt_dirs = glob(os.path.join(self.save_path, f'temp_epoch_*_step_*'))
                        for existing_ckpt_dir in existing_ckpt_dirs:
                            if existing_ckpt_dir != save_dir:
                                shutil.rmtree(existing_ckpt_dir)
                        os.makedirs(save_dir, exist_ok=True)
                        self.save_checkpoint(save_dir)
                        
                    visualization_dir = os.path.join(save_dir, 'visualization')
                    self.eval_latent_motion_reconstruction(visualization_dir)
                
                batch_idx += 1
                step += 1
                if step >= 0:
                    use_paired_loss = True
            # while batch is not None:
            #     with self.accelerator.accumulate(self.latent_motion_tokenizer):

            #         self.latent_motion_tokenizer.train()
            #         self.optimizer.zero_grad()
            #         loss = self.calculate_loss(batch, train=True)
            #         self.accelerator.backward(loss['loss'])
            #         self.optimizer.step()

            #         for key in loss:
            #             if key not in log_loss:
            #                 log_loss[key] = 0.0
            #             log_loss[key] += loss[key].detach() / self.print_steps

            #         cum_load_time += load_time / self.print_steps

            #     if (batch_idx+1) % self.print_steps == 0:

            #         with torch.no_grad():
            #             batch, _ = self.eval_prefetcher.next_without_none()
            #             self.latent_motion_tokenizer.eval()
            #             loss = self.calculate_loss(batch, train=True)

            #             for key in loss:
            #                 eval_log_loss[key] = loss[key].detach()

            #         self.log(log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step)
            #         log_loss = {}
            #         eval_log_loss = {}

            #         cum_load_time = 0
            #         clock = time()
            #         self.scheduler.step()

            #     if step % self.save_steps == 0:
            #         self.accelerator.wait_for_everyone()
            #         save_dir = os.path.join(self.save_path, f'temp_epoch_{epoch}_step_{step}')
            #         if self.is_main:
            #             existing_ckpt_dirs = glob(os.path.join(self.save_path, f'temp_epoch_*_step_*'))
            #             for existing_ckpt_dir in existing_ckpt_dirs:
            #                 if existing_ckpt_dir != save_dir:
            #                     shutil.rmtree(existing_ckpt_dir)
            #             os.makedirs(save_dir, exist_ok=True)
            #             self.save_checkpoint(save_dir)

            #         visualization_dir = os.path.join(save_dir, 'visualization')
            #         self.eval_latent_motion_reconstruction(visualization_dir)

            #     batch_idx += 1
            #     step += 1
            #     batch, load_time = self.train_prefetcher.next()



    @torch.no_grad()
    def eval_latent_motion_reconstruction(self, visualization_dir):
        os.makedirs(visualization_dir, exist_ok=True)
        self.print(f"Saving visualization results to {visualization_dir} ...")
        # batch, _ = self.eval_prefetcher.next_without_none()
        batch = next(self.eval_dataloader)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        # orig_rgb_seq = torch.stack([batch['rgb_initial'], batch['rgb_future']], dim=1) # (b, 2, cameras, c, h, w)
        
        orig_rgb_seq = batch['rgbs'] # (b, 2, cameras, c, h, w)
        rgb_seq = torch.zeros_like(orig_rgb_seq)
        for i in range(orig_rgb_seq.shape[2]):
            rgb_seq[:,:,i] = self.rgb_preprocessor(orig_rgb_seq[:,:,i], train=True)

        self.latent_motion_tokenizer.eval()
        if self.paired_loss:
            outputs_same_corner = self.latent_motion_tokenizer(
                cond_pixel_values1=rgb_seq[:, 0, 0], # (b, c, h, w)
                target_pixel_values1=rgb_seq[:, 1, 0],
                cond_pixel_values2=rgb_seq[:, 0, 0],
                target_pixel_values2=rgb_seq[:, 1, 0],
                return_recons_only=True
            )
            outputs_diff_corner1 = self.latent_motion_tokenizer(
                cond_pixel_values1=rgb_seq[:, 0, 0],
                target_pixel_values1=rgb_seq[:, 1, 0],
                cond_pixel_values2=rgb_seq[:, 0, 1],
                target_pixel_values2=rgb_seq[:, 1, 1],
                return_recons_only=True
            )
            
            outputs_diff_corner2 = self.latent_motion_tokenizer(
                cond_pixel_values1=rgb_seq[:, 0, 1],
                target_pixel_values1=rgb_seq[:, 1, 1],
                cond_pixel_values2=rgb_seq[:, 0, 0],
                target_pixel_values2=rgb_seq[:, 1, 0],
                return_recons_only=True
            )
                
            recons_rgb_future_same_corner = self.rgb_preprocessor.post_process(outputs_same_corner["recons_pixel_values"]).detach().cpu()
            recons_rgb_future_diff_corner1 = self.rgb_preprocessor.post_process(outputs_diff_corner1["recons_pixel_values"]).detach().cpu()
            gt_latent_motion_ids_diff_corner1 = outputs_diff_corner1["indices"].detach().cpu()
            gt_latent_motion_ids_diff_corner2 = outputs_diff_corner2["indices"].detach().cpu()
            orig_rgb_seq_same_corner = self.rgb_preprocessor.post_process(rgb_seq[:,:,0]).detach().cpu()
            orig_rgb_seq_diff_corner1 = self.rgb_preprocessor.post_process(rgb_seq[:,:,1]).detach().cpu()
            for i in range(orig_rgb_seq_same_corner.shape[0]):
                visualize_latent_motion_reconstruction(
                    initial_frame=orig_rgb_seq_same_corner[i,0],
                    next_frame=orig_rgb_seq_same_corner[i,1],
                    recons_next_frame=recons_rgb_future_same_corner[i],
                    latent_motion_ids=gt_latent_motion_ids_diff_corner1[i],
                    path=os.path.join(visualization_dir, f"{self.process_index}-{i}-samecorner.png")
                )
                
            for i in range(orig_rgb_seq_diff_corner1.shape[0]):
                visualize_latent_motion_reconstruction(
                    initial_frame=orig_rgb_seq_diff_corner1[i,0],
                    next_frame=orig_rgb_seq_diff_corner1[i,1],
                    recons_next_frame=recons_rgb_future_diff_corner1[i],
                    latent_motion_ids=gt_latent_motion_ids_diff_corner1[i],
                    path=os.path.join(visualization_dir, f"{self.process_index}-{i}-diffcorner.png")
                )
            
        else:
            outputs = self.latent_motion_tokenizer(
                cond_pixel_values=rgb_seq[:, 0, 0],
                target_pixel_values=rgb_seq[:, 1, 0],
                return_recons_only=True
            )    
            recons_rgb_future = self.rgb_preprocessor.post_process(outputs["recons_pixel_values"]).detach().cpu()
            gt_latent_motion_ids = outputs["indices"].detach().cpu()
            orig_rgb_seq = self.rgb_preprocessor.post_process(rgb_seq[:,:,0]).detach().cpu()
            for i in range(orig_rgb_seq.shape[0]):
                visualize_latent_motion_reconstruction(
                    initial_frame=orig_rgb_seq[i,0],
                    next_frame=orig_rgb_seq[i,1],
                    recons_next_frame=recons_rgb_future[i],
                    latent_motion_ids=gt_latent_motion_ids[i],
                    path=os.path.join(visualization_dir, f"{self.process_index}-{i}.png")
                )
            


    def calculate_loss(self, batch, train):
        # image preprocessing
        # orig_rgb_seq = torch.stack([batch['rgb_initial'], batch['rgb_future']], dim=1) # (b, 2, cameras, c, h, w)
        # rgb_seq = torch.zeros_like(orig_rgb_seq)
        # for i in range(orig_rgb_seq.shape[2]):
        #     rgb_seq[:,:,i] = self.rgb_preprocessor(orig_rgb_seq[:,:,i], train=train)
        orig_rgb_seq = batch['rgbs'] # (b, 2, cameras, c, h, w)
        rgb_seq = torch.zeros_like(orig_rgb_seq)
        for i in range(orig_rgb_seq.shape[2]):
            rgb_seq[:,:,i] = self.rgb_preprocessor(orig_rgb_seq[:,:,i], train=True)

        # compute loss
        loss_sum = {}
        for i in range(orig_rgb_seq.shape[2]):
            loss = self.latent_motion_tokenizer(
                cond_pixel_values=rgb_seq[:, 0, i],
                target_pixel_values=rgb_seq[:, 1, i]
            )
            
            for key in loss:
                if key not in loss_sum:
                    loss_sum[key] = 0.0
                loss_sum[key] += loss[key] / orig_rgb_seq.shape[2]
        return loss_sum
    
    def calculate_paired_loss(self, batch, train, use_paired_loss=True):
        # image preprocessing
        # orig_rgb_seq = torch.stack([batch['rgb_initial'], batch['rgb_future']], dim=1) # (b, 2, cameras, c, h, w)
        # rgb_seq = torch.zeros_like(orig_rgb_seq)
        # for i in range(orig_rgb_seq.shape[2]):
        #     rgb_seq[:,:,i] = self.rgb_preprocessor(orig_rgb_seq[:,:,i], train=train)
        orig_rgb_seq = batch['rgbs'] # (b, 2, cameras, c, h, w)
        rgb_seq = torch.zeros_like(orig_rgb_seq)
        for i in range(orig_rgb_seq.shape[2]):
            rgb_seq[:,:,i] = self.rgb_preprocessor(orig_rgb_seq[:,:,i], train=True)
        loss_sum = {}
        N = orig_rgb_seq.shape[2]
        if use_paired_loss:
            for i in range(N*N):
                idx1 = i // N
                idx2 = i % N
                loss = self.latent_motion_tokenizer(
                    cond_pixel_values1=rgb_seq[:, 0, idx1],
                    target_pixel_values1=rgb_seq[:, 1, idx1],
                    cond_pixel_values2=rgb_seq[:, 0, idx2],
                    target_pixel_values2=rgb_seq[:, 1, idx2]
                )
                for key in loss:
                    if key not in loss_sum:
                        loss_sum[key] = 0.0
                    loss_sum[key] += loss[key] / (N*N)
        else:
            for i in range(N):
                loss = self.latent_motion_tokenizer(
                    cond_pixel_values1=rgb_seq[:, 0, i],
                    target_pixel_values1=rgb_seq[:, 1, i],
                    cond_pixel_values2=rgb_seq[:, 0, i],
                    target_pixel_values2=rgb_seq[:, 1, i]
                )
                for key in loss:
                    if key not in loss_sum:
                        loss_sum[key] = 0.0
                    loss_sum[key] += loss[key] / N
        
        return loss_sum


    def log(self, log_loss, eval_log_loss, epoch, batch_idx, step):
        for key in log_loss:
            log_loss[key] = self.accelerator.gather_for_metrics(log_loss[key]).mean()
        for key in eval_log_loss:
            eval_log_loss[key] = self.accelerator.gather_for_metrics(eval_log_loss[key]).mean()
        # load_pecnt = torch.tensor(cum_load_time / (time()-clock)).to(self.device)
        # load_pecnt = self.accelerator.gather_for_metrics(load_pecnt).mean()
        # fps = (self.bs_per_gpu*self.print_steps*2) / (time()-clock)
        # fps = self.accelerator.gather_for_metrics(torch.tensor(fps).to(self.device)).sum()
        text = 'Train Epoch: {} Batch: {} Step: {} LR: {}'.format(epoch, batch_idx, step, self.scheduler.get_last_lr()[0])
        
        # text = 'Train Epoch: {} [{}/{} ({:.0f}%)] FPS:{:.5f} Load Pertentage:{:.5f} LR:{}'.format(
        #     epoch, 
        #     batch_idx * self.bs_per_gpu * self.accelerator.num_processes, 
        #     len(self.train_prefetcher), 
        #     100. * batch_idx * self.bs_per_gpu * self.accelerator.num_processes / len(self.train_prefetcher),
        #     fps,
        #     load_pecnt,
        #     self.scheduler.get_last_lr()[0],
        # )
        for key in log_loss:
            text = text + ' {}: {:.5f}'.format(key, log_loss[key])
        for key in eval_log_loss:
            text = text + ' eval_{}: {:.5f}'.format(key, eval_log_loss[key])
        self.print(text)
        if self.is_main:
            for key in log_loss:
                self.writer.add_scalar(key, log_loss[key], step)
            for key in eval_log_loss:
                self.writer.add_scalar('eval_'+key, eval_log_loss[key], step)
            self.writer.add_scalar("learning rate", self.scheduler.get_last_lr()[0], step)
            # self.writer.add_scalar("FPS", fps, step)
            # self.writer.add_scalar("loading time in total time", load_pecnt, step)

class LatentMotionTokenizer_Trainer_Multiview:
    def __init__(
        self,
        latent_motion_tokenizer,
        rgb_preprocessor,
        train_dataloader,
        eval_dataloader,
        save_path,
        save_epochs=1,
        save_steps=10000,
        num_epochs=20,
        print_steps=100,
        lr_max=0.0001,
        weight_decay=0.,
        num_warmup_epochs=1,
        gradient_accumulation_steps=4,
        resume_ckpt_path=None,
        bs_per_gpu=32,
        max_epoch=None,
        paired_loss = False,
        lm_restrict_weight=None,
        paired_method = None
    ):
        if resume_ckpt_path is not None:
            print(f"resuming Latent Motion Tokenizer from {resume_ckpt_path} ...")
            missing_keys, unexpected_keys = latent_motion_tokenizer.load_state_dict(torch.load(os.path.join(resume_ckpt_path, 'pytorch_model.bin'), map_location='cpu'), strict=False)
            missing_root_keys = set([k.split(".")[0] for k in missing_keys])
            print('load ', resume_ckpt_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys)
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        accelerator= Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs]
        )
        self.gradient_accumulation_steps = gradient_accumulation_steps

        total_prints_per_epoch = len(train_dataloader.dataset) // (print_steps * bs_per_gpu * accelerator.num_processes)

        optimizer = get_optimizer(
                        [p for n, p in latent_motion_tokenizer.named_parameters() if p.requires_grad], 
                        lr=lr_max, 
                        wd=weight_decay
                    )
        
        linear_warmup_total_iters = min(num_warmup_epochs*total_prints_per_epoch, 5000000 // (print_steps * bs_per_gpu * accelerator.num_processes))
        scheduler = LinearWarmup_CosineAnnealing(
                        optimizer=optimizer,
                        linear_warmup_start_factor=0.5,
                        linear_warmup_total_iters=linear_warmup_total_iters,
                        cosine_annealing_T_max=num_epochs*total_prints_per_epoch-linear_warmup_total_iters,
                        cosine_annealing_eta_min=5e-5
                    )

        latent_motion_tokenizer, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            latent_motion_tokenizer, optimizer, train_dataloader, eval_dataloader, 
            device_placement=[True, True, False, False]
        )
        
        self.writer = SummaryWriter(os.path.join(save_path, 'logs'))
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_prints_per_epoch = total_prints_per_epoch
        self.latent_motion_tokenizer = latent_motion_tokenizer
        self.optimizer = optimizer
        self.train_prefetcher = DataPrefetcher(train_dataloader, self.device)
        self.eval_prefetcher = DataPrefetcher(eval_dataloader, self.device)
        self.rgb_preprocessor = rgb_preprocessor.to(self.device)
        self.save_path = save_path
        self.save_epochs = save_epochs
        self.save_steps = save_steps
        self.max_epoch = max_epoch
        self.num_epochs = num_epochs
        self.print_steps = print_steps
        self.bs_per_gpu = bs_per_gpu
        self.paired_loss = paired_loss
        self.lm_restrict_weight = lm_restrict_weight
        self.paired_method = paired_method
        print(f"paired method: {self.paired_method}, paired loss: {self.paired_loss}, lm_restrict_weight: {self.lm_restrict_weight}")


    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def process_index(self):
        return self.accelerator.process_index

    def print(self, msg):
        self.accelerator.print(msg)

    def save_checkpoint(self, save_dir):
        unwrapped_latent_motion_tokenizer = self.accelerator.unwrap_model(self.latent_motion_tokenizer)
        state_dict = unwrapped_latent_motion_tokenizer.get_state_dict_to_save()
        
        torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
        omegaconf.OmegaConf.save(unwrapped_latent_motion_tokenizer.config, os.path.join(save_dir, "config.yaml"))

        self.print(f"A new model checkpoint is saved to {save_dir}!!!")
        
    def train(self):
        eval_loss_steps = len(self.train_prefetcher) // len(self.eval_prefetcher)
        step = 0
        self.accelerator.print(
            f"Accelerate world size: {self.accelerator.num_processes} (GPUs), "
            f"process_index: {self.accelerator.process_index}, "
            f"local_process_index: {self.accelerator.local_process_index}, "
            f"device: {self.accelerator.device}"
        )
        
        for epoch in range(self.num_epochs+1):
            if epoch != 0:
                self.accelerator.wait_for_everyone()
                save_dir = os.path.join(self.save_path, f'saved_epoch_{epoch}_step_{step}')

                if self.is_main:
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_checkpoint(save_dir)
                    
                visualization_dir = os.path.join(save_dir, 'visualization')
                self.eval_latent_motion_reconstruction(visualization_dir)

                if epoch == self.num_epochs:
                    break
                if (self.max_epoch is not None) and (epoch >= self.max_epoch):
                    break

            log_loss = {}
            eval_log_loss = {}
            
            cum_load_time = 0 
            clock = time()
            batch_idx = 0
            batch, load_time = self.train_prefetcher.next()
            
            while batch is not None:
                with self.accelerator.accumulate(self.latent_motion_tokenizer):

                    self.latent_motion_tokenizer.train()
                    self.optimizer.zero_grad()
                    if self.paired_loss:
                        if self.paired_method == 'lm_restrict':
                            assert self.lm_restrict_weight is not None, "lm_restrict_weight must be set when paired_method is 'lm_restrict'"
                            loss = self.calculate_paired_loss_lm_restrict(batch, train=True, weight=self.lm_restrict_weight)
                        elif self.paired_method == 'paired':
                            loss = self.calculate_paired_loss(batch, train=True)
                        elif self.paired_method == 'paired_half':
                            loss = self.calculate_paired_loss_half(batch, train=True)
                        elif self.paired_method == '3d':
                            loss = self.calculate_3d_loss(batch, train=True)
                        else:    
                            raise ValueError(f"Unknown paired_method: {self.paired_method}")
                    else:
                        loss = self.calculate_loss(batch, train=True)
                    self.accelerator.backward(loss['loss'])
                    self.optimizer.step()

                    for key in loss:
                        if key not in log_loss:
                            log_loss[key] = 0.0
                        log_loss[key] += loss[key].detach() / self.print_steps

                    cum_load_time += load_time / self.print_steps

                if (batch_idx+1) % self.print_steps == 0:

                    with torch.no_grad():
                        batch, _ = self.eval_prefetcher.next_without_none()
                        self.latent_motion_tokenizer.eval()
                        if self.paired_loss:
                            loss = self.calculate_paired_loss(batch, train=True)
                        else:
                            loss = self.calculate_loss(batch, train=True)

                        for key in loss:
                            eval_log_loss[key] = loss[key].detach()

                    self.log(log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step)
                    wandb.log({"Epoch": epoch, "Step": step, "loss": log_loss, "num_processes": self.accelerator.num_processes})
                    log_loss = {}
                    eval_log_loss = {}

                    cum_load_time = 0
                    clock = time()
                    self.scheduler.step()

                if step % self.save_steps == 0:
                    self.accelerator.wait_for_everyone()
                    save_dir = os.path.join(self.save_path, f'temp_epoch_{epoch}_step_{step}')
                    if self.is_main:
                        existing_ckpt_dirs = glob(os.path.join(self.save_path, f'temp_epoch_*_step_*'))
                        for existing_ckpt_dir in existing_ckpt_dirs:
                            if existing_ckpt_dir != save_dir:
                                shutil.rmtree(existing_ckpt_dir)
                        os.makedirs(save_dir, exist_ok=True)
                        self.save_checkpoint(save_dir)

                    visualization_dir = os.path.join(save_dir, 'visualization')
                    self.eval_latent_motion_reconstruction(visualization_dir)

                batch_idx += 1
                step += 1
                batch, load_time = self.train_prefetcher.next()



    @torch.no_grad()
    def eval_latent_motion_reconstruction(self, visualization_dir):
        os.makedirs(visualization_dir, exist_ok=True)
        self.print(f"Saving visualization results to {visualization_dir} ...")
        batch, _ = self.eval_prefetcher.next_without_none()

        orig_rgb_seq_static = torch.cat([batch['rgb_initial_static'], batch['rgb_future_static']], dim=1) # (b, 2, c, h, w)
        rgb_seq_static = self.rgb_preprocessor(orig_rgb_seq_static, train=True)
        orig_rgb_seq_gripper = torch.cat([batch['rgb_initial_gripper'], batch['rgb_future_gripper']], dim=1) # (b, 2, c, h, w)
        rgb_seq_gripper = self.rgb_preprocessor(orig_rgb_seq_gripper, train=True)

        self.latent_motion_tokenizer.eval()
        if self.paired_loss:
            if self.paired_method == '3d':
                outputs = self.latent_motion_tokenizer(
                    cond_pixel_values1=rgb_seq_static[:,0], # (b, c, h, w)
                    target_pixel_values1=rgb_seq_static[:,1],
                    cond_pixel_values2=rgb_seq_gripper[:,0],
                    target_pixel_values2=rgb_seq_gripper[:,1],
                    return_recons_only=True,
                )
                recons_rgb_future_static = self.rgb_preprocessor.post_process(outputs["recons_pixel_values1"]).detach().cpu()
                recons_rgb_future_gripper = self.rgb_preprocessor.post_process(outputs["recons_pixel_values2"]).detach().cpu()
                gt_latent_motion_ids = outputs["indices"].detach().cpu()
                orig_rgb_seq_static = self.rgb_preprocessor.post_process(rgb_seq_static).detach().cpu()
                orig_rgb_seq_gripper = self.rgb_preprocessor.post_process(rgb_seq_gripper).detach().cpu()
                for i in range(orig_rgb_seq_static.shape[0]):
                    visualize_latent_motion_reconstruction(
                        initial_frame=orig_rgb_seq_static[i,0],
                        next_frame=orig_rgb_seq_static[i,1],
                        recons_next_frame=recons_rgb_future_static[i],
                        latent_motion_ids=gt_latent_motion_ids[i],
                        path=os.path.join(visualization_dir, f"{self.process_index}-{i}-static.png")
                    )
                for i in range(orig_rgb_seq_gripper.shape[0]):
                    visualize_latent_motion_reconstruction(
                        initial_frame=orig_rgb_seq_gripper[i,0],
                        next_frame=orig_rgb_seq_gripper[i,1],
                        recons_next_frame=recons_rgb_future_gripper[i],
                        latent_motion_ids=gt_latent_motion_ids[i],
                        path=os.path.join(visualization_dir, f"{self.process_index}-{i}-gripper.png")
                    )
            else:
                
                cond1 = torch.cat([
                    rgb_seq_static[:,0],
                    rgb_seq_gripper[:,0],
                    rgb_seq_static[:,0],
                    rgb_seq_gripper[:,0]
                ], dim=0)
                target1 = torch.cat([
                    rgb_seq_static[:,1],
                    rgb_seq_gripper[:,1],
                    rgb_seq_static[:,1],
                    rgb_seq_gripper[:,1]
                ], dim=0)
                cond2 = torch.cat([
                    rgb_seq_static[:,0],
                    rgb_seq_gripper[:,0],
                    rgb_seq_gripper[:,0],
                    rgb_seq_static[:,0]
                ], dim=0)
                target2 = torch.cat([
                    rgb_seq_static[:,1],
                    rgb_seq_gripper[:,1],
                    rgb_seq_gripper[:,1],
                    rgb_seq_static[:,1]
                ], dim=0)
                corners = ['static', 'gripper', 'gripper', 'static']
                
                outputs = self.latent_motion_tokenizer(
                    cond_pixel_values1=cond1,
                    target_pixel_values1=target1,
                    cond_pixel_values2=cond2,
                    target_pixel_values2=target2,
                    return_recons_only=True,
                    corner=corners,
                    bs_per_gpu=self.bs_per_gpu
                )
                batch_size = rgb_seq_static.shape[0]
                outputs_static = {k: v[:batch_size] for k, v in outputs.items()}
                outputs_gripper = {k: v[batch_size:2*batch_size] for k, v in outputs.items()}
                outputs_diff_corner1 = {k: v[2*batch_size:3*batch_size] for k, v in outputs.items()}
                outputs_diff_corner2 = {k: v[3*batch_size:] for k, v in outputs.items()}
                
                
                recons_rgb_future_static = self.rgb_preprocessor.post_process(outputs_static["recons_pixel_values"]).detach().cpu()
                recons_rgb_future_diff_corner1 = self.rgb_preprocessor.post_process(outputs_diff_corner1["recons_pixel_values"]).detach().cpu()
                recons_rgb_future_diff_corner2 = self.rgb_preprocessor.post_process(outputs_diff_corner2["recons_pixel_values"]).detach().cpu()
                recons_rgb_future_gripper = self.rgb_preprocessor.post_process(outputs_gripper["recons_pixel_values"]).detach().cpu()
                
                gt_latent_motion_ids_static = outputs_static["indices"].detach().cpu()
                gt_latent_motion_ids_diff_corner1 = outputs_diff_corner1["indices"].detach().cpu()
                gt_latent_motion_ids_diff_corner2 = outputs_diff_corner2["indices"].detach().cpu()
                gt_latent_motion_ids_gripper = outputs_gripper["indices"].detach().cpu()
                
                orig_rgb_seq_static = self.rgb_preprocessor.post_process(rgb_seq_static).detach().cpu()
                orig_rgb_seq_gripper = self.rgb_preprocessor.post_process(rgb_seq_gripper).detach().cpu()
                for i in range(orig_rgb_seq_static.shape[0]):
                    visualize_latent_motion_reconstruction(
                        initial_frame=orig_rgb_seq_static[i,0],
                        next_frame=orig_rgb_seq_static[i,1],
                        recons_next_frame=recons_rgb_future_static[i],
                        latent_motion_ids=gt_latent_motion_ids_static[i],
                        path=os.path.join(visualization_dir, f"{self.process_index}-{i}-static.png")
                    )
                
                for i in range(orig_rgb_seq_static.shape[0]):
                    visualize_latent_motion_reconstruction(
                        initial_frame=orig_rgb_seq_gripper[i,0],
                        next_frame=orig_rgb_seq_gripper[i,1],
                        recons_next_frame=recons_rgb_future_diff_corner1[i],
                        latent_motion_ids=gt_latent_motion_ids_diff_corner1[i],
                        path=os.path.join(visualization_dir, f"{self.process_index}-{i}-diffcorner1.png")
                    )
                
                for i in range(orig_rgb_seq_static.shape[0]):
                    visualize_latent_motion_reconstruction(
                        initial_frame=orig_rgb_seq_static[i,0],
                        next_frame=orig_rgb_seq_static[i,1],
                        recons_next_frame=recons_rgb_future_diff_corner2[i],
                        latent_motion_ids=gt_latent_motion_ids_diff_corner2[i],
                        path=os.path.join(visualization_dir, f"{self.process_index}-{i}-diffcorner2.png")
                    )
                    
                for i in range(orig_rgb_seq_static.shape[0]):
                    visualize_latent_motion_reconstruction(
                        initial_frame=orig_rgb_seq_gripper[i,0],
                        next_frame=orig_rgb_seq_gripper[i,1],
                        recons_next_frame=recons_rgb_future_gripper[i],
                        latent_motion_ids=gt_latent_motion_ids_gripper[i],
                        path=os.path.join(visualization_dir, f"{self.process_index}-{i}-gripper.png")
                    )
                
        else:
            cond = torch.cat([
                rgb_seq_static[:, 0],
                rgb_seq_gripper[:, 0]
            ], dim=0)
            target = torch.cat([
                rgb_seq_static[:, 1],
                rgb_seq_gripper[:, 1]
            ], dim=0)
            corners = ['static', 'gripper']
            
            outputs = self.latent_motion_tokenizer(
                cond_pixel_values=cond,
                target_pixel_values=target,
                return_recons_only=True,
                corner = corners,
                bs_per_gpu=self.bs_per_gpu
            )
            
            batch_size = rgb_seq_static.shape[0]
            outputs_static = {k: v[:batch_size] for k, v in outputs.items()}
            outputs_gripper = {k: v[batch_size:] for k, v in outputs.items()}
            recons_rgb_future_static = self.rgb_preprocessor.post_process(outputs_static["recons_pixel_values"]).detach().cpu()
            recons_rgb_future_gripper = self.rgb_preprocessor.post_process(outputs_gripper["recons_pixel_values"]).detach().cpu()
            gt_latent_motion_ids_static = outputs_static["indices"].detach().cpu()
            gt_latent_motion_ids_gripper = outputs_gripper["indices"].detach().cpu()
            
            # orig_rgb_seq = orig_rgb_seq.detach().cpu()
            orig_rgb_seq_static = self.rgb_preprocessor.post_process(rgb_seq_static).detach().cpu()
            orig_rgb_seq_gripper = self.rgb_preprocessor.post_process(rgb_seq_gripper).detach().cpu()

            for i in range(orig_rgb_seq_static.shape[0]):
                visualize_latent_motion_reconstruction(
                    initial_frame=orig_rgb_seq_static[i,0],
                    next_frame=orig_rgb_seq_static[i,1],
                    recons_next_frame=recons_rgb_future_static[i],
                    latent_motion_ids=gt_latent_motion_ids_static[i],
                    path=os.path.join(visualization_dir, f"{self.process_index}-{i}-static.png")
                )
                visualize_latent_motion_reconstruction(
                    initial_frame=orig_rgb_seq_gripper[i,0],
                    next_frame=orig_rgb_seq_gripper[i,1],
                    recons_next_frame=recons_rgb_future_gripper[i],
                    latent_motion_ids=gt_latent_motion_ids_gripper[i],
                    path=os.path.join(visualization_dir, f"{self.process_index}-{i}-gripper.png")
                )


    def calculate_loss(self, batch, train):
        # image preprocessing
        rgb_seq_static = torch.cat([batch['rgb_initial_static'], batch['rgb_future_static']], dim=1)
        rgb_seq_static = self.rgb_preprocessor(rgb_seq_static, train=train)
        rgb_seq_gripper = torch.cat([batch['rgb_initial_gripper'], batch['rgb_future_gripper']], dim=1)
        rgb_seq_gripper = self.rgb_preprocessor(rgb_seq_gripper, train=train)
        cond = torch.cat([
            rgb_seq_static[:, 0],
            rgb_seq_gripper[:, 0]
        ], dim=0)
        target = torch.cat([
            rgb_seq_static[:, 1],
            rgb_seq_gripper[:, 1]
        ], dim=0)
        corners = ['static', 'gripper']
        
        outputs = self.latent_motion_tokenizer(
            cond_pixel_values=cond,
            target_pixel_values=target,
            corner=corners,
            bs_per_gpu=self.bs_per_gpu,
        )
        
        return outputs
    
    def calculate_3d_loss(self, batch, train):
        rgb_seq_static = torch.cat([batch['rgb_initial_static'], batch['rgb_future_static']], dim=1)
        rgb_seq_static = self.rgb_preprocessor(rgb_seq_static, train=train)
        rgb_seq_gripper = torch.cat([batch['rgb_initial_gripper'], batch['rgb_future_gripper']], dim=1)
        rgb_seq_gripper = self.rgb_preprocessor(rgb_seq_gripper, train=train)
        outputs = self.latent_motion_tokenizer(
            cond_pixel_values1=rgb_seq_static[:,0],
            target_pixel_values1=rgb_seq_static[:,1],
            cond_pixel_values2=rgb_seq_gripper[:,0],
            target_pixel_values2=rgb_seq_gripper[:,1],
        )
        
        return outputs

    def calculate_paired_loss(self, batch, train):
        rgb_seq_static = torch.cat([batch['rgb_initial_static'], batch['rgb_future_static']], dim=1)
        rgb_seq_static = self.rgb_preprocessor(rgb_seq_static, train=train)
        rgb_seq_gripper = torch.cat([batch['rgb_initial_gripper'], batch['rgb_future_gripper']], dim=1)
        rgb_seq_gripper = self.rgb_preprocessor(rgb_seq_gripper, train=train)
        cond1 = torch.cat([
            rgb_seq_static[:,0],
            rgb_seq_gripper[:,0],
            rgb_seq_static[:,0],
            rgb_seq_gripper[:,0]
        ], dim=0)
        target1 = torch.cat([
            rgb_seq_static[:,1],
            rgb_seq_gripper[:,1],
            rgb_seq_static[:,1],
            rgb_seq_gripper[:,1]
        ], dim=0)
        cond2 = torch.cat([
            rgb_seq_static[:,0],
            rgb_seq_gripper[:,0],
            rgb_seq_gripper[:,0],
            rgb_seq_static[:,0]
        ], dim=0)
        target2 = torch.cat([
            rgb_seq_static[:,1],
            rgb_seq_gripper[:,1],
            rgb_seq_gripper[:,1],
            rgb_seq_static[:,1]
        ], dim=0)
        corners = ['static', 'gripper', 'gripper', 'static']
        
        outputs = self.latent_motion_tokenizer(
            cond_pixel_values1=cond1,
            target_pixel_values1=target1,
            cond_pixel_values2=cond2,
            target_pixel_values2=target2,
            corner=corners,
            bs_per_gpu=self.bs_per_gpu
        )
        return outputs
    
    def calculate_paired_loss_lm_restrict(self, batch, train, weight=0.1):
        rgb_seq_static = torch.cat([batch['rgb_initial_static'], batch['rgb_future_static']], dim=1)
        rgb_seq_static = self.rgb_preprocessor(rgb_seq_static, train=train)
        rgb_seq_gripper = torch.cat([batch['rgb_initial_gripper'], batch['rgb_future_gripper']], dim=1)
        rgb_seq_gripper = self.rgb_preprocessor(rgb_seq_gripper, train=train)
        cond1 = torch.cat([
            rgb_seq_static[:,0],
            rgb_seq_gripper[:,0],
            rgb_seq_static[:,0],
            rgb_seq_gripper[:,0]
        ], dim=0)
        target1 = torch.cat([
            rgb_seq_static[:,1],
            rgb_seq_gripper[:,1],
            rgb_seq_static[:,1],
            rgb_seq_gripper[:,1]
        ], dim=0)
        cond2 = torch.cat([
            rgb_seq_static[:,0],
            rgb_seq_gripper[:,0],
            rgb_seq_gripper[:,0],
            rgb_seq_static[:,0]
        ], dim=0)
        target2 = torch.cat([
            rgb_seq_static[:,1],
            rgb_seq_gripper[:,1],
            rgb_seq_gripper[:,1],
            rgb_seq_static[:,1]
        ], dim=0)
        corners = ['static', 'gripper', 'gripper', 'static']
        
        outputs = self.latent_motion_tokenizer(
            cond_pixel_values1=cond1,
            target_pixel_values1=target1,
            cond_pixel_values2=cond2,
            target_pixel_values2=target2,
            return_latent_motion_embeddings=True,
            corner=corners,
            bs_per_gpu=self.bs_per_gpu
        )
        bs = rgb_seq_static.shape[0]
        lm1 = outputs['latent_motion_embeddings'][:bs]
        lm2 = outputs['latent_motion_embeddings'][bs:2*bs]
        lm_restrict = F.mse_loss(lm1, lm2)* weight
        outputs.pop('latent_motion_embeddings')
        # print(outputs['loss'], lm_restrict)
        outputs['lm_restrict'] = lm_restrict
        outputs['loss'] += lm_restrict
        return outputs
        
        
    def calculate_paired_loss_half(self, batch, train):
        rgb_seq_static = torch.cat([batch['rgb_initial_static'], batch['rgb_future_static']], dim=1)
        rgb_seq_static = self.rgb_preprocessor(rgb_seq_static, train=train)
        rgb_seq_gripper = torch.cat([batch['rgb_initial_gripper'], batch['rgb_future_gripper']], dim=1)
        rgb_seq_gripper = self.rgb_preprocessor(rgb_seq_gripper, train=train)
        cond1 = torch.cat([
            rgb_seq_static[:,0],
            rgb_seq_static[:,0]
        ], dim=0)
        target1 = torch.cat([
            rgb_seq_static[:,1],
            rgb_seq_static[:,1]
        ], dim=0)
        cond2 = torch.cat([
            rgb_seq_static[:,0],
            rgb_seq_gripper[:,0]
        ], dim=0)
        target2 = torch.cat([
            rgb_seq_static[:,1],
            rgb_seq_gripper[:,1]
        ], dim=0)
        corners = ['static', 'gripper']
        
        outputs = self.latent_motion_tokenizer(
            cond_pixel_values1=cond1,
            target_pixel_values1=target1,
            cond_pixel_values2=cond2,
            target_pixel_values2=target2,
            corner=corners,
            bs_per_gpu=self.bs_per_gpu
        )
        return outputs    
        


    def log(self, log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step):
        for key in log_loss:
            log_loss[key] = self.accelerator.gather_for_metrics(log_loss[key]).mean()
        for key in eval_log_loss:
            eval_log_loss[key] = self.accelerator.gather_for_metrics(eval_log_loss[key]).mean()
        load_pecnt = torch.tensor(cum_load_time / (time()-clock)).to(self.device)
        load_pecnt = self.accelerator.gather_for_metrics(load_pecnt).mean()
        fps = (self.bs_per_gpu*self.print_steps*2) / (time()-clock)
        fps = self.accelerator.gather_for_metrics(torch.tensor(fps).to(self.device)).sum()

        text = 'Train Epoch: {} [{}/{} ({:.0f}%)] FPS:{:.5f} Load Pertentage:{:.5f} LR:{}'.format(
            epoch, 
            batch_idx * self.bs_per_gpu * self.accelerator.num_processes, 
            len(self.train_prefetcher), 
            100. * batch_idx * self.bs_per_gpu * self.accelerator.num_processes / len(self.train_prefetcher),
            fps,
            load_pecnt,
            self.scheduler.get_last_lr()[0],
        )
        for key in log_loss:
            text = text + ' {}: {:.5f}'.format(key, log_loss[key])
        for key in eval_log_loss:
            text = text + ' eval_{}: {:.5f}'.format(key, eval_log_loss[key])
        self.print(text)
        if self.is_main:
            for key in log_loss:
                self.writer.add_scalar(key, log_loss[key], step)
            for key in eval_log_loss:
                self.writer.add_scalar('eval_'+key, eval_log_loss[key], step)
            self.writer.add_scalar("learning rate", self.scheduler.get_last_lr()[0], step)
            self.writer.add_scalar("FPS", fps, step)
            self.writer.add_scalar("loading time in total time", load_pecnt, step)

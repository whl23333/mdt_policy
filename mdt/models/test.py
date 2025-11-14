import torch

ckpt = torch.load('/data/250010208/whl/results/mdt_policy/logs/runs/2025-10-27/22-31-05/saved_models/epoch=42.ckpt', map_location='cpu')
state_dict = ckpt['state_dict']
print(state_dict.keys())
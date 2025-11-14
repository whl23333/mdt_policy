import contextlib
import logging
from pathlib import Path
from typing import Union
import importlib

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
import pyhash
import torch
from hydra.core.global_hydra import GlobalHydra

from mdt.utils.utils import add_text, format_sftp_path

hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)


def load_class(name):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_evaluation_checkpoint(cfg):
    epoch = cfg.epoch_to_load if "epoch_to_load" in cfg else -1
    overwrite_cfg = cfg.overwrite_module_cfg if "overwrite_module_cfg" in cfg else {}
    module_path = str(Path(cfg.module_path).expanduser())
    pl_module = load_pl_module_from_checkpoint(
        module_path,
        epoch=epoch,
        overwrite_cfg=overwrite_cfg,
    ).cuda()
    return pl_module


# def get_checkpoint_i_from_dir(dir, i: int = -1):
#     ckpt_paths = list(dir.rglob("*.ckpt"))
#     if i == -1:
#         for ckpt_path in ckpt_paths:
#             if ckpt_path.stem == "last":
#                 return ckpt_path

#     # Search for ckpt of epoch i
#     for ckpt_path in ckpt_paths:
#         split_path = str(ckpt_path).split("_")
#         for k, word in enumerate(split_path):
#             if word == "epoch":
#                 if int(split_path[k + 1]) == i:
#                     return ckpt_path

#     sorted(ckpt_paths, key=lambda f: f.stat().st_mtime)
#     return ckpt_paths[i]

def get_checkpoint_i_from_dir(dir, i: int = -1):
    ckpt_paths = list(dir.rglob("*.ckpt"))
    if i == -1:
        for ckpt_path in ckpt_paths:
            if ckpt_path.stem == "last":
                return ckpt_path

    # Search for ckpt of epoch i
    for ckpt_path in ckpt_paths:
        split_path = str(ckpt_path).split("=")  # 修改为支持 "epoch=40" 格式
        for k, word in enumerate(split_path):
            if word.endswith("epoch"):
                if int(split_path[k + 1].split(".")[0]) == i:  # 提取数字部分
                    return ckpt_path

    ckpt_paths = sorted(ckpt_paths, key=lambda f: f.stat().st_mtime)
    return ckpt_paths[i]


def get_config_from_dir(dir):
    dir = Path(dir)
    config_yaml = list(dir.rglob("*.hydra/config.yaml"))[0]
    return OmegaConf.load(config_yaml)


def load_pl_module_from_checkpoint(
    filepath: Union[Path, str],
    epoch: int = 1,
    overwrite_cfg: dict = {},
    use_ema_weights: bool = False
):
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if filepath.is_dir():
        filedir = filepath
        ckpt_path = get_checkpoint_i_from_dir(dir=filedir, i=epoch)
        print("Resolved checkpoint path:", ckpt_path)
    elif filepath.is_file():
        assert filepath.suffix == ".ckpt", "File must have .ckpt extension"
        ckpt_path = filepath
        filedir = filepath.parents[0]
    else:
        raise ValueError(f"not valid file path: {str(filepath)}")
    config = get_config_from_dir(filedir)
    class_name = config.model.pop("_target_")
    if "_recursive_" in config.model:
        del config.model["_recursive_"]
    print(f"class_name {class_name}")
    module_class = load_class(class_name)
    print(f"Loading model from {ckpt_path}")
    load_cfg = {**config.model, **overwrite_cfg}
    # Use strict=False to tolerate non-critical key mismatches from external/frozen modules
    model = module_class.load_from_checkpoint(ckpt_path, strict=False, **load_cfg)
    k_list = []
    for k, v in model.named_parameters():
        k_list.append(k)
    root_k = set([kk.split(".")[0] for kk in k_list])
    print(f"model.latent_motion_pred: {model.latent_motion_pred}")
    print("Root keys in model:", root_k)

    # Summarize state_dict mismatches (strict=False): missing/unexpected/shape mismatches
    try:
        ckpt_blob = torch.load(ckpt_path, map_open_rl=None) if False else torch.load(ckpt_path, map_location="cpu")
        ckpt_state = ckpt_blob.get("state_dict", ckpt_blob) if isinstance(ckpt_blob, dict) else {}
        if isinstance(ckpt_state, dict):
            model_state = model.state_dict()
            model_keys = set(model_state.keys())
            ckpt_keys = set(ckpt_state.keys())

            unexpected_keys = sorted(list(ckpt_keys - model_keys))
            missing_keys = sorted(list(model_keys - ckpt_keys))
            missing_root_keys = set([k.split(".")[0] for k in missing_keys])

            shape_mismatch = []
            matched = 0
            for k in ckpt_keys & model_keys:
                v_ckpt = ckpt_state[k]
                v_model = model_state[k]
                ckpt_shape = getattr(v_ckpt, "shape", None)
                model_shape = getattr(v_model, "shape", None)
                if ckpt_shape == model_shape:
                    matched += 1
                else:
                    shape_mismatch.append((k, ckpt_shape, model_shape))

            total_model = len(model_state)
            coverage = matched / total_model if total_model else 1.0

            def _preview(seq, n=20):
                return list(seq)[:n]

            print("=== Checkpoint load summary (strict=False) ===")
            print(f"Model keys: {total_model} | Ckpt keys: {len(ckpt_state)} | Matched (name+shape): {matched} ({coverage:.1%})")
            print(f"Missing root keys in ckpt: {len(missing_root_keys)} | showing all:")
            print(missing_root_keys)
            print(f"Unexpected keys in ckpt: {len(unexpected_keys)} | showing up to 20:")
            for k in _preview(unexpected_keys):
                print(f"  + {k}")
            print(f"Shape mismatches: {len(shape_mismatch)} | showing up to 20:")
            for k, cs, ms in _preview(shape_mismatch):
                print(f"  * {k}: ckpt{tuple(cs) if cs is not None else cs} vs model{tuple(ms) if ms is not None else ms}")
            print("============================================")
    except Exception as e:
        print(f"Warning: failed to summarize state_dict mismatches: {e}")
    # Load EMA weights if they exist and the flag is set (safely)
    if use_ema_weights:
        try:
            checkpoint_data = torch.load(ckpt_path, map_location="cpu")
            ema_loaded = False
            
            # Helper to compute similarity between base and candidate EMA by L2 diff ratio
            def _diff_metrics(base_sd: dict, cand_sd: dict):
                ratios = []
                try:
                    from statistics import median
                except Exception:
                    median = lambda x: sorted(x)[len(x)//2] if x else None
                for k, b in base_sd.items():
                    c = cand_sd.get(k)
                    if not isinstance(b, torch.Tensor) or not isinstance(c, torch.Tensor):
                        continue
                    if b.shape != getattr(c, 'shape', None):
                        continue
                    if b.numel() == 0:
                        continue
                    bn = torch.linalg.norm(b.float()).item()
                    dn = torch.linalg.norm((c.float() - b.float())).item()
                    ratios.append(dn / (bn + 1e-12))
                if not ratios:
                    return {"count": 0, "mean": None, "median": None, "p95": None}
                ratios_sorted = sorted(ratios)
                p95 = ratios_sorted[int(0.95 * (len(ratios_sorted) - 1))]
                return {
                    "count": len(ratios),
                    "mean": sum(ratios) / len(ratios),
                    "median": median(ratios),
                    "p95": p95,
                }
            # Try common locations for EMA state dicts
            ema_candidates = []
            if isinstance(checkpoint_data, dict):
                # 1) Top-level common keys
                for k in [
                    "ema_state_dict",
                    "state_dict_ema",
                    "model_ema_state_dict",
                ]:
                    if k in checkpoint_data and isinstance(checkpoint_data[k], dict):
                        ema_candidates.append(checkpoint_data[k])
                # 2) Under callbacks (e.g., pl EMA callback)
                cb = checkpoint_data.get("callbacks") if isinstance(checkpoint_data.get("callbacks"), dict) else None
                if cb:
                    # Base state_dict for order-based fallback
                    base_sd = checkpoint_data.get("state_dict", {}) if isinstance(checkpoint_data, dict) else {}
                    base_names = tuple(base_sd.keys()) if isinstance(base_sd, dict) else None
                    # Explicit EMA callback handling with base-order detection
                    if "EMA" in cb and isinstance(cb["EMA"], dict):
                        ema_cb = cb["EMA"]
                        # If only ema_weights list exists without names, align using base state_dict key order
                        only_list = isinstance(ema_cb.get("ema_weights"), (list, tuple)) and not any(
                            isinstance(ema_cb.get(nk), (list, tuple)) for nk in ["param_names", "names", "parameter_names", "keys"]
                        )
                        if isinstance(ema_cb.get("ema_weights"), dict):
                            only_list = False
                            ema_weights = ema_cb["ema_weights"]
                            missing_keys, unexpected_keys = model.load_state_dict(ema_weights, strict=False)
                            missing_root_keys = set([k.split(".")[0] for k in missing_keys])
                            print(f"Loaded EMA weights from checkpoint EMA callback dict | missing root keys: {missing_root_keys} | unexpected keys: {len(unexpected_keys)}")
                            ema_loaded = True
                            
                        if only_list and base_names and len(ema_cb["ema_weights"]) == len(base_names):
                            ema_sd = dict(zip(base_names, ema_cb["ema_weights"]))
                            # Print detection and similarity
                            print("Detected EMA weights without names; using base state_dict key order mapping.")
                            metrics = _diff_metrics(base_sd, ema_sd) if isinstance(base_sd, dict) else {"count": 0}
                            if metrics.get("count", 0) > 0:
                                print(
                                    f"EMA mapping: base-order | L2 ratio mean={metrics['mean']:.4e}, median={metrics['median']:.4e}, p95={metrics['p95']:.4e} over {metrics['count']} params"
                                )
                            # Load filtered subset matching model
                            model_state = model.state_dict()
                            filtered = {k: v for k, v in ema_sd.items() if k in model_state and getattr(v, "shape", None) == model_state[k].shape}
                            print(f"Attempting EMA load (base-order): {len(filtered)} params")
                            load_res = model.load_state_dict(filtered, strict=False)
                            print(f"EMA load result -> missing: {len(load_res.missing_keys)}, unexpected: {len(load_res.unexpected_keys)}")
                            ema_loaded = True
                        # Also handle nested 'ema' dict with list-only weights
                        if not ema_loaded and isinstance(ema_cb.get("ema"), dict):
                            nested = ema_cb["ema"]
                            nested_weights = None
                            if isinstance(nested.get("ema_weights"), (list, tuple)):
                                nested_weights = nested["ema_weights"]
                            elif isinstance(nested.get("weights"), (list, tuple)):
                                nested_weights = nested["weights"]
                            only_list_nested = nested_weights is not None and not any(
                                isinstance(nested.get(nk), (list, tuple)) for nk in ["param_names", "names", "parameter_names", "keys"]
                            )
                            if only_list_nested and base_names and len(nested_weights) == len(base_names):
                                ema_sd = dict(zip(base_names, nested_weights))
                                print("Detected nested EMA weights without names; using base state_dict key order mapping.")
                                metrics = _diff_metrics(base_sd, ema_sd) if isinstance(base_sd, dict) else {"count": 0}
                                if metrics.get("count", 0) > 0:
                                    print(
                                        f"EMA mapping: base-order (nested) | L2 ratio mean={metrics['mean']:.4e}, median={metrics['median']:.4e}, p95={metrics['p95']:.4e} over {metrics['count']} params"
                                    )
                                model_state = model.state_dict()
                                filtered = {k: v for k, v in ema_sd.items() if k in model_state and getattr(v, "shape", None) == model_state[k].shape}
                                print(f"Attempting EMA load (base-order nested): {len(filtered)} params")
                                load_res = model.load_state_dict(filtered, strict=False)
                                print(f"EMA load result -> missing: {len(load_res.missing_keys)}, unexpected: {len(load_res.unexpected_keys)}")
                                ema_loaded = True
                    # # Explicit EMA callback handling
                    # if "EMA" in cb and isinstance(cb["EMA"], dict):
                    #     ema_cb = cb["EMA"]
                    #     # Direct weights + names on EMA callback
                    #     if "ema_weights" in ema_cb:
                    #         weights = ema_cb.get("ema_weights")
                    #         name_key_candidates = ["param_names", "names", "parameter_names", "keys"]
                    #         names = None
                    #         for nk in name_key_candidates:
                    #             if isinstance(ema_cb.get(nk), (list, tuple)):
                    #                 names = ema_cb.get(nk)
                    #                 break
                    #         if isinstance(weights, (list, tuple)) and isinstance(names, (list, tuple)) and len(weights) == len(names):
                    #             ema_candidates.append(dict(zip(names, weights)))
                    #         elif isinstance(weights, (list, tuple)) and names is None:
                    #             print("Warning: EMA callback has 'ema_weights' but no names list; skipping unsafe EMA mapping.")
                    #     # Nested 'ema' dict
                    #     if isinstance(ema_cb.get("ema"), dict):
                    #         nested = ema_cb["ema"]
                    #         if "ema_weights" in nested or "weights" in nested:
                    #             weights = nested.get("ema_weights", nested.get("weights"))
                    #             name_key_candidates = ["param_names", "names", "parameter_names", "keys"]
                    #             names = None
                    #             for nk in name_key_candidates:
                    #                 if isinstance(nested.get(nk), (list, tuple)):
                    #                     names = nested.get(nk)
                    #                     break
                    #             if isinstance(weights, (list, tuple)) and isinstance(names, (list, tuple)) and len(weights) == len(names):
                    #                 ema_candidates.append(dict(zip(names, weights)))
                    # Generic callbacks
                    # for v in cb.values():
                    #     if isinstance(v, dict):
                    #         for k in ["ema_state_dict", "state_dict", "model_state_dict"]:
                    #             if k in v and isinstance(v[k], dict):
                    #                 ema_candidates.append(v[k])
                    #         # Fallback: list of tensors + names
                    #         if "ema_weights" in v:
                    #             weights = v.get("ema_weights")
                    #             # Accept several possible keys for parameter names
                    #             name_key_candidates = ["param_names", "names", "parameter_names", "keys"]
                    #             names = None
                    #             for nk in name_key_candidates:
                    #                 if isinstance(v.get(nk), (list, tuple)):
                    #                     names = v.get(nk)
                    #                     break
                    #             if isinstance(weights, (list, tuple)) and isinstance(names, (list, tuple)) and len(weights) == len(names):
                    #                 ema_candidates.append(dict(zip(names, weights)))
                    #             elif isinstance(weights, (list, tuple)) and names is None:
                    #                 print("Warning: Found 'ema_weights' in callbacks but no accompanying names list (param_names/names). Skipping unsafe EMA mapping.")
                    #         # Nested 'ema' dicts in generic callbacks
                    #         if isinstance(v.get("ema"), dict):
                    #             nested = v["ema"]
                    #             if "ema_weights" in nested or "weights" in nested:
                    #                 weights = nested.get("ema_weights", nested.get("weights"))
                    #                 name_key_candidates = ["param_names", "names", "parameter_names", "keys"]
                    #                 names = None
                    #                 for nk in name_key_candidates:
                    #                     if isinstance(nested.get(nk), (list, tuple)):
                    #                         names = nested.get(nk)
                    #                         break
                    #                 if isinstance(weights, (list, tuple)) and isinstance(names, (list, tuple)) and len(weights) == len(names):
                    #                     ema_candidates.append(dict(zip(names, weights)))

            # # Try to load the first compatible EMA state_dict
            # model_state = model.state_dict()
            # for ema_sd in ema_candidates:
            #     if not isinstance(ema_sd, dict):
            #         continue
            #     # Filter to keys present in model and matching shape
            #     filtered = {k: v for k, v in ema_sd.items() if k in model_state and getattr(v, "shape", None) == model_state[k].shape}
            #     if not filtered:
            #         continue
            #     missing_before = set(model_state.keys()) - set(filtered.keys())
            #     unexpected_before = set(filtered.keys()) - set(model_state.keys())
            #     print(f"Attempting EMA load: {len(filtered)} params | missing ignored: {len(missing_before)} | unexpected ignored: {len(unexpected_before)}")
            #     load_res = model.load_state_dict(filtered, strict=False)
            #     print(f"EMA load result -> missing: {len(load_res.missing_keys)}, unexpected: {len(load_res.unexpected_keys)}")
            #     ema_loaded = True
            #     break

            if not ema_loaded:
                print("Warning: No compatible EMA weights found or shapes mismatched; skipping EMA load.")
        except Exception as e:
            print(f"Warning: Failed to load EMA weights safely: {e}. Proceeding without EMA.")

    print(f"Finished loading model {ckpt_path}")
    return model

# def load_pl_module_from_checkpoint(
#     filepath: Union[Path, str],
#     epoch: int = 1,
#     overwrite_cfg: dict = {},
#     use_ema_weights: bool = False
# ):
#     if isinstance(filepath, str):
#         filepath = Path(filepath)

#     if filepath.is_dir():
#         filedir = filepath
#         ckpt_path = get_checkpoint_i_from_dir(dir=filedir, i=epoch)
#     elif filepath.is_file():
#         assert filepath.suffix == ".ckpt", "File must have .ckpt extension"
#         ckpt_path = filepath
#         filedir = filepath.parents[0]
#     else:
#         raise ValueError(f"not valid file path: {str(filepath)}")
#     config = get_config_from_dir(filedir)
#     class_name = config.model.pop("_target_")
#     if "_recursive_" in config.model:
#         del config.model["_recursive_"]
#     print(f"class_name {class_name}")
#     module_class = load_class(class_name)
#     print(f"Loading model from {ckpt_path}")
#     load_cfg = {**config.model, **overwrite_cfg}
#     model = module_class.load_from_checkpoint(ckpt_path, **load_cfg)
#      # Load EMA weights if they exist and the flag is set
#     if use_ema_weights:
#         checkpoint_data = torch.load(ckpt_path)
#         if "ema_weights" in checkpoint_data['callbacks']['EMA']:
#             ema_weights_list = checkpoint_data['callbacks']['EMA']['ema_weights']
            
#             # Convert list of tensors to a state_dict format
#             ema_weights_dict = {name: ema_weights_list[i] for i, (name, _) in enumerate(model.named_parameters())}
            
#             model.load_state_dict(ema_weights_dict)
#             print("Successfully loaded EMA weights from checkpoint!")
#         else:
#             print("Warning: No EMA weights found in checkpoint!")

#     print(f"Finished loading model {ckpt_path}")
#     return model



def get_default_model_and_env(train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None, device_id=0):
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    cfg = OmegaConf.load(train_cfg_path)
    lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf/datamodule/datasets")
    # we don't want to use shm dataset for evaluation
    datasets_cfg = hydra.compose("vision_lang.yaml", overrides=["lang_dataset.lang_folder=" + lang_folder])
    # since we don't use the trainer during inference, manually set up data_module
    cfg.datamodule.datasets = datasets_cfg
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["lang"]
    device = torch.device(f"cuda:{device_id}")

    if lang_embeddings is None:
        lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)

    if env is None:
        rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "conf/callbacks/rollout/default.yaml")
        env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)

    checkpoint = format_sftp_path(checkpoint)
    print(f"Loading model from {checkpoint}")
    
    # new stuff
    epoch = cfg.epoch_to_load if "epoch_to_load" in cfg else -1
    overwrite_cfg = cfg.overwrite_module_cfg if "overwrite_module_cfg" in cfg else {}
    module_path = str(Path(train_folder).expanduser())
    model = load_pl_module_from_checkpoint(
        module_path,
        epoch=epoch,
        overwrite_cfg=overwrite_cfg,
    )
    # model = Hulc.load_from_checkpoint(checkpoint)
    model.freeze()
    if cfg.model.action_decoder.get("load_action_bounds", False):
        model.action_decoder._setup_action_bounds(cfg.datamodule.root_data_dir, None, None, True)
    model = model.cuda(device)
    print("Successfully loaded model.")

    return model, env, data_module, lang_embeddings


def get_default_beso_and_env(train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None, device_id=0, eval_cfg_overwrite={}):
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    def_cfg = OmegaConf.load(train_cfg_path)
    eval_override_cfg = OmegaConf.create(eval_cfg_overwrite)
    cfg = OmegaConf.merge(def_cfg, eval_override_cfg)
    lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf/datamodule/datasets")
    # we don't want to use shm dataset for evaluation
    # GlobalHydra.instance().clear()
    # datasets_cfg = hydra.initialize("datamodule/datasets/vision_lang.yaml")
    # since we don't use the trainer during inference, manually set up data_module
    # cfg.datamodule.datasets = datasets_cfg
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["lang"]
    mean, std = data_module.get_normalize_val('rgb_static')
    logger.info(f"Normalization parameters for 'rgb_static': mean={mean}, std={std}")
    if device_id != 'cpu':
        device = torch.device(f"cuda:{device_id}")
    else:
        device = 'cpu'

    if lang_embeddings is None:
        lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)

    if env is None:
        rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "conf/callbacks/rollout/default.yaml")
        env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)

    checkpoint = format_sftp_path(checkpoint)
    print(f"Loading model from {checkpoint}")
    
    # new stuff
    epoch = cfg.epoch_to_load if "epoch_to_load" in cfg else -1
    overwrite_cfg = cfg.overwrite_module_cfg if "overwrite_module_cfg" in cfg else {}
    module_path = str(Path(train_folder).expanduser())
    model = load_pl_module_from_checkpoint(
        module_path,
        epoch=epoch,
        overwrite_cfg=overwrite_cfg,
        use_ema_weights=True
    )
    model.freeze()
    model = model.cuda(device)
    print("Successfully loaded model.")

    return model, env, data_module, lang_embeddings


def join_vis_lang(img, lang_text):
    """Takes as input an image and a language instruction and visualizes them with cv2"""
    img = img[:, :, ::-1].copy()
    img = cv2.resize(img, (500, 500))
    add_text(img, lang_text)
    cv2.imshow("simulation cam", img)
    cv2.waitKey(1)


class LangEmbeddings:
    def __init__(self, val_dataset_path, lang_folder, device=torch.device("cuda:0")):
        embeddings = np.load(Path(val_dataset_path) / lang_folder / "embeddings.npy", allow_pickle=True).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}
        self.device = device

    def get_lang_goal(self, task):
        return {"lang": torch.from_numpy(self.lang_embeddings[task]).to(self.device).squeeze(0).float()}


def imshow_tensor(window, img_tensor, wait=0, resize=True, keypoints=None, text=None):
    img_tensor = img_tensor.squeeze()
    img = np.transpose(img_tensor.cpu().numpy(), (1, 2, 0))
    img = np.clip(((img / 2) + 0.5) * 255, 0, 255).astype(np.uint8)

    if keypoints is not None:
        key_coords = np.clip(keypoints * 200 + 100, 0, 200)
        key_coords = key_coords.reshape(-1, 2)
        cv_kp1 = [cv2.KeyPoint(x=pt[1], y=pt[0], _size=1) for pt in key_coords]
        img = cv2.drawKeypoints(img, cv_kp1, None, color=(255, 0, 0))

    if text is not None:
        add_text(img, text)

    if resize:
        cv2.imshow(window, cv2.resize(img[:, :, ::-1], (500, 500)))
    else:
        cv2.imshow(window, img[:, :, ::-1])
    cv2.waitKey(wait)


def print_task_log(demo_task_counter, live_task_counter, mod):
    print()
    logger.info(f"Modality: {mod}")
    for task in demo_task_counter:
        logger.info(
            f"{task}: SR = {(live_task_counter[task] / demo_task_counter[task]) * 100:.0f}%"
            + f" |  {live_task_counter[task]} of {demo_task_counter[task]}"
        )
    s = sum(demo_task_counter.values())
    success_rate = (sum(live_task_counter.values()) / s if s > 0 else 0) * 100
    logger.info(f"Average Success Rate {mod} = {success_rate:.0f}%")
    logger.info(
        f"Success Rates averaged throughout classes = {np.mean([live_task_counter[task] / demo_task_counter[task] for task in demo_task_counter]) * 100:.0f}%"
    )


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_env_state_for_initial_condition(initial_condition):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    # we want to have a "deterministic" random seed for each initial condition
    seed = hasher(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs

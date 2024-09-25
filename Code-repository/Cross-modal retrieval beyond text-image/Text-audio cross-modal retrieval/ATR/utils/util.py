
"""Code based on the implementation of "audio-retrieval":
https://github.com/oncescuandreea/audio-retrieval
"""

import os
import copy
import json
import uuid
import random
import logging
import itertools
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch
import psutil
import humanize
from PIL import Image
from typeguard import typechecked


@typechecked
def filter_cmd_args(cmd_args: List[str], remove: List[str]) -> List[str]:
    drop = []
    for key in remove:
        if key not in cmd_args:
            continue
        pos = cmd_args.index(key)
        drop.append(pos)
        if len(cmd_args) > (pos + 1) and not cmd_args[pos + 1].startswith("--"):
            drop.append(pos + 1)
    for pos in sorted(drop, reverse=True):
        cmd_args.pop(pos)
    return cmd_args


@typechecked
def get_short_uuid() -> str:
    """Return a 7 alpha-numeric character random string.  We could use the full uuid()
    for better uniqueness properties, but it makes the filenames long and its not
    needed for our purpose (simply grouping experiments that were run with the same
    configuration).
    """
    return str(uuid.uuid4()).split("-")[0]


@typechecked
def parse_grid(x: str, evaluation: str='train') -> Dict[str, List[str]]:
    """Parse compact command line strings of the form:
        --key1 val_a|val_b --key2 val_c|val_d

    (here a vertical bar represents multiple values)

    into a grid of separate strings e.g:
        --key1 val_a --key2 val_c
        --key1 val_a --key2 val_d
        --key1 val_b --key2 val_c
        --key1 val_b --key2 val_d

    """
    args = x.split(" ")
    group_id = get_short_uuid()
    grid_opts, parsed = {}, []
    for ii, token in enumerate(args):
        if "|" in token:
            grid_opts[ii] = token.split("|")
    grid_idx, grid_vals = [], []
    for ii, val in grid_opts.items():
        grid_idx.append(ii)
        grid_vals.append(val)
    grid_vals = list(itertools.product(*grid_vals))
    for cfg in grid_vals:
        base = copy.deepcopy(args)
        for ii, val in zip(grid_idx, cfg):
            base[ii] = val
        if evaluation == 'train':
            base.append(f"--group_id {group_id}")
        else:
            pass
        parsed.append(" ".join(base))
    return {group_id: parsed}


@typechecked
def set_seeds(seed: int):
    """Set seeds for randomisation libraries.

    Args:
        seed: the seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def memory_summary():
    vmem = psutil.virtual_memory()
    msg = (
        f">>> Currently using {vmem.percent}% of system memory "
        f"{humanize.naturalsize(vmem.used)}/{humanize.naturalsize(vmem.available)}"
    )
    print(msg)


def flatten_dict(x, keysep="-"):
    flat_dict = {}
    for key, val in x.items():
        if isinstance(val, dict):
            flat_subdict = flatten_dict(val)
            flat_dict.update({f"{key}{keysep}{subkey}": subval
                              for subkey, subval in flat_subdict.items()})
        else:
            flat_dict.update({key: val})
    return flat_dict


def expert_tensor_storage(experts, feat_aggregation):
    
    expert_storage = {"fixed": set(), "variable": set(), "flaky": set()}
    # fixed_sz_experts, variable_sz_experts, flaky_experts = set(), set(), set()
    for expert, config in feat_aggregation.items():
        if config["temporal"] in {"vlad"}:
            expert_storage["variable"].add(expert)
        elif all([x in {"avg", "max", "ent", "std"} for x in config["temporal"].split("-")]):
            expert_storage["fixed"].add(expert)
        else:
            raise ValueError(f"unknown temporal strategy: {config['temporal']}")
        # some "flaky" experts are only available for a fraction of videos - we need
        # to pass this information (in the form of indices) into the network for any
        # experts present in the current dataset
        if config.get("flaky", False):
            expert_storage["flaky"].add(expert)
    
    # we only allocate storage for experts used by the current dataset
    for key, value in expert_storage.items():
        expert_storage[key] = value.intersection(set(experts))
    return expert_storage


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def path2str(x):
    """Recursively convert pathlib objects to strings to enable serialization"""
    for key, val in x.items():
        if isinstance(val, dict):
            path2str(val)
        elif isinstance(val, Path):
            x[key] = str(val)


def write_json(content, fname, paths2strs=False):
    if paths2strs:
        path2str(content)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in itertools.repeat(data_loader):
        yield from loader


def compute_trn_config(config, logger=None):
    trn_config = {}
    feat_agg = config["data_loader"]["args"]["feat_aggregation"]
    for static_expert in feat_agg.keys():
        if static_expert in feat_agg:
            if "trn_seg" in feat_agg[static_expert].keys():
                trn_config[static_expert] = feat_agg[static_expert]["trn_seg"]
    return trn_config


@typechecked
def compute_dims(
        config,
        logger: logging.Logger = None,
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, int], int]:
    if logger is None:
        logger = config.get_logger('utils')

    experts = config["experts"]
    # TODO(Samuel): clean up the logic since it's a little convoluted
    ordered = sorted(config["experts"]["modalities"])

#    if experts["drop_feats"]:
#        to_drop = experts["drop_feats"].split(",")
#        logger.info(f"dropping: {to_drop}")
#        ordered = [x for x in ordered if x not in to_drop]

    feat_agg = config["data_loader"]["args"]["feat_aggregation"]
    dims = []
    arch_args = config["arch"]["args"]
    vlad_clusters = arch_args["vlad_clusters"]
#    msg = f"It is not valid to use both the `use_ce` and `mimic_ce_dims` options"
#    assert not (arch_args["use_ce"] and arch_args.get("mimic_ce_dims", False)), msg
    for expert in ordered:
        temporal = feat_agg[expert]["temporal"]
        if expert == "lms" and temporal in ["vlad","rvlad","vf","meanP","maxP"]:
            in_dim, out_dim = 128 * vlad_clusters["lms"], 128
        elif expert == "efficient_latent" and temporal in ["vlad","rvlad","vf","meanP","maxP"]:
            
            in_dim, out_dim = 1408 * vlad_clusters["vggish"], 1408

        elif expert == "vggish" and temporal in ["vlad","rvlad","vf","meanP","maxP"]:
            in_dim, out_dim = 128 * vlad_clusters["vggish"], 128
        elif expert == "panns_cnn14" and temporal in ["vlad","rvlad","vf","meanP","maxP"]:# and temporal == "vlad":
            in_dim, out_dim = 2048 * vlad_clusters["panns_cnn14"], 2048
        elif expert == "vggsound" and temporal in ["vlad","rvlad","vf","meanP","maxP"]:# and temporal == "vlad":
            in_dim, out_dim = 512 * vlad_clusters["vggsound"], 512
        elif expert == "panns_cnn10" and temporal in ["vlad","rvlad","vf","meanP","maxP"]:# and temporal == "vlad":
            in_dim, out_dim = 512 * vlad_clusters["panns_cnn10"], 512
        elif expert == "panns_cnn14" and temporal =="seqLSTM":
            in_dim, out_dim = arch_args["hidden_size"][expert], 2048
        elif expert == "panns_cnn14" and temporal =="seqTransf":
            in_dim, out_dim = arch_args["transformer_width"][expert], 2048
        else:
            common_dim = feat_agg[expert]["feat_dims"][feat_agg[expert]["type"]]
            # account for aggregation of multilpe forms (e.g. avg + max pooling)
            common_dim = common_dim * len(feat_agg[expert]["temporal"].split("-"))
            in_dim, out_dim = common_dim, common_dim

        # For the CE architecture, we need to project all features to a common
        # dimensionality
#        is_ce = config["arch"]["type"] == "CENet"
#        if is_ce and (arch_args["use_ce"] or arch_args.get("mimic_ce_dims", False)):
#            out_dim = experts["ce_shared_dim"]

        dims.append((expert, (in_dim, out_dim)))
    expert_dims = OrderedDict(dims)

#    if vlad_clusters["text"] == 0:
#        msg = "vlad can only be disabled for text with single tokens"
#        assert config["data_loader"]["args"]["max_tokens"]["text"] == 1, msg

 #   if config["experts"]["text_agg"] == "avg":
 #       if hasattr(config["arch"]["args"], "vlad_clusters"):
 #           msg = "averaging can only be performed with text using single tokens"
 #           assert config["arch"]["args"]["vlad_clusters"]["text"] == 0, msg
 #       assert config["data_loader"]["args"]["max_tokens"]["text"] == 1

    # To remove the dependency of dataloader on the model architecture, we create a
    # second copy of the expert dimensions which accounts for the number of vlad
    # clusters
    raw_input_dims = OrderedDict()
    for expert, dim_pair in expert_dims.items():
        raw_dim = dim_pair[0]
        if expert in {"lms","vggish", "panns_cnn10", "panns_cnn14", "vggsound"}:
            if feat_agg[expert]["temporal"] == "vlad":
                raw_dim = raw_dim // vlad_clusters.get(expert, 1)
        raw_input_dims[expert] = raw_dim

    #with open(config["text_embedding_model_configs"], "r") as f:
    #    text_embedding_model_configs = json.load(f)
    #text_dim = text_embedding_model_configs[experts["text_feat"]]["dim"]
    text_dim = 300
    return expert_dims, raw_input_dims, text_dim


def ensure_tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    return x


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()

def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, model_size=512, warmup_iters=100, last_epoch=-1, verbose=False):
        self.model_size = model_size
        self.warmup_iters = warmup_iters
        self.factors = [group["lr"] / (self.model_size ** (-0.5) * self.warmup_iters ** (-0.5)) for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch, verbose)

    def _get_closed_form_lr(self):
        current_iter = self._step_count
        current_lrs = []
        for factor in self.factors:
            current_lr = factor * self.model_size ** (-0.5) * min(current_iter ** (-0.5), current_iter * self.warmup_iters ** (-1.5))
            current_lrs.append(current_lr)
        return current_lrs

    def get_lr(self):
        return self._get_closed_form_lr()

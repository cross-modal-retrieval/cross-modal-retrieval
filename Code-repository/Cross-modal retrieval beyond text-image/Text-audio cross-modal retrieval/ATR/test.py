"""Code based on the implementation of "audio-retrieval":
https://github.com/oncescuandreea/audio-retrieval
"""

import os
import copy
import pickle
import random
import logging
import argparse
import numpy as np
import torch
from typing import Tuple, Dict
from pathlib import Path
from mergedeep import Strategy, merge
from typeguard import typechecked

import model.model_pooling as module_arch
import model.metric as module_metric
from trainer.trainer import verbose, ctxt_mgr
from utils.util import compute_dims, compute_trn_config
from parse_config import ConfigParser

@typechecked
def compress_predictions(query_masks:np.ndarray, sims: np.ndarray, topk: int =10):
    """We store the indices of the top-k predictions, rather than the full similarity
    matrix, to reduce storage requirements.

    NOTE: The similarity matrix contains 'num_queries x num_audios' elements, where
    'num_queries = num_audios x max_num_queries_per_video'. We first mask out
    locations in the similarity matrix that correspond to invalid queries (These are
    produced by videos with fewer than 'max_num_queries_per_video' descriptions).
    """

    # validate the input shapes
    assert query_masks.ndim == 2, "Expected query_masks to be a matrix"
    query_num_videos, query_max_per_video = query_masks.shape
    sims_queries, sims_num_videos = sims.shape
    msg = (f"Expected sims and query masks ti represent the same number of videos"
           f"(found {sims_num_videos} v {query_num_videos}")
    assert query_num_videos == sims_num_videos, msg
    msg = (f"Expected sims and query masks to represent the same number of queries "
           f"(found {sims_queries} v {query_num_videos * query_max_per_video}")
    assert query_max_per_video * query_num_videos == sims_queries, msg

    valid_sims = sims[query_masks.flatten().astype(np.bool)]
    ranks = np.argsort(-valid_sims, axis = 1)
    return ranks[:,:topk]

@typechecked
def get_model_and_data_loaders(
        config:ConfigParser,
        logger: logging.Logger,
        ckpt_path: Path,
        device: str
) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader]:
    expert_dims, raw_input_dims, text_dim = compute_dims(config)

    data_root = config['data_loader']['root'] 
    text_encoder = config['data_loader']['text_encoder']
    if config['data_loader']['dataset'] in ["CLOTHO", "CLOTHO_V2"]:
        from data_loader.CLOTHO_dataloader import create_train_dataloader, create_val_dataloader
        index_file_path = os.path.join(data_root,"Index","test.json")
    elif config['data_loader']['dataset'] == "AudioCaps":
        from data_loader.AudioCaps_dataloader import create_train_dataloader, create_val_dataloader
        index_file_path = os.path.join(data_root, 'Index', 'test_filtered.csv')
   
    audio_h5 = []
    for audio_expert in config['experts']['modalities']:
        audio_file_path = os.path.join(data_root,
                                       'AudioExpert',
                                        audio_expert,
                                        "test",
                                        '%s.h5'%audio_expert
        )
        audio_h5.append(audio_file_path)
    
    w2v_file_path = os.path.join(data_root, 'TextEmbeddings', '%s_test.pkl'%text_encoder)

    test_dataloader = create_val_dataloader(w2v_file_path,
                                           audio_h5,
                                           config['experts']['modalities'],
                                           index_file_path,
                                           config['data_loader']['max_words'],
                                           config['data_loader']['audio_padding_length'],
                                           split = 'test')
   
    trn_config = compute_trn_config(config)
    model = config.init(
        name='arch',
        module=module_arch,
        audio_dims=expert_dims,
        text_dim=text_dim,
        feat_aggregation=config["data_loader"]["args"]["feat_aggregation"]
    )
 
    ckpt_path = config._args.resume
    logger.info(f"Loading checkpoint: {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] >1:
        model = torch.nn.DataParallel(model)
    
    # support backends compatibility
    deprecated = ["ce.moe_fc_bottleneck1","ce.moe_cg","ce.moe_fc_proj"]
    for mod in deprecated:
        for suffix in ("weight","bias"):
            key = f"{mod}.{suffix}"
            if key in state_dict:
                print(f"WARNING: Removing deprecated key {key} from model")
                state_dict.pop(key)
    model.load_state_dict(state_dict)

    return model, test_dataloader

def evaluation(config, logger = None, trainer= None):

    if logger is None:
        logger = config.get_logger('test')

    if getattr(config._args, "eval_from_training_config", False):
        eval_conf = copy.deepcopy(config)
        merge(eval_conf._config, config["eval_settings"], strategy=Strategy.REPLACE)
        config = eval_conf

    logger.info("Running evaluation with configuration:")
    logger.info(config)

    # Set the random initial seeds
    seed = config["seed"]
    logger.info(f"Setting experiment random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    # prepare model for testing. Note that some datasets fail to fit the retrieval 
    # set on the GPU, so we run them on CPU
    if torch.cuda.is_available() and not config.get("disable_gpu", True):
        device = 'cuda'
    else:
        device = 'cpu'

    logger.info(f"Running evaluation on {device}")

    model, data_loader = get_model_and_data_loaders(
            config =config,
            logger= logger,
            ckpt_path=Path(config._args.resume),
            device = device
    )
    logger.info(model)
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, samples in enumerate(data_loader):
            disable_nan_checks = config._config["disable_nan_checks"]

            with ctxt_mgr(samples, device, disable_nan_checks) as xx:
                output = model(**xx)
            if config['loss']['type'] == "MaxMarginRankingLoss":
                    sims = output["cross_view_conf_matrix"].data.cpu().float().numpy()
            elif config['loss']['type'] == "MaxMarginRankingLoss2":
                    sims, sims_audio, sims_text = output["cross_view_conf_matrix"]
                    sims = sims.data.cpu().float().numpy()
                    sims_audio = sims_audio.cpu().float().numpy()
                    sims_text = sims_text.cpu().float().numpy()
            
#            sims = output['cross_view_conf_matrix'].data.cpu().float().numpy()
            dataset = config['data_loader']['dataset']
            with open('audiocaps.npy','wb') as f:
                np.save(f,sims)
            nested_metrics = {}

            for metric in metrics:
                metric_name = metric.__name__
                res = metric(sims)
                verbose(epoch=0, metrics=res, name= dataset,mode=metric_name)
                if trainer is not None:
                    if not trainer.mini_train:
                        trainer.writer.set_step(step=0, mode="val")
                    # avoid tensorboad folding by prefixing
                    metric_name_ = f"test_{metric_name}"
                    trainer.log_metrics(res, metric_name=metric_name_, mode ="val")
                nested_metrics[metric_name] = res
    log = {}
    for subkey, subval in nested_metrics.items():
        for subsubkey, subsubval in subval.items():
            log[f"test_{subkey}_{subsubkey}"] = subsubval
    for key, value in log.items():
        logger.info(" {:15s}: {}".format(str(key),value))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--config', default=None, type=str, help="config file path")
    args.add_argument('--resume', type=Path, help='path to checkpoint for evaluation')
    args.add_argument('--device', help='indices of GPUs to enable')
    args.add_argument('--eval_from_training_config', action="store_true",
                      help="if true, evaluate directly from a training config file.")
    args.add_argument("--custom_args", help="qualified key,val pairs")
    args.add_argument("--per_class", action="store_true",
                      help="if true, evaluate retrieval task only on specific class")
    eval_config = ConfigParser(args)

    cfg_msg = "For evaluation, a model checkpoint must be specified via the --resume flag"
    assert eval_config._args.resume, cfg_msg
    evaluation(eval_config)

"""Module for computing performance metrics
"""
import math
import numbers
from pathlib import Path

import numpy as np
import torch
import scipy.stats
from sklearn.metrics import average_precision_score

def t2a_metrics(sims, query_masks=None):
    """Compute retrieval metrics from a similarity matrix.

    Args:
        sims (th.Tensor) : N x M matrix of similarities between embeddings, where
            x_{i,j} = <text_emb[i], audio_embed[j]>
        query_masks (th.Tensor): mask any missing queries from dataset (two videos
            in MSRVTT only have 19, rather than 20 captions, in our case no)
    
    Returns:
        (dict[str:float]): retrieval metrics
    """
    assert sims.ndim == 2, "expected a matrix"
    num_queries, num_audios = sims.shape
    dists = -sims
    sorted_dists = np.sort(dists, axis = 1)

    # The indices are computed such that they slice out the ground truth distances
    # from the psuedo-rectangular dist matrix
    queries_per_audio = num_queries // num_audios
    gt_idx = [[np.ravel_multi_index([ii, jj], (num_queries, num_audios))
               for ii in range(jj * queries_per_audio, (jj + 1) * queries_per_audio)]
               for jj in range(num_audios)]
    gt_idx = np.array(gt_idx)
    gt_dists = dists.reshape(-1)[gt_idx.reshape(-1)]
    gt_dists = gt_dists[:, np.newaxis]
    rows, cols = np.where((sorted_dists - gt_dists) == 0) # find column position of GT
    break_ties = "averaging"

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            _, idx = np.unique(rows, return_index=True)
            cols = cols[idx]
        elif break_ties == "averaging":
            # fast implementation, based on this code:
            # https://stackoverflow.com/a/49239335
            locs = np.argwhere((sorted_dists - gt_dists) == 0)

            # Find the split indices
            steps = np.diff(locs[:, 0])
            splits = np.nonzero(steps)[0] + 1
            splits = np.insert(splits, 0, 0)

            # Compute the result columns
            summed_cols = np.add.reduceat(locs[:, 1], splits)
            counts = np.diff(np.append(splits, locs.shape[0]))
            avg_cols = summed_cols / counts
            if False:
                print("Running slower code to verify rank averaging across ties")
                # slow, but more interpretable version, used for testing
                avg_cols_slow = [np.mean(cols[rows == idx]) for idx in range(num_queries)]
                assert np.array_equal(avg_cols, avg_cols_slow), "slow vs fast difference"
                print("passed num check")
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {})"
    if cols.size != num_queries:
        import ipdb; ipdb.set_trace()
    assert cols.size == num_queries, msg

    if False:
        #sanity check against old logic for square metrices
        gt_dists_old = np.diag(dists)
        gt_dists_old = gt_dists_old[:, np.newaxis]
        _, cols_old = np.where((sorted_dists - gt_dists_old) == 0)
        assert np.array_equal(cols_old, cols), "new metrics doesn't match"

    return cols2metrics(cols, num_queries)

def a2t_metrics(sims, query_masks =None):

    """Compute retrieval metrics from a similarity matrix.

    Args:
        sims (th.Tensor) : N x M matrix of similarities between embeddings, where
            x_{i,j} = <text_emb[i], audio_embed[j]>
        query_masks (th.Tensor): mask any missing queries from dataset (two videos
            in MSRVTT only have 19, rather than 20 captions, in our case no)
    
    Returns:
        (dict[str:float]): retrieval metrics
    
    NOTES: We find the closest "GT caption" in the style of VSE, which corresponds
    to finding the rank of the closest relevant caption in embedding space:
    github.com/ryankiros/visual-semantic-embedding/blob/master/evaluation.py#L52-L56
    """

    # switch axes of text and audio
    sims = sims.T

    if False:
        # experiment with toy example
        sims = np.ones((3,3))
        sims[0, 0] = 2
        sims[1, 1:2] = 2
        sims[2.:] = 2
        query_masks = None
    
    assert sims.ndim == 2, "expected a matrix"
    num_queries, num_caps = sims.shape
    dists = -sims
    caps_per_audio = num_caps // num_queries
    break_ties = "averaging"

    MISSING_VAL = 1E8
    query_ranks = []
    for ii in range(num_queries):
        row_dists = dists[ii, :]
        
        # NOTE: Using distance subtraction to perform the ranking is easier to make
        # deterministic than using argsort, which suffers from the issue of defining
        # "stability" for equal distances.  Example of distance subtraction code:
        # github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/train.py
        sorted_dists = np.sort(row_dists)

        min_rank = np.inf
        for jj in range(ii * caps_per_audio, (ii + 1) * caps_per_audio):
            ranks = np.where((sorted_dists - row_dists[jj]) == 0)[0]
            if break_ties == "optimistically":
                rank = ranks[0]
            elif break_ties == "averaging":
                #NOTE: If there is more than one caption per audio, its possible for 
                #the method to do "worse than chance" in the degenerate case when all
                #similarities are tied
                rank = ranks.mean()
            if rank < min_rank:
                min_rank = rank
        query_ranks.append(min_rank)
    query_ranks = np.array(query_ranks)

    # sanity check against old version of code
    if False:
        sorted_dists = np.sort(dists, axis = 1)
        gt_dists_old = np.diag(dists)
        gt_dists_old = gt_dists_old[:, np.newaxis]
        rows_old, cols_old = np.where((sorted_dists - gt_dists_old) == 0)
        if rows_old.size > num_queries:
            _, idx = np.unique(rows_old, return_index = True)
            cols_old = cols_old[idx]
        num_diffs = (1 - (cols_old == query_ranks)).sum()
        msg = f"new metric doesn't match in {num_diffs} places"
        assert np.array_equal(cols_old, query_ranks), msg
    
    return cols2metrics(query_ranks, num_queries)

def cols2metrics(cols, num_queries):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(cols == 0)) / num_queries
    metrics["R5"] = 100 * float(np.sum(cols < 5)) / num_queries
    metrics["R10"] = 100 * float(np.sum(cols < 10)) / num_queries
    metrics["R50"] = 100 * float(np.sum(cols < 50)) / num_queries
    metrics["MedR"] = np.median(cols) + 1
    metrics["MeanR"] = np.mean(cols) + 1
    stats = [metrics[x] for x in ("R1", "R5", "R10")]
    metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics

def mean_average_precision(sims, query_masks=None):
    ap_meter = APMeter()
    ap_meter.add(output = sims.T, target = query_masks.T)
    return {"mAP": ap_meter.value().mean()}

class AverageMeter(object):
    """Compytes and stores the average and current value"""
    def __init__(self, name, fmt =':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt +'{ ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manners.
    This class is abstract, but provides a sGktandard interface for all meters to follow
    """

    def reset(self):
        '''Resets the meter to default settings.'''
        pass
    def add(self, value):
        '''Log a new value to the meter
        Args:
            value: Next result to include.
        '''
        pass
    def value(self):
        '''Get the value of the meter in the current state.'''
        pass





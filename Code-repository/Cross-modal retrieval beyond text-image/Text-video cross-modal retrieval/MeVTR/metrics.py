from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import torch



def compute_metrics_together(sim_matrix, mask):
    ind_gt = mask
    ind_sort = np.argsort(np.argsort(-sim_matrix)) + 1
    ind_mask = np.ma.array(ind_gt * ind_sort, mask=ind_gt == 0)
    # ind_mask = ind_mask.masked_fill(ind_mask == 0, 1000000000)

    ind_gt_t = ind_gt.T
    ind_sort_t = np.argsort(np.argsort(-sim_matrix.T)) + 1
    ind_mask_t = np.ma.array(ind_gt_t * ind_sort_t, mask=ind_gt_t == 0)
    # ind_mask_t = ind_mask_t.masked_fill(ind_mask_t == 0, 1000000000)

    rk_v2t = {}
    rk_t2v = {}

    rk_v2t['mean_mean'] = np.mean(ind_mask.mean(axis=1))
    rk_v2t['mean_median'] = np.mean(np.ma.median(ind_mask, axis=1))
    rk_t2v['mean_mean'] = np.mean(ind_mask_t.mean(axis=1))
    rk_t2v['mean_median'] = np.mean(np.ma.median(ind_mask_t, axis=1))

    rk_v2t['median_mean'] = np.median(ind_mask.mean(axis=1))
    rk_v2t['median_median'] = np.median(np.ma.median(ind_mask, axis=1))
    rk_t2v['median_mean'] = np.median(ind_mask_t.mean(axis=1))
    rk_t2v['median_median'] = np.median(np.ma.median(ind_mask_t, axis=1))

    for k in [1, 5, 10, 50, 100]:
        # print('==================')
        # print(np.sum(ind_mask <= k, axis=1))
        # print(batch_ncaption.cpu().numpy())
        # print('==================')
        r = np.mean(np.mean(ind_mask <= k, axis=1))
        r_t = np.mean(np.mean(ind_mask_t <= k, axis=1))
        rk_v2t['hit_ratio_' + str(k)] = r
        rk_t2v['hit_ratio_' + str(k)] = r_t

        r_t = np.mean(ind_mask_t.min(axis=1) <= k)
        r = np.mean(ind_mask.min(axis=1) <= k)
        rk_v2t['best_' + str(k)] = r
        rk_t2v['best_' + str(k)] = r_t

        r_t = np.mean(ind_mask_t.max(axis=1) <= k)
        r = np.mean(ind_mask.max(axis=1) <= k)
        rk_v2t['worst_' + str(k)] = r
        rk_t2v['worst_' + str(k)] = r_t

    return rk_t2v, rk_v2t

def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))

# below two functions directly come from: https://github.com/Deferf/Experiments
def tensor_text_to_video_metrics(sim_tensor, top_k = [1,5,10]):
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)

    # Permute sim_tensor so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sim_matrices = sim_tensor.permute(1, 0, 2)
    first_argsort = torch.argsort(stacked_sim_matrices, dim = -1, descending= True)
    second_argsort = torch.argsort(first_argsort, dim = -1, descending= False)

    # Extracts ranks i.e diagonals
    ranks = torch.flatten(torch.diagonal(second_argsort, dim1 = 1, dim2 = 2))

    # Now we need to extract valid ranks, as some belong to inf padding values
    permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(permuted_original_data), torch.isnan(permuted_original_data))
    valid_ranks = ranks[mask]
    # A quick dimension check validates our results, there may be other correctness tests pending
    # Such as dot product localization, but that is for other time.
    #assert int(valid_ranks.shape[0]) ==  sum([len(text_dict[k]) for k in text_dict])
    if not torch.is_tensor(valid_ranks):
      valid_ranks = torch.tensor(valid_ranks)
    results = {f"R{k}": float(torch.sum(valid_ranks < k) * 100 / len(valid_ranks)) for k in top_k}
    results["MedianR"] = float(torch.median(valid_ranks + 1))
    results["MeanR"] = float(np.mean(valid_ranks.numpy() + 1))
    results["Std_Rank"] = float(np.std(valid_ranks.numpy() + 1))
    results['MR'] = results["MedianR"]
    return results

def tensor_video_to_text_sim(sim_tensor):
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)
    # Code to avoid nans
    sim_tensor[sim_tensor != sim_tensor] = float('-inf')
    # Forms a similarity matrix for use with rank at k
    values, _ = torch.max(sim_tensor, dim=1, keepdim=True)
    return torch.squeeze(values).T

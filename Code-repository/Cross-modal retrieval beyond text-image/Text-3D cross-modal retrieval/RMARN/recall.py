import numpy as np
import torch
import tqdm
import gc
import logging
logging.basicConfig(level=logging.INFO)

def get_sim(txt_feature, cld_feature, sim_func, txt_block_size=10, cld_block_size=10):
    """
        get similarities
    :param cld_block_size:   cld_num per block
    :param txt_block_size:   txt_num per block
    :param sim_func:       ([bb1, ss1, h1], [bb2, ss2, h2]) -> [bb1, bb2, h2]
    :param txt_feature:    b1, s1, h1
    :param cld_feature:    b2, s2, h2
    :return:               [b1, b2]
    """
    logging.info("Starting get_sim")
    print(f"txt.shape: {txt_feature.shape} cld.shape: {cld_feature.shape}")
    b1, s1, h1 = txt_feature.shape
    b2, s2, h2 = cld_feature.shape

    sims = np.zeros([b1, b2])

    for txt_blk_idx_start in tqdm.trange(0, b1, txt_block_size):    # tqdm.trange
        for cld_blk_idx_start in range(0, b2, cld_block_size):

            txt_blk = txt_feature[txt_blk_idx_start: txt_blk_idx_start + txt_block_size]
            cld_blk = cld_feature[cld_blk_idx_start: cld_blk_idx_start + cld_block_size]

            sim = sim_func(txt_blk, cld_blk)

            sims[
                txt_blk_idx_start: txt_blk_idx_start + txt_block_size,
                cld_blk_idx_start: cld_blk_idx_start + cld_block_size
            ] = sim

            del txt_blk, cld_blk, sim  # Delete unused variables
            gc.collect()  # Force garbage collection
    logging.info("Finished get_sim")
    return sims


def get_rank(txt_feature, cld_feature, link, sim_func, t2c=True, txt_block_size=10, cld_block_size=10):
    """
        get rank
    :param t2c:              text to cloud (True) or cloud to text (False)
    :param link:             [[txt_i, cld_i], ...]
    :param txt_feature:      see get_sim
    :param cld_feature:      see get_sim
    :param sim_func:         see get_sim
    :param txt_block_size:   see get_sim
    :param cld_block_size:   see get_sim
    :return:  b1 or b2
    """
    logging.info("Starting get_rank")
    print('getting similarities')
    sims = get_sim(txt_feature, cld_feature, sim_func, txt_block_size, cld_block_size)
    if not t2c:
        sims = sims.transpose([1, 0])
        link = [[r[1], r[0]] for r in link]

    print("getting ranks")
    ranks = np.full([sims.shape[0]], -1)
    for r in tqdm.tqdm(link):
        i1, i2 = r
        tgt_sim = sims[i1, i2]
        rank = np.sum((sims[i1] < tgt_sim).astype(int)) + 1
        if ranks[i1] == -1 or (ranks[i1] > -1 and ranks[i1] > rank):
            ranks[i1] = rank

    have_rank = np.nonzero(ranks > -1)[0]
    if len(have_rank) < sims.shape[0]:
        import warnings
        warnings.warn(
            f"There exists {'text' if t2c else 'cloud'} having no matched {'cloud' if t2c else 'text'} in link"
        )
        ranks = ranks[have_rank]
    del sims, have_rank  # Delete unused variables
    gc.collect()  # Force garbage collection
    logging.info("Finished get_rank")
    return ranks


def evaluation(txt_feature, cld_feature, link, sim_func, t2c=True, txt_block_size=10, cld_block_size=10, recall_idx=None):
    """
        get recall and mean rank
    :param recall_idx:       [recall_{i}, ...]
    :param txt_feature:    see get_rank
    :param cld_feature:    see get_rank
    :param link:           see get_rank
    :param sim_func:       see get_rank
    :param t2c:            see get_rank
    :param txt_block_size: see get_rank
    :param cld_block_size: see get_rank
    :return:   [r_{i}], mr, mrr
    """
    if recall_idx is None:
        recall_idx = [1, 5, 10]
    rank = get_rank(txt_feature, cld_feature, link, sim_func, t2c, txt_block_size, cld_block_size)
    recalls = []

    print("calculating recalls")
    for recall_i in tqdm.tqdm(recall_idx):
        recalls.append(
            np.mean(rank <= recall_i).astype(float)
        )

    mr = np.mean(rank)
    mrr = np.mean(1 / rank)

    del rank  # Delete unused variables
    gc.collect()  # Force garbage collection

    return recalls, mr, mrr


def check_gpu_memory():
    """

    """
    if not torch.cuda.is_available():
        print("CUDA is not available")

def t_recall():
    device = torch.device("cuda")
    # baseline 7.9 test1
    link_path = '/your_path/to/7.27_link.json'
    dataset = PairDataset("", "", link_path, device=device)
    print(f'dataset size: {len(dataset)}')
    d_model = 512
    nhead = 8
    dropout = 0.1
    model = RetrievalModel(d_model, nhead, dropout, rank=2, num_layers=1)
    txt, cld, link = dataset.get_all()

    model, txt, cld = model.to(device), txt.to(device), cld.to(device)
    model.eval()
    recalls, mr, mrr = evaluation(txt, cld, link, sim_func=model, t2c=True, recall_idx=[1, 5, 10])
    print(f"Recall@1: {recalls[0]:.4f},  Recall@5: {recalls[1]:.4f}, Recall@10: {recalls[2]:.4f}")

    del model, txt, cld, link  # Delete unused variables
    gc.collect()  # Force garbage collection

if __name__ == '__main__':
    t_recall()

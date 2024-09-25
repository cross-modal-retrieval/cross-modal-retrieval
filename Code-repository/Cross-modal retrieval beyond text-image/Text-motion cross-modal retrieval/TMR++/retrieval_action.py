import os
from omegaconf import DictConfig
import logging
import hydra
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)


def compute_sim_matrix(model, dataset, keyids, batch_size=256):
    import torch
    import numpy as np
    from src.data.collate import collate_text_motion
    from src.model.tmr import get_sim_matrix

    device = model.device

    nsplit = int(np.ceil(len(dataset) / batch_size))
    with torch.inference_mode():
        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        all_data_splitted = np.array_split(all_data, nsplit)

        # by batch (can be too costly on cuda device otherwise)
        latent_texts = []
        latent_motions = []
        sent_embs = []
        
        for data in tqdm(all_data_splitted, leave=True):
            batch = collate_text_motion(data, device=device)

            # Text is already encoded
            text_x_dict = batch["text_x_dict"]
            motion_x_dict = batch["motion_x_dict"]
            sent_emb = batch["sent_emb"]

            # Encode both motion and text
            latent_text = model.encode(text_x_dict, sample_mean=True)
            latent_motion = model.encode(motion_x_dict, sample_mean=True)

            latent_texts.append(latent_text)
            latent_motions.append(latent_motion)
            sent_embs.append(sent_emb)

        latent_texts = torch.cat(latent_texts)
        action_latent_text = torch.unique(latent_texts, dim=0)

        action_latent_text_idx = {tuple(action_latent_text[i].to("cpu").numpy()): i for i in range(len(action_latent_text))}

        latent_motions = torch.cat(latent_motions)
        motion_cat_idx = [action_latent_text_idx[tuple(latent_texts[i].to("cpu").numpy())] for i in range(len(latent_motions))]

        #sent_embs = torch.cat(sent_embs)
        sim_matrix = get_sim_matrix(action_latent_text, latent_motions)
    returned = {
        "sim_matrix": sim_matrix.cpu().numpy(),
        "motion_cat_idx": motion_cat_idx
    }
    return returned

@hydra.main(version_base=None, config_path="configs", config_name="retrieval")
def retrieval(newcfg: DictConfig) -> None:
    device = newcfg.device
    run_dir = newcfg.run_dir
    ckpt_name = newcfg.ckpt
    batch_size = newcfg.batch_size
    save_file_name = newcfg.save_file_name
    split = newcfg.split

    assert split == "test" 
    protocols = ["normal"]

    save_dir = os.path.join(run_dir, save_file_name)
    os.makedirs(save_dir, exist_ok=True)

    # Load last config
    from src.config import read_config
    import src.prepare  # noqa

    cfg = read_config(run_dir)

    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    from src.load import load_model_from_cfg
    from src.model.metrics import all_contrastive_metrics_action_retrieval, print_latex_metrics

    pl.seed_everything(cfg.seed)

    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)


    data = newcfg.data
    if data is None:
        data = cfg.data

    datasets = {}
    for protocol in protocols:
        # Load the dataset if not already
        if protocol not in datasets:
            dataset = instantiate(data, split=split)
            datasets.update(
                {key: dataset for key in ["normal", "threshold", "guo"]}
            )
        dataset = datasets[protocol]

        # Compute sim_matrix for each protocol
        protocol = "normal"
        result = compute_sim_matrix(
            model, dataset, dataset.keyids, batch_size=batch_size
        )

        # Compute the metrics
        sim_matrix = result["sim_matrix"]
        motion_cat_idx = result["motion_cat_idx"]
        
        protocol_name = protocol
        metrics = all_contrastive_metrics_action_retrieval(sim_matrix, motion_cat_idx, norm_metrics=True)

        print_latex_metrics(metrics, ranks=[1, 2, 3, 5, 10], t2m=False, m2t=True, MedR=False)

        metric_name = f"{protocol_name}.yaml"
        path = os.path.join(save_dir, metric_name)
        save_metric(path, metrics)

        logger.info(f"Testing done, metrics saved in:\n{path}")


if __name__ == "__main__":
    retrieval()

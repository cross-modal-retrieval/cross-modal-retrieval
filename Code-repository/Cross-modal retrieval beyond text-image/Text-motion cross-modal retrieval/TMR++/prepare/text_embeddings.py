import sys
import logging
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="text_embeddings", version_base="1.3")
def text_embeddings(cfg: DictConfig):
    device = cfg.device

    from src.data.text import save_token_embeddings, save_sent_embeddings

    annotations_filename = cfg.annotations_filename

    # Compute token embeddings
    modelname = cfg.data.text_to_token_emb.modelname
    modelpath = cfg.data.text_to_token_emb.modelpath
    logger.info(f"Compute token embeddings for {modelname}")
    path = cfg.data.text_to_token_emb.path
    output_folder_name = cfg.output_folder_name_token
    save_token_embeddings(path, annotations_filename=annotations_filename,
                          output_folder_name=output_folder_name,
                          modelname=modelname, modelpath=modelpath,
                          device=device)

    # Compute sent embeddings
    modelname = cfg.data.text_to_sent_emb.modelname
    modelpath = cfg.data.text_to_sent_emb.modelpath
    logger.info(f"Compute sentence embeddings for {modelname}")
    path = cfg.data.text_to_sent_emb.path
    output_folder_name = cfg.output_folder_name_sent
    save_sent_embeddings(path, annotations_filename=annotations_filename,
                         output_folder_name=output_folder_name,
                         modelname=modelname, modelpath=modelpath,
                         device=device)


if __name__ == "__main__":
    text_embeddings()

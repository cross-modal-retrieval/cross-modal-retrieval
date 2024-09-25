import hydra
from hydra.utils import instantiate
import logging
from omegaconf import DictConfig
import os
import pytorch_lightning as pl

from src.config import read_config, save_config

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    # Resuming if needed
    ckpt = None

    if cfg.ckpt is not None:
        ckpt = cfg.ckpt

    if cfg.resume_dir is not None:
        assert cfg.ckpt is not None
        max_epochs = cfg.trainer.max_epochs
        ckpt = os.path.join(cfg.resume_dir, 'logs', 'checkpoints', f'{cfg.ckpt}.ckpt')
        cfg = read_config(cfg.resume_dir)
        cfg.trainer.max_epochs = max_epochs
        logger.info("Resuming training")
        logger.info(f"The config is loaded from: \n{cfg.run_dir}")
    else:
        if "ckpt_path" in cfg and cfg.ckpt_path is not None:
            ckpt = cfg.ckpt_path
        config_path = save_config(cfg)
        logger.info("Training script")
        logger.info(f"The config can be found here: \n{config_path}")

    pl.seed_everything(cfg.seed)

    text_to_token_emb = instantiate(cfg.data.text_to_token_emb)
    text_to_sent_emb = instantiate(cfg.data.text_to_sent_emb)

    logger.info("Loading the dataloaders")
    train_dataset = instantiate(cfg.data, split="train",
                                text_to_token_emb=text_to_token_emb,
                                text_to_sent_emb=text_to_sent_emb)

    if "data_val" not in cfg:
        data_val = cfg.data
    else:
        data_val = cfg.data_val
        text_to_token_emb = instantiate(cfg.data_val.text_to_token_emb)
        text_to_sent_emb = instantiate(cfg.data_val.text_to_sent_emb)

    val_dataset = instantiate(data_val, split="val",
                              text_to_token_emb=text_to_token_emb,
                              text_to_sent_emb=text_to_sent_emb)

    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )

    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
    )

    logger.info("Loading the model")
    model = instantiate(cfg.model)

    logger.info(f"Using checkpoint: {ckpt}")
    logger.info("Training")
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt)


if __name__ == "__main__":
    train()

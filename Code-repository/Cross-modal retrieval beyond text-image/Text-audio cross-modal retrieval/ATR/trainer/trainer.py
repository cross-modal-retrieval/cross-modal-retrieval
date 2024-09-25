
"""Code based on the implementation of "audio-retrieval":
https://github.com/oncescuandreea/audio-retrieval
"""

from contextlib import contextmanager

import copy
import torch
import numpy as np

from base import BaseTrainer
from utils import memory_summary

def verbose(epoch, metrics, mode, name='TEST'):
    r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
    msg = f"[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}"
    msg += f". R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
    msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"

@contextmanager
def ctxt_mgr(samples, device, disable_nan_checks):
    """Provide a context for managing temporary, cloned copies of retrieval
    sample tensors.

    The rationale here is that to use nan-checking in the model (to validate the 
    positions of missing experts), we need to modify the underlying tensors. This 
    function lets the evaluation code run (and modify) temporary copies, without
    modifying the originals.
    """
    if disable_nan_checks:
        print("running without nan checks")
        yield samples
    else:
        exp_dict = samples["experts"].items()
        exp_mask_dict = samples["expert_masks"].items()
        experts = {key: val.clone().to(device) for key, val in exp_dict}
        expert_masks = {key : val.clone().to(device) for key, val in exp_mask_dict}
        samples_ = {
            "experts":experts,
            "ind": samples["ind"],
            "text": samples["text"].to(device),
            "expert_masks": expert_masks
        }
        if "text_token_masks" in samples:
            samples_["text_token_masks"] = samples["text_token_masks"].to(device)
        try:
           yield samples_
        finally:
            del samples_

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loaders,
                 lr_scheduler, disable_nan_checks, skip_first_n_saves,
                 include_optim_in_ckpts, force_cpu_val, distil_loss, distil_params,
                 cache_targets=set(), num_keep_ckpts=3, mini_train=False, val_freq=1,
                 skip_tboard=False):
        super().__init__(model, loss, metrics, optimizer,config, 
                         mini_train=mini_train, skip_tboard=skip_tboard,
                         num_keep_ckpts=num_keep_ckpts)
        self.config = config
        self.cache_targets = cache_targets
        self.data_loaders = data_loaders
        self.lr_scheduler = lr_scheduler
        self.mini_train = mini_train
        self.disable_nan_checks = disable_nan_checks
        self.len_epoch = len(self.data_loaders["train"])
        self.log_step = int(np.sqrt(data_loaders["train"].batch_size))
        self.force_cpu_val = force_cpu_val
        self.val_freq = val_freq
        self.skip_first_n_saves = skip_first_n_saves
        self.include_optim_in_ckpts = include_optim_in_ckpts
        self.seen = {"train": 0, "val": 0}
        self.distil_loss = distil_loss
        self.distil_params = distil_params
        self.tt_loss = torch.nn.SmoothL1Loss(reduction="elementwise_mean")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        total_loss = 0
        self.model.train()
        memory_summary()

#        if epoch  == 1:
#            self.logger.info("start the first validation")
#            nested_log, cached_preds = self._valid_epoch(epoch)
 
        for batch_idx, minibatch in enumerate(self.data_loaders["train"]):
            for key, val in minibatch["experts"].items():
                minibatch["experts"][key] = val.to(self.device)

            for key, val in minibatch["expert_masks"].items():
                minibatch["expert_masks"][key] = val.to(self.device)

            for key in {"text","text_token_masks"}:
                if key in minibatch:
                    minibatch[key] = minibatch[key].to(self.device)
            self.optimizer.zero_grad()
            output = self.model(**minibatch)
            if self.config["loss"]["type"] =="MaxMarginRankingLoss2":
                sims, audio_sims, text_sims = output["cross_view_conf_matrix"]
                loss = self.loss(sims,audio_sims, text_sims)
            else:
                loss = self.loss(output["cross_view_conf_matrix"])
 
            loss.backward()
            self.optimizer.step()

            sample_key = list(minibatch["experts"].keys())[0]
            batch_size = minibatch["experts"][sample_key].shape[0]
            self.seen["train"] += batch_size

            if not self.skip_tboard:
                self.writer.set_step(self.seen["train"], mode="train")
                self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()

            if batch_idx % self.log_step == 0:
                prog = self._progress(batch_idx)
                self.logger.info(f"Train Epoch: {epoch} {prog} Loss: {loss.item():.6f}")
                
            if batch_idx == self.len_epoch or (self.mini_train and batch_idx > 3):
                break
            if self.config["lr_scheduler"]["type"] == "noam":
                self.lr_scheduler.step()
        log = {'loss': total_loss / self.len_epoch}
        if epoch % self.val_freq == 0:
            nested_log, cached_preds = self._valid_epoch(epoch)
            log.update(nested_log)
        else:
            nested_log, cached_preds = {}, None
            self.logger.info(f"skipping val for epoch: {epoch}")
        

        if self.lr_scheduler is not None:
            if self.config["lr_scheduler"]["type"] != "noam":
                self.lr_scheduler.step()

        self.logger.info(f"LR {self.lr_scheduler.get_lr()}")
        return log, cached_preds

    def log_metrics(self, metric_store, metric_name, mode):
        if not self.skip_tboard:
            print(f"logging metrics: {metric_name}")
            self.writer.set_step(step=self.seen[mode], mode = mode)
            for key, value in metric_store.items():
                self.writer.add_scalar(f"{metric_name}/{key}", value)

    def _valid_epoch(self, epoch):
        """
        Validate mode after an epoch of training and store results to disk.

        Args:
            epoch (int): the current epoch

        Returns:
            A log that contains information about validation

        NOTE: The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        if not self.skip_tboard:
            self.writer.mode = "val"
        cached_preds = {key:{"vid_name":[], "preds": [], "labels":[]}
                        for key in self.cache_targets}

        with torch.no_grad():
            for batch_idx, samples in enumerate(self.data_loaders["val"]):
#            if "retrieval" in self.data_loaders.data_loaders:
#                samples, meta = self.data_loaders["retrieval"]

                sample_key = list(samples["experts"].keys())[0]
                batch_size = samples["experts"][sample_key].shape[0]
                self.seen["val"] += batch_size
                num_queries = samples["text"].shape[0] * samples["text"].shape[1]
                safe_queries = 6000
                if num_queries > safe_queries:
                    partitions = int(np.ceil(num_queries / safe_queries))
                    chunk_size = int(np.ceil(samples["text"].shape[0] / partitions))
                    texts = copy.deepcopy(samples["text"])
                    text_masks = copy.deepcopy(samples['text_token_masks'])
                    sim_chunks = []
                    for chunk_idx in range(partitions):
                        chunk_start = chunk_idx * chunk_size
                        chunk_stop = (chunk_idx + 1) * chunk_size
                        samples["text"] = texts[chunk_start:chunk_stop]
                        samples["text_token_masks"] = text_masks[chunk_start:chunk_stop]
                        if samples['text'].shape[0] == 0:
                            continue
                        with ctxt_mgr(samples, self.device, 
                                      self.disable_nan_checks) as xx:
                            output = self.model(**xx)
                        sims = output["cross_view_conf_matrix"].data
                        sim_chunks.append(sims)

                    samples["text"] = texts # restore pointer to original tensor
                    samples["text_token_masks"] = text_masks
                    del texts, text_masks
                    sims = torch.cat(sim_chunks, dim=0).data.cpu().float().numpy()
                else:
                    with ctxt_mgr(samples, self.device, self.disable_nan_checks) as xx:
                        output = self.model(**xx)
                    self.model = self.model.to(self.device)
                    if self.config['loss']['type'] == "MaxMarginRankingLoss":
                        sims = output["cross_view_conf_matrix"].data.cpu().float().numpy()
                    elif self.config['loss']['type'] == "MaxMarginRankingLoss2":
                        sims, sims_audio, sims_text = output["cross_view_conf_matrix"]
                        sims = sims.data.cpu().float().numpy()
                        sims_audio = sims_audio.cpu().float().numpy()
                        sims_text = sims_text.cpu().float().numpy()
                # sample the loss (using only the first query for each video)
                #queries_per_vid = meta["query_masks"].shape[1]
                queries_per_vid = 5
                if self.config['loss']['type'] == "MaxMarginRankingLoss":
                    sims_ = torch.from_numpy(sims).view(-1, queries_per_vid, sims.shape[-1])
                    loss = self.loss(sims_[:, 0, :].contiguous())
                elif self.config['loss']['type'] == "MaxMarginRankingLoss2":
                    sims_ = torch.from_numpy(sims).view(-1, queries_per_vid, sims.shape[-1])
                    sims_text_ = torch.from_numpy(sims_text).view(-1, queries_per_vid, sims.shape[-1], queries_per_vid)
                    sims_audio_ = torch.from_numpy(sims_audio)
                    loss = self.loss(sims_[:,0,:].contiguous(), sims_audio_, sims_text_[:,0,:,0].contiguous())
                if not self.skip_tboard:
                    self.writer.add_scalar('first_query_loss', loss.item())
                dataset = self.config['data_loader']['dataset']
                nested_metrics = {}
                for metric in self.metrics:
                    metric_name = metric.__name__
                    res = metric(sims)
                    if metric_name == 'mean_average_precision':
                        print(f"Epoch: {epoch}, mean AP: {res['mAP']}")
                    else:
                        verbose(epoch=epoch, metrics=res, name=dataset, mode=metric_name)
                    
                    self.log_metrics(res, metric_name=metric_name, mode="val")
                    nested_metrics[metric_name] = res
                
#                num_test_caps = self.data_loaders.num_test_captions
#                if num_test_caps == 1 and meta["raw_captions"] is not None:
#                    if self.visualizer is not None:
#                        self.visualizer.visualize_ranking(
#                                sims = sims,
#                                meta = meta,
#                                epoch = epoch,
#                                nested_metrics=nested_metrics,
#                            )
                return {"nested_val_metrics": nested_metrics}, cached_preds
        
    def _progress(self,batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loaders, 'n_samples'):
            current = batch_idx * self.data_loaders.batch_size
            total = self.data_loaders.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

            



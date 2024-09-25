from typing import Dict, List, Optional
from torch import Tensor

import torch
from .tmr import TMR

class TMRTextAveraging(TMR):
    """Compatible with AugmentedTextMotionDataset Dataset object and collate_text_motion_multiple_texts collate function."""

    # Forward: X => motions
    def forward(
        self,
        inputs,
        text_slices: Optional[List[int]] = None,
        lengths: Optional[List[int]] = None,
        mask: Optional[Tensor] = None,
        sample_mean: Optional[bool] = None,
        fact: Optional[float] = None,
        return_all: bool = False,
    ) -> List[Tensor]:

        # Encoding the inputs and sampling if needed
        latent_vectors, distributions = self.encode(
            inputs, sample_mean=sample_mean, fact=fact, return_distribution=True
        )

        # Averages over the different text embbedings for each sample.
        if text_slices is not None:
            latent_vectors = [torch.mean(latent_vectors[i:j], dim=0) for i, j in text_slices]
            latent_vectors = torch.stack(latent_vectors, dim=0)
            distributions = list(distributions)
            distributions[0] = torch.stack([torch.mean(distributions[0][i:j], dim=0) for i, j in text_slices], dim=0)
            distributions[1] = torch.stack([torch.mean(distributions[1][i:j], dim=0) for i, j in text_slices], dim=0)
            distributions = tuple(distributions)

        # Decoding the latent vector: generating motions
        motions = self.decode(latent_vectors, lengths, mask)

        if return_all:
            return {"motions": motions,
                    "latent_vectors": latent_vectors,
                    "distributions": distributions}

        return {"motions": motions}

    def call_models(self, batch):
        text_x_dict = batch["text_x_dict"]
        motion_x_dict = batch["motion_x_dict"]
        text_slices = batch["text_slices"]

        mask = motion_x_dict["mask"]

        # text -> motion
        t_results = self(text_x_dict, mask=mask, return_all=True, text_slices=text_slices)

        # motion -> motion
        m_results = self(motion_x_dict, mask=mask, return_all=True)

        return t_results, m_results

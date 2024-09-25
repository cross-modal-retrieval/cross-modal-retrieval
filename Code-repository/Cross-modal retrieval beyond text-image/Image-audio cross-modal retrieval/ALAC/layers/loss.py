import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn

class LabelSmoothCELoss(_Loss):
    def __init__(self, smooth_ratio, num_classes):
        super(LabelSmoothCELoss, self).__init__()
        self.smooth_ratio = smooth_ratio
        self.v = self.smooth_ratio / num_classes

    def forward(self, input, label):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.v)
        y = label.to(torch.long).view(-1, 1)
        one_hot.scatter_(1, y, 1-self.smooth_ratio+self.v)

        loss = - torch.sum(F.log_softmax(input, 1) *
                           (one_hot.detach())) / input.size(0)
        return loss


class ClipInfoCELoss(_Loss):
    # def __init__(self, partition_num):
    def __init__(self):
        super(ClipInfoCELoss, self).__init__()
        
        # self.partition_num = partition_num

    # def forward(self, logits_per_image, logits_per_text, batch):
    #def forward(self, logits):
    #    labels = torch.arange(len(logits)).cuda()
    #    loss_i = F.cross_entropy(logits, labels)
    #    loss_t = F.cross_entropy(logits.t(), labels)
    #    loss = (loss_i+loss_t)/2
    #    return loss
    def forward(self, logits_per_image, logits_per_text):
        
        bs, l_bs = logits_per_image.shape
        if l_bs == bs:
            labels = torch.arange(len(logits_per_image)).cuda()
        else:
            labels = link.get_rank() * bs + torch.arange(0, bs, dtype=torch.long).cuda()
        

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i+loss_t)/2
        return loss#, labels

def D(p, z):
    # [N, E]
    z = z.detach() # stop gradient
    p = p / p.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    # [N E] [N E] -> [N]
    return (p * z).sum(dim=1).mean() # dot product & batch coeff normalization

def D_minimize(p, z):  # ..., X, size; ..., Y, size; choose the minimize one
    z = z.detach()
    p = p / p.norm(dim=-1, keepdim=True)
    z = (z / z.norm(dim=-1, keepdim=True)).permute(0, 2, 1)
    sim = torch.bmm(p, z)
    return sim.max(dim=-1)[0].mean(dim=-1).mean()


class SimsiamLoss(nn.Module):
    def __init__(self, symmetry=True):
        super(SimsiamLoss, self).__init__()
        self.symmetry = symmetry

    def forward(self, p1, z1, p2, z2, minimize_loss=False,):
        if self.symmetry:
            if minimize_loss:
                D1 = D_minimize(p1, z2)
                D2 = D_minimize(p2, z1)
                # import ipdb
                # ipdb.set_trace()
                return -0.5 * (D1.mean() + D2.mean())
            else:
                D1 = D(p1, z2)
                D2 = D(p2, z1)
                return -0.5 * (D(p1, z2)  + D(p2, z1) )
                
                
import torch
import torch.nn.functional as F


class NTXentLoss(torch.nn.Module):

    def __init__(self, batch_size, temperature=0.1, use_cosine_similarity=True, alpha_weight=0.75):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        
        self.lce=LabelSmoothCELoss(0.4,512)
        #self.dual_softmax_loss=dual_softmax_loss()

    def softXEnt(self, target, logits):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, zis, zjs,
                norm=True,
                weights=1.0):
        temperature = self.temperature
        alpha = self.alpha_weight

        """
        Pytorch implementation of the loss  SimCRL function by googleresearch: https://github.com/google-research/simclr
        @article{chen2020simple,
                title={A Simple Framework for Contrastive Learning of Visual Representations},
                author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2002.05709},
                year={2020}
                }
        @article{chen2020big,
                title={Big Self-Supervised Models are Strong Semi-Supervised Learners},
                author={Chen, Ting and Kornblith, Simon and Swersky, Kevin and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2006.10029},
                year={2020}
                }
        """

        LARGE_NUM = 1e9
        """Compute loss for model.
        Args:
        hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.
        tpu_context: context information for tpu.
        weights: a weighting number or vector.
        Returns:
        A loss scalar.
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
        """
        # Get (normalized) hidden1 and hidden2.
        if norm:
            zis = F.normalize(zis, p=2, dim=-1)
            zjs = F.normalize(zjs, p=2, dim=-1)

        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
        #labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
        labels = labels.cuda()
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)

        """
        Different from Image-Image contrastive learning
        In the case of Image-Text contrastive learning we do not compute the similarity function between the Image-Image and Text-Text pairs  
        """
        # logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large,0, 1)) / temperature
        # logits_aa = logits_aa - masks * LARGE_NUM
        # logits_bb = torch.matmul(hidden2,  torch.transpose(hidden2_large,0, 1)) / temperature
        # logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, -1, -2)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, -1, -2)) / temperature

        loss_a = self.softXEnt(labels, logits_ab)
        loss_b = self.softXEnt(labels, logits_ba)
        #loss_a = self.lce(logits_ab,labels)
        #loss_b = self.lce(logits_ba,labels)

        return alpha * loss_a + (1 - alpha) * loss_b



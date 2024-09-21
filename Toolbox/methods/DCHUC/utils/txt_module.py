import torch
from torch import nn
from torch.nn import functional as F

LAYER1_NODE = 10240


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0.01)


class TxtModule(nn.Module):
    def __init__(self, y_dim, bit):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TxtModule, self).__init__()
        self.module_name = "text_model"

        # full-conv layers
        self.conv1 = nn.Conv2d(1, LAYER1_NODE, kernel_size=(y_dim, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(LAYER1_NODE, bit, kernel_size=1, stride=(1, 1))
        self.apply(weights_init)
        self.classifier = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            self.conv2,
            )

    def forward(self, x):
        x = self.classifier(x)
        x = x.squeeze()
        tanh = nn.Tanh()
        x = tanh(x)
        return x


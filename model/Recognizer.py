from torch import nn
import torch.nn.init as init

from .STGCNPlus import STGCN


class Classify(nn.Module):
    def __init__(self, in_channels, num_class, dropout=0, use_batch_norm=False):
        super(Classify, self).__init__()
        layers = []
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(in_channels))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_channels, num_class))
        self.classifier = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                # 使用 Xavier 初始化方法初始化权重
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # 将偏置项初始化为零
                    init.constant_(m.bias, 0)
    def forward(self, x):
        return self.classifier(x)


class Recognizer(nn.Module):
    def __init__(self,
                 model_args,
                 head_args):
        super(Recognizer, self).__init__()
        self.backbone = STGCN(**model_args)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = Classify(**head_args)

    def forward(self, x):
        x = self.backbone(x)
        N, M, C, T, V = x.shape
        x = x.reshape(N*M, C, T, V)
        x = self.pool(x)
        x = x.reshape(N, M, C).mean(dim=1)
        x = self.head(x)

        return x
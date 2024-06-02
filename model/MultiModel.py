import torch
import torch.nn as nn

from .STGCNPlus import STGCN
from .PredDecoder import PredDecoder
from .Recognizer import Classify

class MultiTaskModel(nn.Module):
    """
    input: [N, M, T, V, C]

    """
    def __init__(self,
                 backbone_cfg,
                 head_cfg,
                 pred_input_frames=10,
                 pred_output_frames=25,
                 **kwargs):
        super(MultiTaskModel, self).__init__()
        self.pred_input_frames = pred_input_frames
        self.pred_output_frames = pred_output_frames
        self.inference_start = - self.pred_input_frames * (2 ** len(backbone_cfg['down_stages']))

        self.pred_gt_start = -self.pred_output_frames
        self.pred_start = self.pred_gt_start - self.pred_input_frames * (2 ** len(backbone_cfg['down_stages']))
        pred_out_channels = backbone_cfg['base_channels'] * ( 2 ** len(backbone_cfg['inflate_stages']))

        self.backbone = STGCN(**backbone_cfg)
        self.pred_decoder = PredDecoder(input_channels=pred_out_channels,
                                        output_channels=backbone_cfg['in_channels'],
                                        input_time_frame=self.pred_input_frames,
                                        joints_to_consider=17)
        self.classifier = Classify(**head_cfg)

    def forward(self, x):
        N, M, T, V, C = x.size()

        # pred input
        x_encode = x[:, :, self.pred_start:self.pred_gt_start, :, :]  # T=40
        # x_encode = x[:, :, -35:-25, :, :]

        # start pred
        encoded_feat = self.backbone(x_encode)
        encoded_feat = torch.squeeze(encoded_feat, dim=1)
        pred_frames = self.pred_decoder(encoded_feat).permute(0, 1, 3, 2)

        # cls input
        origin = torch.squeeze(x[:, :, :self.pred_gt_start, :, :], dim=1)
        recog_input = torch.cat([origin, pred_frames], dim=1)

        # start cls
        recog_feat = self.backbone(recog_input)
        pool = nn.AdaptiveAvgPool2d(1)
        recog_feat = pool(recog_feat).reshape(N*M, -1)
        recog_logits = self.classifier(recog_feat)

        return pred_frames, recog_logits

    def cls(self, x, mode='pred'):
        N, M, T, V, C = x.size()

        x = torch.squeeze(x, dim=1)

        if mode == 'pred':
            x_encode = x[:, self.inference_start:, :, :]
            encoded_feat = self.backbone(x_encode)
            encoded_feat = torch.squeeze(encoded_feat, dim=1)
            pred_frames = self.pred_decoder(encoded_feat).permute(0, 1, 3, 2)
            recog_input = torch.cat([x, pred_frames], dim=1)
        else:
            recog_input = x

        # start cls
        recog_feat = self.backbone(recog_input)
        pool = nn.AdaptiveAvgPool2d(1)
        recog_feat = pool(recog_feat).reshape(N*M, -1)
        recog_logits = self.classifier(recog_feat)

        return recog_logits
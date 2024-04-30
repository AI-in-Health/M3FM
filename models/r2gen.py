import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
import torch.nn.functional as F

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr
        self.affine_a = nn.Linear(1024, 2048)
        self.affine_b = nn.Linear(1024, 2048)
        self.affine_c = nn.Linear(1024, 2048)
        self.affine_d = nn.Linear(1024, 2048)
        self.affine_aa = nn.Linear(1024, 2048)
        self.affine_bb = nn.Linear(1024, 2048)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        #new add
        att_feats_0=F.relu(self.affine_a(att_feats_0))
        fc_feats_0=F.relu(self.affine_b(fc_feats_0))
        att_feats_1=F.relu(self.affine_c(att_feats_1))
        fc_feats_1=F.relu(self.affine_d(fc_feats_1))

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats1, fc_feats1 = self.visual_extractor(images)
        att_feats=F.relu(self.affine_aa(att_feats1))
        fc_feats=F.relu(self.affine_bb(fc_feats1))

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output


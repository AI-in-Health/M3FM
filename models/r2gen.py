import torch
import torch.nn as nn
import numpy as np

from modules.text_extractor import TextExtractor
from modules.encoder_decoder import Transformer
import torch.nn.functional as F

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.text_extractor = TextExtractor(args)
        self.encoder_decoder = Transformer()
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr
        '''
        self.affine_a = nn.Linear(512, 2048)
        self.affine_b = nn.Linear(512, 2048)
        self.affine_c = nn.Linear(512, 2048)
        self.affine_d = nn.Linear(512, 2048)
        self.affine_aa = nn.Linear(512, 2048)
        self.affine_bb = nn.Linear(512, 2048)
        '''

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)



    def forward_mimic_cxr(self, reports,reports_ids):
        fc_feats= self.text_extractor(reports).cuda()
        output = self.encoder_decoder(fc_feats,reports_ids).cuda()

        return output


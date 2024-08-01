import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class VisualExtractor(nn.Module):
    # prepare for the demo image and text
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        self.model.from_pretrained()
        self.model.cuda()
        self.processor = MedCLIPProcessor()
        with torch.no_grad():
            self.prompt = torch.load('prompt/prompt.pth')


    def forward(self, images):
        a=[]
        for i in images:
            inputs = self.processor( text="lungs",images=i,return_tensors="pt",padding=True)
            outputs = self.model(**inputs)
            feats = outputs['img_embeds']
            a.append(feats)      
        batch_feats = torch.stack(a, dim=0)

        ha = []
        for i in range(batch_feats.shape[0]):
            b = batch_feats[i].unsqueeze(1)
            b = b.repeat(self.prompt.shape[0], 1, 1).transpose(-2, -1)
            c_t = torch.bmm(self.prompt, b)
            c_t = c_t.float()
            alpha = F.softmax(c_t)
            aa = alpha * self.prompt
            sum_a = aa.sum(axis=0)
            ha.append(sum_a)
        featsem = torch.stack(ha, dim=0)

        feats = torch.cat((featsem, batch_feats), dim=2)

        patch_feats = feats.repeat(1, 49, 1)
        batch_feats1 = feats.squeeze(1)
        avg_feats = batch_feats1


        return patch_feats, avg_feats
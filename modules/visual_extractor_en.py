import torch
import torch.nn as nn
import torchvision.models as models
import clip
from PIL import Image
import torch.nn.functional as F

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device,jit=False)
        self.model.float()
        with torch.no_grad():
            self.prompt = torch.load('prompt/prompt_en.pth').cuda()

        self.affine_aa = nn.Linear(768, 512).cuda()
    def forward(self, images):
        prompt= F.relu(self.affine_aa(self.prompt)).cuda()
        feature = []
        image = self.preprocess(Image.open(images)).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image)
        feature.append(image_features)
        '''
        for i in images:
            image = self.preprocess(Image.open(i)).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image)
            feature.append(image_features)
        '''
        batch_feats = torch.stack(feature, dim=0)
        prompt_add=[]
        for i in range(batch_feats.shape[0]):
            b=batch_feats[i].unsqueeze(1)
            b=b.repeat(prompt.shape[0], 1, 1).transpose(-2, -1)
            b = b.float()
            c_t = torch.bmm(prompt.float() , b)
            c_t = c_t.float()
            alpha = F.softmax(c_t)
            aa = alpha * prompt
            sum_a = aa.sum(axis=0)
            prompt_add.append(sum_a)
        prompt_feat=torch.stack(prompt_add, dim=0)
        feats=prompt_feat


        patch_feats = feats.repeat(1, 49, 1)
        avg_feats = feats.squeeze(1)

        avg_feats=avg_feats+batch_feats.squeeze(0)
       

        return avg_feats







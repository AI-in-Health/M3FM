import numpy as np
import torch
import torch.nn as nn
import pickle
from typing import Tuple
from transformers import GPT2LMHeadModel
from modules.decoder import DeCap
from medclip import MedCLIPModel, MedCLIPVisionModelViT
import math
import pdb


class MedCapModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(MedCapModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.tokenizer = tokenizer
        self.model = DeCap(args, tokenizer)

        self.align_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        self.align_model.from_pretrained()
        self.prompt = torch.load(args.prompt)
        if args.dataset == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def noise_injection(self, x, variance=0.001, modality_offset=None, dont_norm=False):
        if variance == 0.0:
            return x
        std = math.sqrt(variance)
        if not dont_norm:
            x = torch.nn.functional.normalize(x, dim=1)
        else:
            x = x + (torch.randn(x.shape) * std)  # todo by some conventions multivraiance noise should be devided by sqrt of dim
        if modality_offset is not None:
            x = x + modality_offset
        return torch.nn.functional.normalize(x, dim=1)

    def align_encode_images_iu_xray(self, images):
        # Split the images
        image1, image2 = images.unbind(dim=1)
        # Encode each image
        feature1 = self.align_model.encode_image(image1)
        feature2 = self.align_model.encode_image(image2)
        if self.args.prompt_load == 'yes':
            sim_1 = feature1 @ self.prompt.T.float()
            sim_1 = (sim_1 * 100).softmax(dim=-1)
            prefix_embedding_1 = sim_1 @ self.prompt.float()
            prefix_embedding_1 /= prefix_embedding_1.norm(dim=-1, keepdim=True)

            sim_2 = feature2 @ self.prompt.T.float()
            sim_2 = (sim_2 * 100).softmax(dim=-1)
            prefix_embedding_2 = sim_2 @ self.prompt.float()
            prefix_embedding_2 /= prefix_embedding_2.norm(dim=-1, keepdim=True)
            averaged_prompt_features = torch.mean(torch.stack([prefix_embedding_1, prefix_embedding_2]), dim=0)
            return averaged_prompt_features
        else:
            # Concatenate the features
            averaged_features = torch.mean(torch.stack([feature1, feature2]), dim=0)
            return averaged_features

    def align_encode_images_mimic_cxr(self, images):
        feature = self.align_model.encode_image(images)
        if self.args.prompt_load == 'yes':
            sim = feature @ self.prompt.T.float()
            sim = (sim * 100).softmax(dim=-1)
            prefix_embedding = sim @ self.prompt.float()
            prefix_embedding /= prefix_embedding.norm(dim=-1, keepdim=True)
            return prefix_embedding
        else:
            return feature

    def forward_iu_xray(self, reports_ids, align_ids, align_masks, images, mode='train', update_opts={}):
        self.align_model.to(self.device)
        self.align_model.eval()
        align_ids = align_ids.long()

        align_image_feature = None
        if self.args.train_mode == 'fine-tuning':
            align_image_feature = self.align_encode_images_iu_xray(images)
        if mode == 'train':
            align_text_feature = self.align_model.encode_text(align_ids, align_masks)
            if self.args.noise_inject == 'yes':
                align_text_feature = self.noise_injection(align_text_feature)

            if self.args.train_mode == 'fine-tuning':
                if self.args.F_version == 'v1':
                    combined_feature = torch.cat([align_text_feature, align_image_feature], dim=-1)
                    align_text_feature = self.fc_reduce_dim(combined_feature)
                if self.args.F_version == 'v2':
                    align_text_feature = align_image_feature

            outputs = self.model(align_text_feature, reports_ids, mode='forward')
            logits = outputs.logits
            logits = logits[:, :-1]
            return logits
        elif mode == 'sample':
            align_image_feature = self.align_encode_images_iu_xray(images)
            outputs = self.model(align_image_feature, reports_ids, mode='sample', update_opts=update_opts)
            return outputs
        else:
            raise ValueError

    def forward_mimic_cxr(self, reports_ids, align_ids, align_masks, images, mode='train', update_opts={}):
        self.align_model.to(self.device)
        self.align_model.eval()
        align_ids = align_ids.long()
        if mode == 'train':
            if self.args.noise_inject == 'yes':
                align_text_feature = self.align_model.encode_text(align_ids, align_masks)
                align_text_feature = self.noise_injection(align_text_feature)
            else:
                align_text_feature = self.align_model.encode_text(align_ids, align_masks)
            outputs = self.model(align_text_feature, reports_ids, mode='forward')
            logits = outputs.logits
            logits = logits[:, :-1]
            return logits
        elif mode == 'sample':
            align_image_feature = self.align_encode_images_mimic_cxr(images)
            outputs = self.model(align_image_feature, reports_ids, mode='sample', update_opts=update_opts)
            return outputs
        else:
            raise ValueError

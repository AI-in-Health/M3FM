import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from multilingual_clip import pt_multilingual_clip
import transformers

import clip
import json
import os
import torch
import torch.nn as nn
import re
import torch.nn.functional as F

class TextExtractor(nn.Module):
    def __init__(self, args):
        super(TextExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
        self.model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(self.model_name, device=self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, device=self.device)
        self.clean_report = self.clean_report_cov
        
        self.transformer = self.model.transformer
        self.LinearTransformation = self.model.LinearTransformation
        
        self.affine_aa = nn.Linear(768, 512).cuda()

        
    def forward(self, reports):
        if isinstance(reports, tuple):
            texts=[]
            for example in reports:
                example=self.clean_report(example)
                texts.append(example)
        else:
            texts=self.clean_report(reports)


        with torch.no_grad():
            txt_tok = self.tokenizer(texts, padding=True, return_tensors='pt').to(self.device)
            embs = self.transformer(**txt_tok)[0]
            att = txt_tok['attention_mask']
            embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
            embeddings = self.LinearTransformation(embs).cuda()


        

        embeddings = F.relu(self.affine_aa(embeddings)).cuda() #batch*768--》batch*512
        return embeddings #batch*512

    def clean_report_mimic_cxr(self,report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report


    def clean_report_cov(self,report):
        report_cleaner = lambda t: t.replace('\n', ' ').strip().lower()
        report=report_cleaner(report)
        return report


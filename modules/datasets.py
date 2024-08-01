import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import jieba
import re


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.clean_report = self.clean_report_cov

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            if self.examples[i]['label2'] == 1:
                self.examples[i]['ids'] = ([1] + tokenizer(self.clean_report(self.examples[i]['report']).split()))[:self.max_seq_length]
                self.examples[i]['ids1']=self.examples[i]['ids'] [:-1]
                #self.examples[i]['ids1'] = ([1] + tokenizer(self.examples[i]['report'])[:-1])[:self.max_seq_length - 1]


            else:
                gtt1 = ' '.join(self.examples[i]['report'])
                tokens = gtt1.split()
                #tokens = jieba.cut(self.clean_report(self.examples[i]['report']), cut_all=False)
                #self.examples[i]['ids'] = ([2] + tokenizer(self.examples[i]['report']))[:self.max_seq_length]
                #self.examples[i]['ids1'] = ([2] + tokenizer(self.examples[i]['report'])[:-1])[:self.max_seq_length - 1]
                self.examples[i]['ids'] = ([2] + tokenizer(tokens))[:self.max_seq_length]
                self.examples[i]['ids1'] = self.examples[i]['ids'] [:-1]


            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            self.examples[i]['reports'] = self.examples[i]['report']

    def __len__(self):
        return len(self.examples)

    '''
    def clean_report_mimic_cxr(self, report):
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
        '''
    def clean_report_cov(self,report):
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



class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_0_path = os.path.join(self.image_dir, image_path[0])
        image_1_path = os.path.join(self.image_dir, image_path[1])
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')

        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length,image_0_path,image_1_path)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_path_all=os.path.join(self.image_dir, image_path)
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        report = example['reports']
        reports_ids_use = example['ids1']
        seq_length = len(report_ids)
        seq_length1 = len(reports_ids_use)
        sample = (image_id, image, report_ids,report, seq_length,seq_length1, image_path_all, reports_ids_use)

        return sample

import pdb

import torch
from torch.utils.data import DataLoader
from .dataset import IuxrayMultiImageDataset, MimiccxrSingleImageDataset
from medclip import MedCLIPProcessor
import numpy as np

class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset
        self.batch_size = args.bs
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split
        self.processor = MedCLIPProcessor()

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, self.processor)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, self.processor)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        image_id_batch, image_batch, report_ids_batch, report_masks_batch, processor_ids_batch, processor_mask_batch, seq_lengths_batch, processor_lenghts_batch = zip(*data)
        image_batch = torch.stack(image_batch, 0)

        max_seq_length = max(seq_lengths_batch)
        target_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
        target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)

        max_processor_length = max(processor_lenghts_batch)
        target_processor_batch = np.zeros((len(processor_ids_batch), max_processor_length), dtype=int)
        target_processor_mask_batch = np.zeros((len(processor_mask_batch), max_processor_length), dtype=int)

        for i, report_ids in enumerate(report_ids_batch):
            target_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(report_masks_batch):
            target_masks_batch[i, :len(report_masks)] = report_masks

        for i, report_ids in enumerate(processor_ids_batch):
            target_processor_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(processor_mask_batch):
            target_processor_mask_batch[i, :len(report_masks)] = report_masks

        return image_id_batch, image_batch, torch.LongTensor(target_batch), torch.FloatTensor(target_masks_batch), torch.FloatTensor(target_processor_batch), torch.FloatTensor(target_processor_mask_batch)

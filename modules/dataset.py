import os
from PIL import Image
import json
from torch.utils.data import Dataset
import numpy as np
import torch


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, processor):
        self.image_dir = args.image_dir
        self.ann_path = args.json_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        self.processor = processor

    def preprocess_text(self, text):
        ids = self.tokenizer(text)[:self.max_seq_length]
        mask = [1] * len(ids)
        text_inputs = self.processor(text=text, return_tensors="pt",truncation=True, padding=False, max_length=self.max_seq_length)
        processor_ids = text_inputs['input_ids'].squeeze(0).tolist()
        processor_mask = text_inputs['attention_mask'].squeeze(0).tolist()
        return ids, mask, processor_ids, processor_mask

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        report = example['report']
        report_ids, report_masks, processor_ids, processor_mask = self.preprocess_text(report)

        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        # MedCLIP processing
        image_inputs_1 = self.processor(images=image_1, return_tensors="pt")
        image_inputs_2 = self.processor(images=image_2, return_tensors="pt")
        image = torch.stack((image_inputs_1.pixel_values[0], image_inputs_2.pixel_values[0]), 0)

        seq_length = len(report_ids)
        processor_length = len(processor_ids)
        sample = (image_id, image, report_ids, report_masks, processor_ids, processor_mask, seq_length, processor_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        report = example['report']
        report_ids, report_masks, processor_ids, processor_mask = self.preprocess_text(report)

        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_inputs = self.processor(images=image, return_tensors="pt")
        image = image_inputs.pixel_values[0]

        seq_length = len(report_ids)
        processor_length = len(processor_ids)
        sample = (image_id, image, report_ids, report_masks, processor_ids, processor_mask, seq_length, processor_length)
        return sample

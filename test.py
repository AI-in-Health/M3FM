import torch
import argparse
from modules.dataloader import R2DataLoader
from modules.tokenizers import Tokenizer
from modules.loss import compute_loss
from modules.metrics import compute_scores
from models.models import MedCapModel
from modules.tester import Tester
import numpy as np
import os
os.environ['CURL_CA_BUNDLE'] = ''
def main():
    parser = argparse.ArgumentParser()

    # Data input Settings
    parser.add_argument('--json_path', default='data/mimic_cxr/annotation.json',
                        help='Path to the json file')
    parser.add_argument('--image_dir', default='data/mimic_cxr/images/',
                        help='Directory of images')

    # Dataloader Settings
    parser.add_argument('--dataset', default='iu_xray', help='dataset for training MedCap')
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--max_seq_length', type=int, default=1024, help='the maximum sequence length of the reports.')

    #Trainer Settings
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--save_dir', type=str, default='results/mimic_cxr/', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='./record_dir/',
                        help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=1000, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Training related
    parser.add_argument('--noise_inject', default='no', choices=['yes', 'no'])

    # Sample related
    parser.add_argument('--sample_method', type=str, default='greedy', help='the sample methods to sample a report.')
    parser.add_argument('--prompt',default='/prompt/prompt.pt')
    parser.add_argument('--prompt_load', default='yes',choices=['yes','no'])

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=7e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98), help='the weight decay.')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--noamopt_warmup', type=int, default=5000, help='.')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9153, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--train_mode', default='base', choices=['base', 'full'],
                        help='Training mode: base (text only training) or full (full supervised training)')
    parser.add_argument('--full_supervised_version', default='v1', choices=['v1', 'v2' , 'v3'],
                        help='Full supervised version: v1 (only get image features) or v2 (feature fusion) or v3(feature fusion+image features')
    parser.add_argument('--clip_update', default='no' , choices=['yes','no'])
    parser.add_argument('--load', type=str, help='whether to load the pre-trained model.')


    args = parser.parse_args()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create tokenizer
    tokenizer = Tokenizer(args)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores
    model = MedCapModel(args, tokenizer)

    # build trainer and start to train
    tester = Tester(model, criterion, metrics, args, test_dataloader)
    tester.test()

if __name__ == '__main__':
    main()

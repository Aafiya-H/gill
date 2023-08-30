import sys
import argparse
import os
import random
import torch
import warnings

import pandas as pd
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler

from gill import models
from gill import utils
from gill import data
from transformers import AutoTokenizer

llm_models = ['facebook/opt-125m', 'facebook/opt-350m', 'facebook/opt-1.3b', 'facebook/opt-2.7b',
              'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b']


def parse_args(args):
   parser = argparse.ArgumentParser(description="Audio Caption training in GILL")
   parser.add_argument('--opt-version', default='facebook/opt-125m',
                      choices=llm_models,
                      help='OPT versions: ' +
                        ' | '.join(llm_models) +
                        ' (default: "facebook/opt-125m")')
   parser.add_argument('--visual-model', default='openai/clip-vit-large-patch14', type=str,
                      help="Visual encoder to use.")
   parser.add_argument("--audio-encoder",default="laion/clap-htsat-fused",type=str,
                       help="audio encoder to use")
   
   # parser.add_argument('--num-clip-tokens', default=77, type=int, metavar='N', help='Number of CLIP token to use for generation.')

   # add arguments for dataset

   parser.add_argument('--log-base-dir', default='./runs', type=str,
            help='Base directory to write logs and ckpts to.')
   
   parser.add_argument('--exp-name', default='frozen', type=str,
            help='Name of experiment, used for saving checkpoints.')
   
   parser.add_argument('--epochs', default=90, type=int, metavar='N',
            help='number of total epochs to run')
   # parser.add_argument('--steps_per_epoch', default=2000, type=int, metavar='N',
   #          help='number of training steps per epoch')
   # parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
   #          help='manual epoch number (useful on restarts)')
   # parser.add_argument('--val_steps_per_epoch', default=-1, type=int, metavar='N',
   #          help='number of validation steps per epoch')
   parser.add_argument('-b', '--batch-size', default=200, type=int,
            metavar='N',
            help='mini-batch size (default: 200), this is the total '
            'batch size of all GPUs on the current node when '
            'using Data Parallel or Distributed Data Parallel')

   parser.add_argument('-d', '--dataset', metavar='DATASET',  help='Dataset to train on', 
                       default='audiocaps', type=str)
   parser.add_argument('--dataset-dir', default='datasets/AudioCaps', type=str,
            help='Dataset directory containing .csv files.')
   parser.add_argument('--audio-dir', default='datasets/AudioCaps', type=str,
            help='Dataset directory containing .csv files.')
   parser.add_argument('--val-batch-size', default=None, type=int)
   parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
            metavar='LR', help='initial learning rate', dest='lr')
   parser.add_argument('--lr-warmup-steps', default=2000, type=int,
            metavar='N', help='Number of steps to warm up lr.')
   parser.add_argument('--lr_schedule_step_size', default=5, type=int,
            metavar='N', help='Number of steps before decaying lr.')
   parser.add_argument('--lr_schedule_gamma', default=0.1, type=float,
            metavar='N', help='Decay parameter for learning rate scheduler.')
   parser.add_argument('--grad-accumulation-steps', default=1, type=int, metavar='N',
                    help='number of gradient accumulation steps')
   parser.add_argument('--grad-clip', default=1.0, type=float, help='gradient clipping amount')
   parser.add_argument('--precision', default='bf16', type=str, choices=['fp32', 'fp16', 'bf16'],
                      help="What precision to train in.")
   
   # number of tokens
   parser.add_argument('--n-visual-tokens', default=4, type=int,
            metavar='N', help='Number of visual tokens to use for the Frozen model.')
   parser.add_argument('--n-audio-tokens',default=4,type=int,
                       metavar='N', help='Number of audio tokens to use for the Frozen model.')
   
   parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
            help='beta1 for Adam')
   parser.add_argument('--beta2', default=0.95, type=float, metavar='M',
            help='beta2 for Adam')
   parser.add_argument('--wd', '--weight-decay', default=0.01, type=float,
            metavar='W', help='weight decay (default: 0.01)',
            dest='weight_decay')
   parser.add_argument('-p', '--print-freq', default=10, type=int,
            metavar='N', help='print frequency (default: 10)')
   parser.add_argument('--resume', default='', type=str, metavar='PATH',
            help='path to latest checkpoint (default: none)')
   parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
            help='evaluate model on validation set')
#    parser.add_argument('--world-size', default=-1, type=int,
#             help='number of nodes for distributed training')
#    parser.add_argument('--rank', default=-1, type=int,
#             help='node rank for distributed training')
#    parser.add_argument('--dist-url', default='tcp://127.0.0.1:1337', type=str,
#             help='url used to set up distributed training')
#    parser.add_argument('--dist-backend', default='nccl', type=str,
#             help='distributed backend')
#    parser.add_argument('--seed', default=None, type=int,
#             help='seed for initializing training. ')
#    parser.add_argument('--gpu', default=None, type=int,
#             help='GPU id to use.')
#    parser.add_argument('--multiprocessing-distributed', action='store_true',
#             help='Use multi-processing distributed training to launch '
#                'N processes per node, which has N GPUs. This is the '
#                'fastest way to use PyTorch for either single node or '
#                'multi node data parallel training')
   return parser.parse_args(args)


def main(args):
   args = parse_args(args)
   args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
   while os.path.exists(args.log_dir):
      args.log_dir = os.path.join(args.log_base_dir, f'{args.exp_name}_{i}')
   os.makedirs(args.log_dir)

   with open(os.path.join(args.log_dir, f'git_info.txt'), 'w') as wf:
      utils.dump_git_status(out_file=wf)

   print(f'Logging to {args.log_dir}.')

   if args.seed is not None:
      random.seed(args.seed)
      torch.manual_seed(args.seed)
      cudnn.deterministic = True
      warnings.warn('You have chosen to seed training. '
               'This will turn on the CUDNN deterministic setting, '
               'which can slow down your training considerably! '
               'You may see unexpected behavior when restarting '
               'from checkpoints.')
      
   # if args.gpu is not None:
   #    warnings.warn('You have chosen a specific GPU. This will completely '
   #             'disable data parallelism.')

   # if args.dist_url == "env://" and args.world_size == -1:
   #    args.world_size = int(os.environ["WORLD_SIZE"])

   # args.distributed = args.world_size > 1 or args.multiprocessing_distributed

   ngpus_per_node = torch.cuda.device_count()
   # if args.multiprocessing_distributed:
   #    # Since we have ngpus_per_node processes per node, the total world_size
   #    # needs to be adjusted accordingly
   #    args.world_size = ngpus_per_node * args.world_size
   #    # Use torch.multiprocessing.spawn to launch distributed processes: the
   #    # main_worker process function
   #    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
   # else:
   #    # Simply call main_worker function
   #    main_worker(args.gpu, ngpus_per_node, args)
   main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
   model_args = models.GILLArgs()
   model_args.opt_version = args.opt_version
   model_args.visual_encoder = args.visual_model
   model_args.audio_encoder = args.audio_encoder

   model_args.freeze_lm = True
   model_args.freeze_vm = True
   model_args.freeze_am = True

   model_args.n_visual_tokens = args.n_visual_tokens
   model_args.n_audio_tokens = args.n_audio_tokens

   # model_args.ret_emb_dim = args.ret_emb_dim
   # model_args.gen_emb_dim = args.gen_emb_dim
   # model_args.text_fc_mode = args.text_fc_mode
   # model_args.ret_text_fc_mode = args.ret_text_fc_mode
   # model_args.num_tokens = args.num_tokens
   # model_args.num_clip_tokens = args.num_clip_tokens

   tokenizer = AutoTokenizer.from_pretrained(args.opt_version, use_fast=False)
   if tokenizer.pad_token is None:
      if args.opt_version in ['EleutherAI/gpt-j-6B']:
         tokenizer.pad_token = tokenizer.eos_token
      else:
         tokenizer.pad_token_id = tokenizer.eos_token_id
      print("tokenizer.pad_token, tokenizer.eos_token:", tokenizer.pad_token, tokenizer.eos_token)
   # Add an image token for loss masking (and visualization) purposes.
   # tokenizer.add_special_tokens({"cls_token": "<|image|>"})  # add special image token to tokenizer

   model = models.GILL(tokenizer,model_args)
   if args.precision == 'fp16':
      model = model.float()
   elif args.precision == 'bf16':
      model = model.bfloat16()

   criterion = nn.CrossEntropyLoss().cuda(args.gpu)
   optimizer_cls = torch.optim.AdamW
   print('Using torch.optim.AdamW as the optimizer.')
   optimizer = optimizer_cls(model.parameters(), args.lr,
                  betas=(args.beta1, args.beta2),
                  weight_decay=args.weight_decay,
                  eps=1e-8)
   
   scheduler_steplr = StepLR(optimizer, step_size=args.lr_schedule_step_size * args.steps_per_epoch, gamma=args.lr_schedule_gamma)
   scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.lr_warmup_steps, after_scheduler=scheduler_steplr)
   
   if args.resume:
      if os.path.isfile(args.resume):
         print("=> loading checkpoint '{}'".format(args.resume))
         checkpoint = torch.load(args.resume)
      # args.start_epoch = checkpoint["epoch"]
      model.load_state_dict(checkpoint['state_dict'], strict=False)
   
   cudnn.benchmark = True

   ## Data loading --> to be changed
   train_audio_df = pd.read_csv("gill/datasets/AudioCaps/train.csv")
   root_train = "gill/datasets/AudioCaps/train"
   downloaded_train_audio_df = pd.DataFrame()
   for audio_file_name in os.listdir(root_train):
      if audio_file_name[-4:] != ".wav":
         continue
      audio_file_name = audio_file_name[:-4]
      downloaded_train_audio_df = pd.concat([downloaded_train_audio_df,train_audio_df[train_audio_df["audiocap_id"]==int(audio_file_name)]],
                                             ignore_index = True)
      
   train_dataset = data.get_audio_dataset(args,"train",tokenizer)
   val_dataset = data.get_audio_dataset(args,"val",tokenizer)

   train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size = args.batch_size, shuffle = True)
   
   val_loader = torch.utils.data.DataLoader(
      val_dataset, batch_size = args.batch_size, shuffle = True)
   
   #training loop
   for epoch in range(args.epochs):

      #train for 1 epoch
      train(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args)
      stripped_state_dict = {
          k: v for k, v in model.state_dict().items() if 
          ('.lm' not in k and '.visual_model' not in k)
      }
      stripped_state_dict = OrderedDict(sorted(stripped_state_dict.items()))
      utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': stripped_state_dict,
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
      }, is_best, os.path.join(args.log_dir, 'ckpt'))
   

def train(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args):
   model.train()

   for i,(audio_features,tokenized_caption,caption_len) in enumerate(train_loader):
      audio_features = audio_features.cuda()
      tokenized_caption = tokenized_caption.cuda()
      

if __name__ == '__main__':
   main(sys.argv[1:])
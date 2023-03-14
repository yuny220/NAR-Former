import torch
from timm.utils import ModelEma
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from model import NetEncoder
from layers import DiffLoss
from utils import model_info
from parallel import DataParallelModel, DataParallelCriterion

def init_layers(args, LOG):
    # Model
    net = NetEncoder(args)

    if not args.do_train:
        return net

    lossMSE = torch.nn.MSELoss(reduction='mean')
    lossRank = DiffLoss(args.rankloss_type)
    lossConsistency = torch.nn.L1Loss()
    model_info(net, LOG)

    if torch.cuda.is_available():
        net = net.cuda(args.device)
        lossMSE = lossMSE.cuda(args.device)
        lossRank = lossRank.cuda(args.device)
        lossConsistency = lossConsistency.cuda(args.device)
        if args.parallel:
            net = DataParallelModel(net)
            lossMSE = DataParallelCriterion(lossMSE)
            lossRank = DataParallelCriterion(lossRank)
    
        # Model EMA
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
                            net,
                            decay=args.model_ema_decay,
                            device=args.device,
                            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)
    else:
        model_ema = None

    return net, model_ema, lossMSE, lossRank, lossConsistency

def init_optim(args, net, nbatches, warm_step=0.1):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_step * nbatches * args.max_epoch, num_training_steps=nbatches * args.max_epoch
    )
    print('warmup steps:', warm_step * nbatches * args.max_epoch)
    return optimizer, scheduler

def auto_load_model(args, model, model_ema=None, optimizer=None, scheduler=None):
    if args.do_train:
        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location="cpu".format(args.device))
            model.load_state_dict(checkpoint['state_dict'])
            print("Resume checkpoint %s" % args.resume)
            if hasattr(args, 'model_ema') and args.model_ema:
                if 'model_ema' in checkpoint.keys():
                    model_ema.ema.load_state_dict(checkpoint['state_dict_ema'])
                else:
                    pretrained_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
                    model_ema.ema.load_state_dict(pretrained_dict)
            if args.finetuning:
                print('Start fine-tuning from 0-th epoch/iter!')
                return 0
            else:
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
                print('With optim & ached!')
                if 'epoch' in checkpoint.keys():
                    start_id = checkpoint['epoch']
                elif 'iter' in checkpoint.keys():
                    start_id = checkpoint['iter']
        else:
            start_id = 0
        print('Start training from %d-th epoch/iter!' % (start_id))
        return start_id

    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path, map_location="cuda:{}".format(args.device))
        if 'state_dict_ema' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict_ema'])
        else:
            pretrained_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
            model.load_state_dict(pretrained_dict) 
        print(torch.load(args.pretrained_path, map_location="cuda:{}".format(args.device))['config'])
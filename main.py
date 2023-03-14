import os
import time
import torch
import random
import numpy as np
from timm.utils import get_state_dict

from utils import *
from config import argLoader
from data_process import init_dataloader

def format_second(secs):
    return "Exa(h:m:s):{:0>2}:{:0>2}:{:0>2}".format( \
            int(secs / 3600), int((secs % 3600) / 60), int(secs % 60))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(config):
    # Load Dataset
    train_loader,val_loader = init_dataloader(config, LOG)
    n_batches = len(train_loader)
    # Init Model
    net, model_ema, lossMSE, lossRank, lossConsistency = init_layers(config, LOG)
    # Optimizer
    optimizer, scheduler = init_optim(config, net, n_batches)
    # Auto Resume
    start_epoch_idx = auto_load_model(config, net, model_ema, optimizer, scheduler)

    # Init Value
    loss_mse, loss_rank, loss_con = 0.0, 0.0, 0.0
    best_tau, best_mape, best_error = -99, 1e5, 0
    if config.model_ema and config.model_ema_eval:
        best_tau_ema, best_mape_ema, best_error_ema = -99, 1e5, 0
    for epoch_idx in range(start_epoch_idx, config.max_epoch):
        metric = Metric()
        t0 = time.time()

        for batch_idx, batch_data in enumerate(train_loader):
            torch.cuda.empty_cache()
            net.train()      
            
            optimizer.zero_grad()
            if  'nasbench' in config.dataset:
                if config.lambda_consistency > 0:
                    data_0, data_1 = batch_data
                    codes0, v_acc0, t_acc0 = data_0
                    codes1, v_acc1, t_acc1 = data_1
                    codes, gt = torch.cat([codes0, codes1], dim=0), torch.cat([v_acc0, v_acc1], dim=0)
                else:
                    codes, v_acc, t_acc = batch_data
                    gt = v_acc
                codes, gt = codes.to(config.device), gt.to(config.device)
                logits = net(None, None, codes, None)

            elif config.dataset == 'nnlqp':
                codes, gt, sf = batch_data[0]["netcode"], batch_data[0]["cost"], batch_data[1]
                codes, gt, sf = codes.to(config.device), gt.to(config.device), sf.to(config.device)
                logits = net(None, None, codes, sf)

            pre = torch.cat([r.to(gt.device) for r in logits], dim=0) if isinstance(logits,list) else logits
            loss_mse = lossMSE(logits, gt) * config.lambda_mse
            if config.lambda_diff > 0:
                loss_rank = lossRank(logits, gt) * config.lambda_diff
            if config.lambda_consistency > 0:
                source_pre, auged_pre = torch.split(pre, pre.shape[0]//2, dim=0)
                loss_con = lossConsistency(source_pre, auged_pre) * config.lambda_consistency
            loss = loss_mse + loss_rank + loss_con
            loss.backward()
            optimizer.step()
            scheduler.step()

            if model_ema is not None:
                model_ema.update(net)

            ps = pre.data.cpu().numpy()[:, 0].tolist()
            gs = gt.data.cpu().numpy()[:, 0].tolist()
            metric.update(ps, gs)
            acc, err, tau = metric.get()

            if (batch_idx + 1) % n_batches == 0:
                t1 = time.time()
                speed = (t1 - t0) / (batch_idx + 1)
                exp_time = format_second(speed * (n_batches * (config.max_epoch - epoch_idx + 1) - batch_idx))

                lr = optimizer.state_dict()['param_groups'][0]['lr']
                LOG.log("Epoch[{}/{}]({}/{}) Lr:{:.7f} Loss:{:.7f} L_MSE:{:.7f} L_rank:{:.7f} L_con:{:.7f} KT:{:.5f} MAPE:{:.5f} " \
                            "ErrBnd(0.1):{:.5f} Speed:{:.2f} ms/iter {}" .format( \
                            epoch_idx, config.max_epoch, batch_idx, n_batches, lr, loss, loss_mse, loss_rank, \
                            loss_con, tau, acc, err, speed * 1000, exp_time))
            loss = None
        
        
        test_freq = 1
        if (epoch_idx+1) % test_freq == 0 :
            acc, err, tau = infer(val_loader, net, config.dataset, config.device)
            if tau > best_tau:
                best_mape, best_error, best_tau = acc, err, tau
                save_check_point(epoch_idx+1, batch_idx+1, config, net.state_dict(), None, None, False, config.dataset + '_model_best.pth.tar')
            
            if config.model_ema and config.model_ema_eval:
                acc_ema, err_ema, tau_ema = infer(val_loader, model_ema.ema, config.dataset, config.device)
                if tau_ema > best_tau_ema:
                    best_mape_ema, best_error_ema, best_tau_ema = acc_ema, err_ema, tau_ema
                    save_check_point(epoch_idx+1, batch_idx+1, config, get_state_dict(model_ema), None, None, False, config.dataset + '_model_best_ema.pth.tar')
            
            LOG.log('CheckPoint_TEST: KT {:.5f}, Best_KT {:.5f}, EMA_KT {:.5f}, Best_EMA_KT {:.5f} '\
                    'MAPE {:.5f}, Best_MAPE {:.5f}, EMA_MAPE {:.5f}, Best_EMA_MAPE {:.5f}, '\
                    'ErrBnd(0.1) {:.5f}, Best_ErrB {:.5f}, EMA_ErrBnd(0.1) {:.5f}, Best_EMA_ErrB {:.5f}, '
                     .format (tau, best_tau, tau_ema, best_tau_ema, acc, best_mape, acc_ema, best_mape_ema, err, best_error, err_ema, best_error_ema))
        
        if (epoch_idx + 1) % config.save_epoch_freq ==0:
            LOG.log('Saving Model after %d-th Epoch.' % (epoch_idx + 1))
            save_check_point(epoch_idx+1, batch_idx+1, config, net.state_dict(), optimizer, scheduler, False, config.dataset + '_checkpoint_Epoch' + str(epoch_idx + 1) + '.pth.tar')
        save_check_point(epoch_idx+1, batch_idx+1, config, net.state_dict(), optimizer, scheduler, False, config.dataset + '_latest.pth.tar')
    LOG.log('Training Finished! Best MAPE: %11.8f, Best ErrBnd(0.1): %11.8f; Best MAPE on EMA: %11.8f, Best ErrBond(0.1) on EMA: %11.8f' \
            % (best_mape, best_error, best_mape_ema, best_error_ema))


def infer(dataloader, net, dataset, device=None, isTest=False):
    metric = Metric()
    with torch.no_grad():
        net.eval()
        for bid, batch_data in enumerate(dataloader):
            if 'nasbench' in dataset:
                codes, v_accus, t_accus = batch_data
                gt = t_accus if isTest else v_accus
                logits = net(None, None, codes.to(device), None) if device!=None else net(None, None, codes, None)
            elif dataset == 'nnlqp':
                codes, gt, sf = batch_data[0]["netcode"], batch_data[0]["cost"], batch_data[1]
                logits = net(None, None, codes.to(device), sf.to(device)) if device!=None else net(ops, adjs, codes, None)
            pre = torch.cat([r.to(gt.device) for r in logits], dim=0) if isinstance(logits, list) else logits 
            ps = pre.data.cpu().numpy()[:, 0].tolist()
            gs = gt.data.cpu().numpy()[:, 0].tolist()
            metric.update(ps, gs)
            acc, err, tau = metric.get()
    return acc, err, tau


def test(config):
    test_loader = init_dataloader(config, LOG)
    net = init_layers(config, LOG)
    auto_load_model(config, net) 
    if torch.cuda.is_available():
        net = net.cuda(config.device)
    acc, err, tau = infer(test_loader, net, config.dataset, config.device, isTest=True)
    LOG.log('Test Finished! KT: %11.8f, MAPE: %11.8f, ErrBnd(0.1): %11.8f' \
            % (tau, acc, err))    

if __name__ == '__main__':
    args = argLoader()
    if  os.path.exists(args.save_path):
        print('Dir already exit! Please check it!')
        raise Exception('Dir already exit! Please check it!')
    else:
        os.makedirs(args.save_path)
    logname = args.save_path + '/logFile'
    LOG = mylog(logFile=logname, reset=False)

    #setup_seed(args.seed)

    print("Totally", torch.cuda.device_count(), "GPUs are available.")
    if args.parallel:
        print("Using data parallel.")
        for device in range(torch.cuda.device_count()):
            print("Device: ", device, "Name: ", torch.cuda.get_device_name(device))
    else:
        torch.cuda.set_device(args.device)
        print("Device: ", args.device, "Name: ", torch.cuda.get_device_name(args.device))

    if args.do_train:
        LOG.log('Configs: %s' % (args))
        train(args)
    else:
        test(args)
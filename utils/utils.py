import random
import shutil
import numpy as np
import torch
from scipy.stats import stats

def model_info(model, LOG):  
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    LOG.log('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        #if 'module.transformer.learnt_q' in name:
            #LOG.log('%50s %s' % (name, list(p[0,0,:].data)))
        LOG.log('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
        i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
        
    LOG.log('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


class Metric(object):
    def __init__(self):
        self.all = self.init_pack()

    def init_pack(self):
        return {
            'ps' : [],
            'gs' : [],
            'apes': [],                                # absolute percentage error
            'errbnd_cnt': np.array([0.0, 0.0, 0.0]),   # error bound count
            'errbnd_val': np.array([0.1, 0.05, 0.01]), # error bound value: 0.1, 0.05, 0.01
        }

    def update_pack(self, ps, gs, pack):
        for i in range(len(ps)):
            ape = np.abs(ps[i] - gs[i]) / gs[i]
            pack['errbnd_cnt'][ape <= pack['errbnd_val']] += 1
            pack['apes'].append(ape)
            pack['ps'].append(ps[i])
            pack['gs'].append(gs[i])

    def measure_pack(self, pack):
        acc = np.mean(pack['apes'])
        err = (pack['errbnd_cnt'] / len(pack['ps']))[0]
        tau = stats.kendalltau(pack['gs'], pack['ps']).correlation
        return acc, err, tau

    def update(self, ps, gs):
        self.update_pack(ps, gs, self.all)

    def get(self):
        return self.measure_pack(self.all)

def save_check_point(epoch, batch, config, state_dict, optimizer, scheduler, is_best, fileName='latest.pth.tar'):
    if optimizer and scheduler:
        state = {'epoch': epoch,
                'batch': batch,
                'config': config,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
    else:
        state = {'epoch': epoch,
                'batch': batch,
                'config': config,
                'state_dict': state_dict}
    path = config.save_path
    dataset = config.dataset
    torch.save(state, path + '/' + fileName)
    if is_best and epoch>(0.7*config.max_epoch):
        shutil.copyfile(path + '/' + fileName, path + '/' + dataset + '_model_best.pth.tar')
        shutil.copyfile(path + '/' + fileName, path + '/' + dataset + '_model_best_epoch_' + str(state['epoch']) + '.pth.tar')    
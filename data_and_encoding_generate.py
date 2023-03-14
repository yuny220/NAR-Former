from ast import arg
from operator import ne
import os
import numpy as np
import torch
import random
import argparse
import sys
from model.model import tokenizer

def argLoader():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nasbench101', help='dataset type')
    parser.add_argument('--data_path', type=str, default='nasbench101.json', help='path of json file')
    parser.add_argument('--save_dir', type=str, default='.', help='path of generated pt files')
    parser.add_argument('--n_percent', type=float, default=0.01, help='train proportion')
    parser.add_argument('--load_all', type=bool, default=True, help='load total dataset')
    parser.add_argument('--multires_x', type=int, default=32, help='dim of operation encoding')
    parser.add_argument('--multires_r', type=int, default=32, help='dim of topo encoding')
    parser.add_argument('--multires_p', type=int, default=32, help='dim of position encoding')
    parser.add_argument('--embed_type', type=str, default='nerf', help='Type of position embedding: nerf|trans')
    parser.add_argument('--split_type', type=str, default='GATES', help='GATES|TNASP')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import json
    r = random.random
    random.seed(2022)
    args = argLoader()
    percent = int(args.n_percent * 100)
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dx, dr, dp = args.multires_x, args.multires_r, args.multires_p
    if args.load_all:
        with open(args.data_path) as f:
            archs = json.load(f)
        num = len(archs)
        print('Total samples: %d' % num)
        data = {}
        for i in range(len(archs)):
            print('Loading %d/%d' % (i, num))
            if args.dataset == 'nasbench101':
                
                data[i] = {
                'index': i,
                'adj': archs[str(i)]['module_adjacency'],
                'ops': archs[str(i)]['module_operations'],
                'params': archs[str(i)]['parameters'],
                'validation_accuracy': archs[str(i)]['validation_accuracy'],
                'test_accuracy': archs[str(i)]['test_accuracy'],
                'training_time': archs[str(i)]['training_time'],
                'netcode': tokenizer(archs[str(i)]['module_operations'],archs[str(i)]['module_adjacency'],dx,dr,dp,args.embed_type)
                            }
            elif args.dataset == 'nasbench201':
                data[i] = {
                    'index': i,
                    'adj': archs[str(i)]['module_adjacency'],
                    'ops': archs[str(i)]['module_operations'],
                    'training_time': archs[str(i)]['training_time'],
                    'test_accuracy': archs[str(i)]['test_accuracy'],
                    'test_accuracy_avg': archs[str(i)]['test_accuracy_avg'],
                    'valid_accuracy': archs[str(i)]['validation_accuracy'],
                    'valid_accuracy_avg': archs[str(i)]['validation_accuracy_avg'],
                    'netcode' : tokenizer(archs[str(i)]['module_operations'], archs[str(i)]['module_adjacency'],dx,dr,dp,args.embed_type)
                    }
            print(len(data[i]['ops']))
        torch.save(data, os.path.join(save_dir, f'all_{args.dataset}.pt'))


    if not args.load_all:
        if args.dataset == 'nasbench101':
            torch.set_num_threads(1)
            train_data = {} 
            test_data = {} 
            val_data = {} 
            with open(args.data_path) as f:
                archs = json.load(f)
                print(len(archs))
                id_list = list(range(0, len(archs)))
                # Split dataset following TNASP
                if args.split_type == 'TNASP':
                    random.shuffle(id_list, random=r)
                    train_list = id_list
                    l1 = int(len(archs)*args.n_percent)
                    lv = int(len(archs)*(args.n_percent + 0.0005))
                    l2 = int(len(archs)*(args.n_percent + 0.0005)) #val 0.05%

                # Split dataset following GATES
                if args.split_type == 'GATES':
                    train_list = id_list[:int(len(archs)*0.9)]
                    random.shuffle(train_list, random=r)
                    l1 = int(len(archs)*0.9*args.n_percent)
                    lv = int(len(archs)*(0.9*args.n_percent + 0.0005))
                    l2 = int(len(archs)*0.9)
            
            for i in train_list[:l1]:
                idx = len(train_data)
                train_data[idx] = {
                    'index': i,
                    'adj': archs[str(i)]['module_adjacency'],
                    'ops': archs[str(i)]['module_operations'],
                    'params': archs[str(i)]['parameters'],
                    'validation_accuracy': archs[str(i)]['validation_accuracy'],
                    'test_accuracy': archs[str(i)]['test_accuracy'],
                    'training_time': archs[str(i)]['training_time'],
                    'netcode': tokenizer(archs[str(i)]['module_operations'],archs[str(i)]['module_adjacency'],dx,dr,dp,args.embed_type)
                }
            torch.save(train_data, os.path.join(save_dir, 'train.pt'))
            
            #for i in id_list[l1:l2]:
            for i in train_list[l1:lv]:
                idx = len(val_data)
                val_data[idx] = {
                    'index': i,
                    'adj': archs[str(i)]['module_adjacency'],
                    'ops': archs[str(i)]['module_operations'],
                    'params': archs[str(i)]['parameters'],
                    'validation_accuracy': archs[str(i)]['validation_accuracy'],
                    'test_accuracy': archs[str(i)]['test_accuracy'],
                    'training_time': archs[str(i)]['training_time'],
                    'netcode': tokenizer(archs[str(i)]['module_operations'],archs[str(i)]['module_adjacency'],dx,dr,dp,args.embed_type)
                }
            torch.save(val_data, os.path.join(save_dir, 'val.pt'))

            for i in id_list[l2:]:
                idx = len(test_data)
                test_data[idx] = {
                    'index': i,
                    'adj': archs[str(i)]['module_adjacency'],
                    'ops': archs[str(i)]['module_operations'],
                    'params': archs[str(i)]['parameters'],
                    'validation_accuracy': archs[str(i)]['validation_accuracy'],
                    'test_accuracy': archs[str(i)]['test_accuracy'],
                    'training_time': archs[str(i)]['training_time'],
                    'netcode': tokenizer(archs[str(i)]['module_operations'],archs[str(i)]['module_adjacency'],dx,dr,dp,args.embed_type)
                }
            torch.save(test_data, os.path.join(save_dir, 'test.pt'))

        elif args.dataset == 'nasbench201':
            with open(args.data_path) as f:
                archs = json.load(f)
                print(len(archs))
            id_list = list(range(0, len(archs)))
            # Split dataset following TNASP
            if args.split_type == 'TNASP':
                random.shuffle(id_list, random=r)
                train_list = id_list
                l1 = int(len(archs)*args.n_percent)
                lv = int(len(archs)*args.n_percent) + 200
                l2 = int(len(archs)*args.n_percent) + 200

            # Split dataset following GATES
            if args.split_type == 'GATES':
                train_list = id_list[:int(len(archs)*0.5)]
                random.shuffle(train_list, random=r)
                l1 = int(len(archs)*0.5*args.n_percent)
                #lv = int(len(archs)*(0.9*args.n_percent + 0.0005))
                l2 = int(len(archs)*0.5)

            train_data, test_data = {}, {}
            for i in train_list[:l1]:
                idx = len(train_data)
                train_data[idx] = {
                    'index': i,
                    'adj': archs[str(i)]['module_adjacency'],
                    'ops': archs[str(i)]['module_operations'],
                    'training_time': archs[str(i)]['training_time'],
                    'test_accuracy': archs[str(i)]['test_accuracy'],
                    'test_accuracy_avg': archs[str(i)]['test_accuracy_avg'],
                    'valid_accuracy': archs[str(i)]['validation_accuracy'],
                    '_accvaliduracy_avg': archs[str(i)]['validation_accuracy_avg'],
                    'netcode' : tokenizer(archs[str(i)]['module_operations'], archs[str(i)]['module_adjacency'],dx,dr,dp,args.embed_type)
                    }
            torch.save(train_data, os.path.join(save_dir, 'train.pt'))

            if args.split_type == 'TNASP':
                val_data = {}
                for i in train_list[l1:lv]:
                    print(i)
                    idx = len(val_data)
                    val_data[idx] = {
                        'index': i,
                        'adj': archs[str(i)]['module_adjacency'],
                        'ops': archs[str(i)]['module_operations'],
                        'training_time': archs[str(i)]['training_time'],
                        'test_accuracy': archs[str(i)]['test_accuracy'],
                        'test_accuracy_avg': archs[str(i)]['test_accuracy_avg'],
                        'valid_accuracy': archs[str(i)]['validation_accuracy'],
                        '_accvaliduracy_avg': archs[str(i)]['validation_accuracy_avg'],
                        'netcode' : tokenizer(archs[str(i)]['module_operations'], archs[str(i)]['module_adjacency'],dx,dr,dp,args.embed_type)
                        }
                torch.save(val_data, os.path.join(save_dir, 'val.pt'))

            for i in id_list[l2:]:
                print(i)
                idx = len(test_data)
                test_data[idx] = {
                    'index': i,
                    'adj': archs[str(i)]['module_adjacency'],
                    'ops': archs[str(i)]['module_operations'],
                    'training_time': archs[str(i)]['training_time'],
                    'test_accuracy': archs[str(i)]['test_accuracy'],
                    'test_accuracy_avg': archs[str(i)]['test_accuracy_avg'],
                    'valid_accuracy': archs[str(i)]['validation_accuracy'],
                    '_accvaliduracy_avg': archs[str(i)]['validation_accuracy_avg'],
                    'netcode' : tokenizer(archs[str(i)]['module_operations'], archs[str(i)]['module_adjacency'],dx,dr,dp,args.embed_type)
                    }
            torch.save(test_data, os.path.join(save_dir, 'test.pt'))
    
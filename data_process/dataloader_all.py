import os
import imp
from tkinter.tix import Tree

import torch
from .nnlqp import GraphLatencyDataset
from .nasbench import NasbenchDataset
from .fixed_length_sampler import FixedLengthBatchSampler


def init_dataset_NNLQP(data_path, test_model_type, override_data, embed_type, finetuning):
    data_root = os.path.join(data_path, "data")
    onnx_root = data_path
    all_latency_file = os.path.join(data_path, "gt.txt")

    model_types = set()
    for line in open(all_latency_file).readlines(): #gt.txt
        model_types.add(line.split()[4])
    test_types = test_model_type.split('|')
    test_model_types = set()
    for type in test_types:
        test_model_types.add(type)
    
    train_model_types = model_types - test_model_types
    assert len(train_model_types) > 0

    if finetuning:
        train_set = GraphLatencyDataset(data_root, onnx_root, all_latency_file, embed_type,
                                        override_data, test_model_types, n_finetuning=400)
        test_set = GraphLatencyDataset(data_root, onnx_root, all_latency_file, embed_type,
                                        override_data, test_model_types, n_finetuning=1600)
    else:
        train_set = GraphLatencyDataset(data_root, onnx_root, all_latency_file, embed_type, 
                                        model_types=train_model_types, override_data=override_data)
        test_set = GraphLatencyDataset(data_root, onnx_root, all_latency_file, embed_type, 
                                        model_types=test_model_types, override_data=override_data)


    return train_set, test_set, train_model_types, test_model_types


def init_dataloader(args, LOG):
    if 'nasbench' in args.dataset:
        if args.do_train:
            trainset = NasbenchDataset(LOG, args.dataset, "train", args.data_path, args.percent, args.lambda_consistency, args.aug_data_path)
            valset = NasbenchDataset(LOG, args.dataset, "val", args.data_path, args.percent)
            train_sampler = FixedLengthBatchSampler(trainset, args.dataset,args.batch_size, include_partial=True)
            val_sampler = FixedLengthBatchSampler(valset, args.dataset, args.batch_size, include_partial=True)
            train_loader = torch.utils.data.DataLoader(trainset, shuffle=(train_sampler is None), num_workers=args.n_workers, pin_memory=True, batch_sampler=train_sampler)
            val_loader = torch.utils.data.DataLoader(valset, shuffle=(val_sampler is None), num_workers=args.n_workers, pin_memory=True, batch_sampler=val_sampler)
            return train_loader, val_loader
        else:
            dataset = NasbenchDataset(LOG, args.dataset, "test", args.data_path)
            sampler = FixedLengthBatchSampler(dataset, args.dataset, args.batch_size, include_partial=True)
            dataLoader = torch.utils.data.DataLoader(dataset, shuffle=(sampler is None), num_workers=args.n_workers, pin_memory=True, batch_sampler=sampler)
            return dataLoader

    if args.dataset == 'nnlqp':
        trainset, testset, trtypes, tetypes = init_dataset_NNLQP(args.data_path, args.test_model_type, args.override_data, args.embed_type,args.finetuning)
        LOG.log("Train model types: {}".format(trtypes))
        LOG.log("Test model types: {}".format(tetypes))

        train_sampler = FixedLengthBatchSampler(trainset, args.dataset,args.batch_size, include_partial=True)
        test_sampler = FixedLengthBatchSampler(testset, args.dataset, args.batch_size, include_partial=True)

        train_loader = torch.utils.data.DataLoader(trainset, shuffle=(train_sampler is None), num_workers=args.n_workers, pin_memory=True, batch_sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(testset, shuffle=(test_sampler is None), num_workers=args.n_workers, pin_memory=True, batch_sampler=test_sampler)
        if args.do_train:
            return train_loader, test_loader
        else:
            return test_loader
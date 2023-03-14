import os
import onnx
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from .feature.graph_feature import extract_graph_feature

def get_torch_data(onnx_file, batch_size, cost_time, embed_type):
    node_features, topo_features, static_features, end_feature, depth_feature = extract_graph_feature(onnx_file, batch_size, embed_type)
    node_feats = torch.from_numpy(np.array(node_features, dtype=np.float32)).type(torch.float)
    topo_feats = torch.from_numpy(np.array(topo_features, dtype=np.float32)).type(torch.float)
    static_feats = torch.from_numpy(np.array(static_features, dtype=np.float32)).type(torch.float)
    end_feats = torch.from_numpy(np.array(end_feature, dtype=np.float32)).type(torch.float).view(1,-1)
    depth_feats = None if depth_feature.all()==None else torch.from_numpy(np.array(depth_feature, dtype=np.float32)).type(torch.float).view(1,-1)

    features = torch.cat([node_feats, topo_feats], dim=1)
    features = torch.cat([features, end_feats], dim=0) if depth_feats.all()==None else torch.cat([features, end_feats, depth_feats], dim=0)
    y = torch.FloatTensor([cost_time])
    data = {"netcode": features, "cost": y}
    return data, static_feats


class GraphLatencyDataset(Dataset):
    # specific a platform
    def __init__(self, root, onnx_dir, latency_file, embed_type, override_data=False,
                model_types=None, train_test_stage=None, n_finetuning=0, sample_num=-1):
        super(GraphLatencyDataset, self).__init__()
        self.data_root = root
        self.onnx_dir = onnx_dir #.../unseen_structre/
        self.latency_file = latency_file #gt.txt
        self.latency_ids = []
        self.override_data = override_data
        self.model_types = model_types
        self.train_test_stage = train_test_stage #=default=None
        self.embed_type = embed_type

        self.device = None
        print("Extract input data from onnx...")
        self.custom_process()
        print("Done.")

        if sample_num > 0:
            random.seed(1234)
            random.shuffle(self.latency_ids)
            self.latency_ids = self.latency_ids[:sample_num]
        random.seed(1234)
        random.shuffle(self.latency_ids)
        
        if n_finetuning > 0:
            self.latency_ids = self.latency_ids[:n_finetuning]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

    def custom_process(self):
        with open(self.latency_file) as f: #gt.txt
            for line in f.readlines():

                line = line.rstrip()
                items = line.split(" ")
                speed_id = str(items[0])
                graph_id = str(items[1])
                batch_size = int(items[2])
                cost_time = float(items[3])
                plt_id = int(items[5])

                if self.model_types and items[4] not in self.model_types:
                    continue

                if self.train_test_stage and items[6] != self.train_test_stage:
                    continue

                onnx_file = os.path.join(self.onnx_dir, graph_id) #self.onnx_dir: ../..dataset/unseen_structure/${graph_id}
                if os.path.exists(onnx_file):
                    data_file = os.path.join(self.data_root, '{}_{}_data.pt'.format(speed_id, plt_id))
                    sf_file = os.path.join(self.data_root, '{}_{}_sf.pt'.format(speed_id, plt_id))
                    graph_name = "{}_{}_{}".format(graph_id, batch_size, plt_id)
                    self.latency_ids.append((data_file, sf_file, graph_name, plt_id))
                    #Example: speed_id=00428, plt_id=1, batch_size=1
                    #data_file: ../../dataset/unseen_structure/data/00428_1_data.pt
                    #graph_name: onnx/nnmeter_alexnet/nnmeter_alexnet_transform_0427.onnx_1_1
                    
                    if (not self.override_data) and os.path.exists(data_file) and os.path.exists(sf_file):
                        continue

                    if len(self.latency_ids) % 1000 == 0:
                        print(len(self.latency_ids))

                    GG = onnx.load(onnx_file)
                    data, sf = get_torch_data(GG, batch_size, cost_time, self.embed_type) #process onnx file，cose_time:latency                                                                          

                    torch.save(data, data_file)
                    torch.save(sf, sf_file)


    def __len__(self):
        return len(self.latency_ids)

    def __getitem__(self, idx):
        data_file, sf_file, graph_name, plt_id = self.latency_ids[idx]
        data = torch.load(data_file)
        sf = torch.load(sf_file)
        return data, sf
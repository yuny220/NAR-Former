from logging import raiseExceptions
import torch
import torch.utils.data as Data

class NasbenchDataset(Data.Dataset):
    def __init__(self, LOG, dataset, part, data_path, percent=0, lbd_consistency=0, aug_data_path=None):
        self.part = part
        self.dataset = dataset
        self.consistency = lbd_consistency
        self.data_path = data_path
        self.aug_data_path = aug_data_path
        self.percent = percent
        LOG.log('Building dataset %s from orignial text documents' % self.part)
        self.data = self.load()
        if self.consistency > 0 :
            LOG.log('Finish Loading dataset %s; Number of samples pairs: %d' % (self.part, len(self.data)))
        else:
            LOG.log('Finish Loading dataset %s; Number of samples: %d' % (self.part, len(self.data)))

    def load(self):
        data_file = self.data_path
        datas = [torch.load(data_file)]

        if self.aug_data_path:
            datas.append(torch.load(self.aug_data_path))
            keys_aug = datas[1].keys()

        loaded_data = []
        data_num = int(self.percent) if self.percent>1 else int(len(datas[0])*self.percent)
        if self.part == 'train':
            keys = list(range(data_num))
        elif self.part == 'val':
            keys = list(range(data_num, data_num + 200))
        elif self.part == 'test':
            keys = list(range(len(datas[0]))) # test all
            #keys = list(range(data_num + 200, data_num+300)) # test 100
            #keys = list(range(data_num + 200, len(datas[0]))) # test rest
        
        for key in keys:
            # use augmentation and consistency loss
            if self.part=='train' and self.consistency > 0:
                if key in keys_aug:                   
                    loaded_data.append([datas[0][key], datas[1][key][1]])
                else:
                    loaded_data.append([datas[0][key], datas[0][key]])

            # use augmentation, no consistency loss     
            if self.part=='train' and (self.consistency == 0.) and self.aug_data_path:            
                loaded_data.append(datas[0][key])
                if key in keys_aug:
                    loaded_data.append(datas[1][key][1])

            # no augmentataion, no consistency loss, val and test
            if (self.part=='train' and not self.aug_data_path) or self.part=='val' or self.part=='test': 
                loaded_data.append(datas[0][key])

        return loaded_data

    def __getitem__(self, index):
        if self.consistency > 0 and self.part=='train':
            data_0 = self.data[index][0]
            data_1 = self.data[index][1]
            if self.dataset == 'nasbench101':
                return self.preprocess_101(data_0), self.preprocess_101(data_1)
            elif self.dataset == 'nasbench201':
                return self.preprocess_201(data_0), self.preprocess_201(data_1)
        else:
            if self.dataset == 'nasbench101':
                return self.preprocess_101(self.data[index])
            elif self.dataset == 'nasbench201':
                return self.preprocess_201(self.data[index])
    
    def preprocess_101(self, data):
        adj = torch.LongTensor(data['adj'])
        ops = torch.LongTensor(data['ops'])
        code = data['netcode']
        P = torch.tensor([data['params']], dtype=torch.float32)
        T = torch.tensor([data['training_time']], dtype=torch.float32)
        V_A = torch.tensor([data['validation_accuracy']], dtype=torch.float32)
        T_A = torch.tensor([data['test_accuracy']], dtype=torch.float32)
        #return ops, adj, code, P, T, V_A, T_A
        return code, V_A, T_A

    def preprocess_201(self, data):
        adj = torch.LongTensor(data['adj'])
        ops = torch.LongTensor(data['ops'])
        code = data['netcode']
        time = torch.tensor([data['training_time']])
        v_acc = torch.tensor([data['valid_accuracy']]) * 0.01
        v_acc_avg = torch.tensor([data['valid_accuracy_avg']]) * 0.01
        t_acc = torch.tensor([data['test_accuracy']]) * 0.01
        t_acc_avg = torch.tensor([data['test_accuracy_avg']]) * 0.01
        #return ops, adj, code, time, v_acc, v_acc_avg, t_acc, t_acc_avg
        return code, v_acc_avg, t_acc_avg
    
    def __len__(self):
        return len(self.data)
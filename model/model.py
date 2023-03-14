import torch
import torch.nn as nn
import torch.nn.functional as F 
from data_process.position_encoding import get_embedder
from layers.transformer import Encoder, RegHead

def tokenizer(ops, matrix, dim_x, dim_r, dim_p, embed_type): 
    # encode operation
    fn, _ = get_embedder(dim_x, embed_type=embed_type)
    code_ops_tmp = [fn(torch.tensor([op], dtype=torch.float32)) for op in ops]
    code_ops_tmp.append(fn(torch.tensor([1e5], dtype=torch.float32)))
    code_ops = torch.stack(code_ops_tmp, dim=0) #(len, dim_x)

    # encode self position
    fn, _ = get_embedder(dim_p, embed_type=embed_type)
    code_pos_tmp = [fn(torch.tensor([i], dtype=torch.float32)) for i in range(len(ops))]
    code_pos_tmp.append(fn(torch.tensor([1e5], dtype=torch.float32)))
    code_pos = torch.stack(code_pos_tmp, dim=0) #(len, dim_p)

    # encode data source of each node
    fn, _ = get_embedder(dim_r, embed_type=embed_type)
    code_sour_tmp = [fn(torch.tensor([-1], dtype=torch.float32))]
    for i in range(1, len(ops)):
        i_sour = 0
        for j in range(i):
            if matrix[j][i] == 1:
                i_sour += fn(torch.tensor([j], dtype=torch.float32))
        code_sour_tmp.append(i_sour)
    code_sour_tmp.append(fn(torch.tensor([1e5], dtype=torch.float32)))
    code_sour = torch.stack(code_sour_tmp, dim=0) #(len, dim_r)

    code = torch.cat([code_ops, code_pos, code_sour], dim=-1)
    return code
    

class NetEncoder(nn.Module):
    def __init__(self, config):
        super(NetEncoder, self).__init__()
        self.config = config  
        self.dim_x = self.config.multires_x
        self.dim_r = self.config.multires_r
        self.dim_p = self.config.multires_p
        self.embed_type = self.config.embed_type
        self.transformer = Encoder(config)
        self.mlp = RegHead(config)
        if config.use_extra_token:
            self.dep_map = nn.Linear(config.graph_d_model, config.graph_d_model)

    def forward(self, X, R, embeddings, static_feats):   
        # Get embedding     
        seqcode = embeddings #(b, l+1(end), d)
        
        # Depth token
        if self.config.use_extra_token:
            if  'nasbench' in self.config.dataset.lower():
                #depth = L
                depth = torch.full((seqcode.shape[0], 1, 1), fill_value=seqcode.shape[1]-1).to(seqcode.device)       
                depth_fn, _ = get_embedder(self.dim_x + self.dim_r + self.dim_p, self.embed_type)
                code_depth = F.relu(self.dep_map(depth_fn(depth))) #(b,1,d)
                seqcode = torch.cat([code_depth, seqcode], dim=1)
            elif 'nnlqp' in self.config.dataset.lower():
                code_rest, code_depth = torch.split(seqcode, [seqcode.shape[1]-1, 1], dim=1)
                code_depth = F.relu(self.dep_map(code_depth))
                seqcode = torch.cat([code_rest, code_depth], dim=1)
        
        aev = self.transformer(seqcode) #multi_stage:aev(b, 1, d)
        predict = self.mlp(aev, static_feats)
        return predict
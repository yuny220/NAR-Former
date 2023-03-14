import math
import torch
import torch.nn as nn
import numpy as np

def get_embedder(multires, embed_type='nerf', input_type='tensor'):
    embed_kwargs = {
                'input_type' : input_type,
                'embedding_type' : embed_type,
                'input_dims' : 1,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
    }
    if input_type=='tensor': 
        embed_kwargs['periodic_fns'] = [torch.sin, torch.cos] 
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed_tensor(x)
    else:
        embed_kwargs['periodic_fns'] = [np.sin, np.cos]
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
    
    return embed, embedder_obj.out_dim

class Embedder():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
            
        max_freq = self.kwargs['max_freq_log2'] # max_freq=multires-1
        N_freqs = self.kwargs['num_freqs'] # N_freqs=multires
        
        dty = self.kwargs['input_type']
        if self.kwargs['embedding_type'] == 'nerf':
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs) if dty=='tensor'\
                            else np.linspace(2.**0., 2.**max_freq, num=N_freqs)
            
            for freq in freq_bands:
                for p_fn in self.kwargs['periodic_fns']: #p_fn=torch.sin, p_fn=torch.cos
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * math.pi * freq))
                    out_dim += d 
        
        elif self.kwargs['embedding_type'] == 'trans':
            dim = self.kwargs['num_freqs']
            freq_bands = [ 1 / (10000**(j/dim)) for j in range(dim)]
            for freq in freq_bands: #
                for p_fn in self.kwargs['periodic_fns']: #p_fn=torch.sin, p_fn=torch.cos
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                    out_dim += d 

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed_tensor(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
    def embed(self, inputs):
        return np.concatenate([fn(inputs) for fn in self.embed_fns])
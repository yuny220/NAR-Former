from operator import length_hint
import numpy as np
from .feature_utils import FEATURE_LENGTH, FEATURE_DIM
from .node_feature import EmbedValue
from data_process.position_encoding import get_embedder

def extract_topology_feature(nx_G, name2id, id2name, embed_type='nerf', input_type='np_array'):
    length = FEATURE_LENGTH["topology"]
    total_dim = FEATURE_DIM["topology"]
    dim = total_dim // length
    fn, _ = get_embedder(dim//2, embed_type, input_type)
    source_emb = fn(EmbedValue.embed_int(-1))
    selfpos_emb = fn(EmbedValue.embed_int(0))
    embedding = [np.concatenate([source_emb, selfpos_emb])]

    for i in range(1, nx_G.number_of_nodes()):
        node = id2name[i]
        feat_s = np.zeros(dim, dtype="float32")
        for child in nx_G.predecessors(node):
            idx = name2id[child]
            feat_s += fn(EmbedValue.embed_int(idx))
        feat_sp = fn(EmbedValue.embed_int(i))
        embedding.append(np.concatenate([feat_s, feat_sp]))

    return embedding

def extract_static_feature(static_info, embed_type='nerf', input_type='np_array'):
    length = FEATURE_LENGTH["static"]
    total_dim = FEATURE_DIM["static"]
    dim = total_dim // length
    fn, _ = get_embedder(dim//2, embed_type, input_type)

    feats = []
    #batch size; int
    feat = fn(EmbedValue.embed_int(static_info[0]))
    feats.append(feat)
    #flops, params, macs; float
    for info in static_info[1:]:
        feat = fn(EmbedValue.embed_float(info))
        feats.append(feat)
    embeddings = np.concatenate(feats)
    return embeddings

def extract_depthtoken_feature(depth, embed_type='nerf', input_type='np_array'):
    dims = [FEATURE_DIM[key] for key in list(FEATURE_DIM.keys())[:-1]]
    total_dim = sum(dims)
    fn, _ = get_embedder(total_dim//2, embed_type, input_type)
    embedding = fn(EmbedValue.embed_int(depth))
    return embedding

def extract_end_feature(end=1e5, embed_type='nerf', input_type='np_array'):
    embeddings = []
    #use 1e5 as end for each part
    for part in list(FEATURE_LENGTH.keys())[:-1]:
        for elem in range(FEATURE_LENGTH[part]):
            fn, _ = get_embedder(FEATURE_DIM[part]//FEATURE_LENGTH[part]//2, embed_type, input_type)
            embeddings.append(fn(EmbedValue.embed_int(end)))
    embeddings = np.concatenate(embeddings)
    return embeddings
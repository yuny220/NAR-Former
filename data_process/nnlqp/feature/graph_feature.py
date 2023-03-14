import numpy as np
import networkx as nx
from .onnx_to_networkx import onnx2nx
from .onnx_shape_infer import custom_shape_infer
from .onnx_flops import calculate_onnx_flops
from .node_feature import extract_node_features
from .other_feature import *

def modify_onnx_batch_size(onnx_G, batch_size): #loaded onnx file； batch_size=1
    # initializer names, in case of names of input include initializers
    init_names = set()
    for init in onnx_G.graph.initializer:
        init_names.add(init.name)

    def _modify(node, dim_idx, value):
        dims = node.type.tensor_type.shape.dim
        if len(dims) > dim_idx:
            value = value[node.name] if isinstance(value, dict) else value
            dims[dim_idx].dim_value = value

    # modify input
    for inp in onnx_G.graph.input:
        if inp.name in init_names:
            continue
        _modify(inp, 0, batch_size)

    # modify output
    for out in onnx_G.graph.output:
        _modify(out, 0, batch_size)

    return


def parse_from_onnx(onnx_path, batch_size):
    pG = onnx2nx(onnx_path) #pG=PGraph(nx_G, input_sizes, output_sizes, switcher.opsets, onnx_G)
    nx_G, onnx_G = pG.data, pG.onnx_G #nx_G:networkx graph, onnx_G:used to get output shape and other static fetures
    
    #draw_graph(nx_G)
    # first we should change the batch_size of input in ONNX model
    modify_onnx_batch_size(onnx_G, batch_size)
    status, newG, output_shapes = custom_shape_infer(onnx_G)
    # if failed modify the batch to original batch size
    assert status is True, "Onnx shape infer error!"

    flops, params, macs, node_flops = calculate_onnx_flops(onnx_G, True)
    return nx_G, output_shapes, flops, params, macs, node_flops, newG


def extract_graph_feature_from_networkx(nx_G, batch_size, output_shapes, flops, params, macs, embed_type, undirected=False, use_extra_token=True): 
    #(1) node features
    node_features = extract_node_features(nx_G, output_shapes, embed_type) ##nx_G:networkx graph

    # get features conducted by idx
    features = []
    name2id = {}
    id2name = {}
    for idx, node in enumerate(nx.topological_sort(nx_G)): #len(features) = num_nodes = len(adjacent)
        features.append(node_features[node])
        name2id[node] = idx
        id2name[idx] = node

    #(2) topo_features
    # get topology features
    topo_features = extract_topology_feature(nx_G, name2id, id2name, embed_type)

    #(3) static features: flops, params, memory_access (GB) + batch_size
    static_info = [batch_size, flops / 1e9, params / 1e9, macs / 1e9]
    static_features = extract_static_feature(static_info, embed_type)
    # test connect relationship
    # xs, ys = np.where(adjacent > 0)
    # for i in range(len(xs)):
    #     print("Conn:", id2name[xs[i]], id2name[ys[i]])

    # feature in features may be a tuple (block_adjacent, block_features, block_static_features)

    #(3) end feature
    end_feature = extract_end_feature(embed_type=embed_type)

    #(4) if use extra token
    depthtoken_feature = extract_depthtoken_feature(len(features), embed_type='trans') if use_extra_token else None
    return features, topo_features, static_features, end_feature, depthtoken_feature


def extract_graph_feature(onnx_path, batch_size, embed_type, return_onnx=False):
    nx_G, output_shapes, flops, params, macs, node_flops, onnx_G = parse_from_onnx(onnx_path, batch_size)
    node_features, topo_features, static_features, end_feature, depth_feature = extract_graph_feature_from_networkx(
        nx_G, batch_size, output_shapes, flops, params, macs, embed_type
    )

    if return_onnx:
        return node_features, topo_features, static_features, end_feature, depth_feature, onnx_G
    else:
        return node_features, topo_features, static_features, end_feature, depth_feature

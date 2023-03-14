#https://blog.csdn.net/jokerxsy/article/details/109733852

import numpy as np
import torch
from torch.utils.data import Sampler
class FixedLengthBatchSampler(Sampler):

    def __init__(self, data_source, dataset, batch_size, include_partial=False, rng=None, maxlen=None,
                 length_to_size=None):
        self.data_source = data_source
        self.dataset = dataset
        self.active = False
        if rng is None:
            rng = np.random.RandomState(seed=11)
        self.rng = rng
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.include_partial = include_partial
        self.length_to_size = length_to_size
        self._batch_size_cache = { 0: self.batch_size }
        self.length_map = self.get_length_map()
        self.reset()
    def get_length_map(self):
        '''
        Create a map of {length: List[example_id]} and maintain how much of
        each list has been seen.
        '''
        # Record the lengths of each example.
        length_map = {} #{70:[0, 23, 3332, ...], 110:[3, 421, 555, ...], length:[dataidx_0, dataidx_1, ...]}
        for i in range(len(self.data_source)):
            if self.dataset == 'nnlqp':
                length = self.data_source[i][0]['netcode'].shape[0]
            elif self.dataset == 'nasbench201' or 'nasbench101':
                length = self.data_source[i][0][0].shape[0] if len(self.data_source[i])==2 else self.data_source[i][0].shape[0]
            if self.maxlen is not None and self.maxlen > 0 and length > self.maxlen:
                continue
            length_map.setdefault(length, []).append(i)
        return length_map

    def get_batch_size(self, length):
        if self.length_to_size is None:
            return self.batch_size
        if length in self._batch_size_cache:
            return self._batch_size_cache[length]
        start = max(self._batch_size_cache.keys())
        batch_size = self._batch_size_cache[start]
        for n in range(start+1, length+1):
            if n in self.length_to_size:
                batch_size = self.length_to_size[n]
            self._batch_size_cache[n] = batch_size
        return batch_size

    def reset(self):
        """

        If include_partial is False, then do not provide batches that are below
        the batch_size.

        If length_to_size is set, then batch size is determined by length.

        """
        # Shuffle the order.
        for length in self.length_map.keys():
            self.rng.shuffle(self.length_map[length])

        # Initialize state.
        state = {} #e.g. {70(length):{'nbatches':3(num_batch), 'surplus':True, 'position':-1}}
        for length, arr in self.length_map.items():
            batch_size = self.get_batch_size(length)
            nbatches = len(arr) // batch_size
            surplus = len(arr) % batch_size
            state[length] = dict(nbatches=nbatches, surplus=surplus, position=-1)

        # Batch order, in terms of length.
        order = [] #[70, 70, 70, 110, ...] length list
        for length, v in state.items():
            order += [length] * v['nbatches']

        ## Optionally, add partial batches.
        if self.include_partial:
            for length, v in state.items():
                if v['surplus'] >= torch.cuda.device_count():
                    order += [length]

        self.rng.shuffle(order)

        self.length_map = self.length_map
        self.state = state
        self.order = order
        self.index = -1

    def get_next_batch(self):
        index = self.index + 1
        length = self.order[index]
        batch_size = self.get_batch_size(length)
        position = self.state[length]['position'] + 1
        start = position * batch_size
        batch_index = self.length_map[length][start:start+batch_size]

        self.state[length]['position'] = position
        self.index = index
        return batch_index

    def __iter__(self):
        self.reset()
        for _ in range(len(self)):
            yield self.get_next_batch()

    def __len__(self):
        return len(self.order)

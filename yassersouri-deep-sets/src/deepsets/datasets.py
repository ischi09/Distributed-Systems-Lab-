from typing import Callable, Any
import random

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

class SetDataset(Dataset):
    def __init__(self, n_samples: int, max_set_size: int, min_value: int, max_value: int, label_generator: Callable[[Any], Any], generate_multiset=False):
        self.n_samples = n_samples
        self.label_generator = label_generator
        self.items = []

        for _ in range(n_samples):
            self.items.append(torch.tensor(np.random.choice(np.arange(min_value, max_value) , (random.randint(1,max_set_size), 1), replace=generate_multiset)))
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        elem = self.items[item]
        return elem, self.label_generator(elem) 



        

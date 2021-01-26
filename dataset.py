from __future__ import print_function
import torch.utils.data as data
import os.path
import torch
import numpy as np
import os
import pickle
import random
import json

import morphologies
from utils import *

class Morphology_Dataset(data.Dataset):
    def __init__(self, morphology_data,train = True,classify=False):
        self.train = train
        self.classify = classify
        self.morphology_data = morphology_data

        self.res = []
        self.morphs = []
        self.vals = []

        for data in morphology_data:
            # mod_morph = []
            # for rot in data["morph"]:
            #     if rot == -1:
            #         mod_morph.append(1)
            #     else:
            #         mod_morph.append(rot)
            if len(data["val_means"]) > 0:
                if self.classify:
                    self.res.append([data["morph"],val_to_class(data["val_means"][-1])])
                    # c = val_to_class(data["val_means"][-1])
                    # if c == 1:
                    #     for i in range(10):
                    #         self.res.append([data["morph"],val_to_class(data["val_means"][-1])])
                    # else:
                    #     self.res.append([data["morph"],val_to_class(data["val_means"][-1])])
                else:
                    #self.res.append([data["morph"],np.mean(data["val_means"])])
                    #penalty = rotor_penalty(data["morph"])
                    self.res.append([data["morph"],data["val_means"][-1]])

        #self.train_len = int(len(self.res) * 0.8)
        self.train_len = len(self.res)
        random.seed(8)
        random.shuffle(self.res) #randomize order
        random.seed()
        if train:
            for i in range(self.train_len):
                self.morphs.append(self.res[i][0])
                self.vals.append(self.res[i][1])
        else:
            for i in range(self.train_len,len(self.res)):
                self.morphs.append(self.res[i][0])
                self.vals.append(self.res[i][1])

        self.ValueMeter = AverageValueMeter()

    def __getitem__(self, index):
        key = torch.Tensor(self.res[index][0])
        val = self.res[index][1]
        morph = thruster_key_to_morph(key)
        return morph,val
        # if self.classify:
        #     return morph, val
        # else:
        #     return morph, val


    def __len__(self):
        if self.train:
            return self.train_len
        else:
            test_len = int(len(self.res)) - self.train_len
            return test_len

if __name__  == '__main__':
    print('Testing Morphology dataset')
    with open('morph_data.json') as f:
        morph_dict = json.load(f)

    d = Morphology_Dataset(morph_dict, train=True)
    morph, val = d.__getitem__(12)
    print(morph)
    print(val)
    new_val = val - rotor_penalty(morph,0.01)
    print(new_val)
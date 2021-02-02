#
#   Binds REP socket to tcp://*:5555
#

import time
import zmq
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import json
import re
from geneticalgorithm_mod import geneticalgorithm as ga
import morphologies
from utils import thruster_keys_to_string

parser = argparse.ArgumentParser()

parser.add_argument('-v', '--verbose', action='store_true', default = False,
                    dest='simple_value', help='Print information when receiving data')

result = parser.parse_args()

def pairServer(train,loadfile='morph_data.json'):
    context = zmq.Context()
    global socket
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:5555")
    print("python server started")
    global Morphologies
    Morphologies = morphologies.Morphologies(train=True,nepoch=0)

    algorithm_param = {'max_num_iteration': 16000,\
                       'population_size':20,\
                       'mutation_probability':0.1,\
                       'elit_ratio': 0.01,\
                       'crossover_probability': 0.5,\
                       'parents_portion': 0.3,\
                       'crossover_type':'uniform',\
                       'max_iteration_without_improv':None}

    varbound = np.array([[-1,1]]*1608)
    vartype = np.array(np.concatenate([np.array([['int']]*8),np.array([['real']]*1600)]))
    model=ga(function=ga_f,dimension=1608,variable_type_mixed=vartype,variable_boundaries=varbound,algorithm_parameters=algorithm_param,function_timeout=600)
    model.run()
    smsg = "End"
    socket.send(smsg.encode('utf-8'))
    Morphologies.save_to_file()


def ga_f(X):
    pop_s = len(X)
    #smsg = np.array2string(X,max_line_width=np.inf,threshold =2000)
    smsg = thruster_keys_to_string(X)
    socket.send(smsg.encode('utf-8'))
    objs = np.zeros(pop_s)
    for i in range(pop_s):
        rmsg = socket.recv().decode()
        key, mean, episode = handleMessage(rmsg)
        print("Episode %d Key %s with Val = %f"%(episode, str(key), mean))
        for data in Morphologies.morph_data:
            if data["morph"] == key:
                    data["val_means"].append(mean)
                    data["episodes"].append(episode)
        objs[i] = -mean
    return objs

def handleMessage(message):
    parts = message.split(',')
    morph = parts[0]
    mean = float(parts[1])
    episode = int(parts[2])
    key = [int(val) for val in re.findall(r'-?\d+', morph)]
    return key, mean, episode

def main():
    start = time.time()
    pairServer(train=False,loadfile='old_morph_datas/morph_data_pen-11.json')
    end = time.time()
    print("%f seconds"%(end-start))

if __name__ == "__main__":
    main()

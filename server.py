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
import threading
import json
import rotor_conv
import morphologies
import re

parser = argparse.ArgumentParser()

parser.add_argument('-v', '--verbose', action='store_true', default = False,
                    dest='simple_value', help='Print information when receiving data')

result = parser.parse_args()

# #Communicate with Python to send data
# def sendData(data):
#     context = zmq.Context()
#     socket = context.socket(zmq.REQ)
#     socket.connect("tcp://localhost:5555")
#     print('connected')
#     socket.send(data.encode('utf-8'))
#     message = socket.recv()
#     print(message.decode('utf-8'))

# #Communicate with Python to fetch data
# def receiveData():
#     context = zmq.Context()
#     socket = context.socket(zmq.REP)
#     socket.RCVTIMEO = 10000
#     #socket.setsockopt( zmq.LINGER, 0)
#     socket.setsockopt( zmq.RCVTIMEO, 10)

#     socket.bind("tcp://*:5556")
#     print("python server started")

#     while True:
#         #  Wait for next request from client
#         message = socket.recv()
#         print(message)
#         socket.send("Python ACK".encode('utf-8'))
#         #plaintext
#         content = message.decode('utf-8')

def pairServer(train,loadfile='morph_data.json'):
    Morphologies = morphologies.Morphologies(train=train,nepoch=5)
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:5555")
    print("python server started")

    if train:
        Morphologies.predict_morph_values()
        Morphologies.save_to_file()
    else:
        with open(loadfile) as f:
            morph_dict = json.load(f)
        Morphologies.morph_data = morph_dict
    thruster_keys = Morphologies.choose_morphologies()
    socket.send(thruster_keys.encode('utf-8'))

    while True:
        #  Wait for next request from client
        message = socket.recv().decode()
        if message == "End":
            print("Exiting")
            if train:
                Morphologies.save_to_file()
            break
        elif message == "New Episode":
            if train:
                Morphologies.predict_morph_values()
                Morphologies.save_to_file()
            thruster_keys = Morphologies.choose_morphologies()
            socket.send(thruster_keys.encode('utf-8'))
            if train:
                Morphologies.episode += 1
                Morphologies.gain = min(1,Morphologies.gain+5e-5)
                Morphologies.train_conv_net()
                print('^^^ Episode %d    Efficiency Penalty Gain: %f \n' %(Morphologies.episode,Morphologies.gain))
        else:
            if train:
                key, mean, std, n, episode = handleMessage(message)
                for data in Morphologies.morph_data:
                    if data["morph"] == key:
                            data["val_means"].append(mean)
                            data["val_stds"].append(std)
                            data["sample_num"].append(n)
                            data["episodes"].append(episode)
                        #print(data)

def handleMessage(message):
    parts = message.split(',')
    morph = parts[0]
    mean = float(parts[1])
    std = float(parts[2])
    n = float(parts[3])
    episode = int(parts[4])
    key = [int(val) for val in re.findall(r'-?\d+', morph)]
    return key, mean, std, n, episode

def main():
    pairServer(train=True,loadfile='old_morph_datas/morph_data_12_18.json')

if __name__ == "__main__":
    main()

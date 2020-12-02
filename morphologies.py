import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import server
import json
import random
import time, datetime
import sys, os

from dataset import *
from utils import *
from rotor_conv import *


class Morphologies():       
    def __init__(self,model='',morph_path = './morph_data.json',train=True,nepoch=5,classify=False):

        self.num_agents = 20
        self.batch_size = 16
        self.train = train
        self.classify = classify
        #-----Choose Morphology Params-----#
        self.epsilon = 0.75
        self.gamma = 0.99
        self.episode = 0
        self.nepoch = nepoch
        #--------------------#

        #3x3x2 binary image map
        #config = torch.zeros([3,3,2],dtype=torch.uint8)
        #stack_list = [config]

        self.morph_data = []
        # ===================CREATE network================================= #
        self.network = RotorConvNet(self.batch_size,classify=self.classify)
        self.network.cuda() #put network on GPU
        self.morph_path = morph_path
        #self.network.apply(weights_init) #initialization of the weight

        if model != '':
            self.network.load_state_dict(torch.load(model))
            print(" Previous weight loaded ")
        # ========================================================== #
        #self.morphs = torch.stack(morphs)

        if self.train or not os.path.exists(self.morph_path):
            morph_keys = []
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            for m in range(3):
                                for n in range(3):
                                    for o in range(3):
                                        for p in range(3):
                                            morph_keys.append([i-1,j-1,k-1,l-1,m-1,n-1,o-1,p-1])

            for morph_key in morph_keys:
                self.morph_data.append({"morph" : morph_key, "val_means" : [], "val_stds": [], "sample_num" : [], "episodes" : [], "pred_val": 0, "pred_history": []})

            # =============Set up conv net logging====================== #
            now = datetime.datetime.now()
            save_path = now.isoformat().replace(":","_")
            self.top_dir = os.path.join('train_log', save_path)
            if not os.path.exists(self.top_dir):
                os.mkdir(self.top_dir)           
            # ========================================================== #
        else:
            with open(self.morph_path) as f:
                self.morph_data = json.load(f)


    def choose_random_morphologies(self,num):
        #random test
        keys = []
        for i in range(num):
            rand_index = random.randint(0,len(self.morph_data)-1)
            morph = self.morph_data[rand_index]["morph"]
            keys.append(morph)
        keys_str = thruster_keys_to_string(keys)
        return key_str

    def choose_morphologies(self):
        num_fixed = 0
        fixed_morph = [1,0,-1,0,1,0,-1,0]        
        fixed_morphs = [fixed_morph for i in range(num_fixed)]
        sorted_data = sorted(self.morph_data, key = lambda i: i['pred_val'],reverse=True)
        chosen_morphs = []
        chosen_index = 0

        if self.train:
            for i in range(self.num_agents - num_fixed):
                ep = random.uniform(0,1)
                if (ep < self.epsilon or sorted_data[0] == 0):
                    rand_index = random.randint(0,len(self.morph_data)-1)
                    morph = self.morph_data[rand_index]["morph"]
                    rotated_key = rotate_to_ref(morph)
                    chosen_morphs.append(rotated_key)
                else:
                    rotated_key = rotate_to_ref(sorted_data[chosen_index]["morph"])
                    chosen_morphs.append(rotated_key)
                    chosen_index += 1
            self.epsilon = max(self.epsilon - 0.025, 0.025)
            keys = fixed_morphs + chosen_morphs
        else:
            sorted_data_len = sorted(self.morph_data, key =lambda i: len(i['val_means']),reverse=True)
            for i in range(self.num_agents):
                rotated_key = rotate_to_ref(sorted_data_len[chosen_index]["morph"])
                chosen_morphs.append(rotated_key)
                chosen_index += 1
            keys = chosen_morphs
        return thruster_keys_to_string(keys)

    def train_conv_net(self):
        WORKERS = 0
        dir_name = os.path.join(self.top_dir, str(self.episode))
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        logname = os.path.join(dir_name, 'log.txt')
        manualSeed = random.randint(1, 10000) # fix seed
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        # ===================CREATE DATASET================================= #
        #Create train/test dataloader
        dataset = Morphology_Dataset(self.morph_data, train=True,classify=self.classify)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=int(WORKERS))
        dataset_test = Morphology_Dataset(self.morph_data , train=False,classify=self.classify)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size,
                                                  shuffle=False, num_workers=int(WORKERS))

        print('training set', dataset.__len__())
        print('testing set', dataset_test.__len__())
        # ========================================================== #

        # ===================CREATE optimizer================================= #
        lrate = 0.001 #learning rate
        optimizer = optim.Adam(self.network.parameters(), lr = lrate)
        # ========================================================== #

        # =============DEFINE stuff for logs ======================================== #
        #meters to record stats on learning
        train_loss = AverageValueMeter()
        val_loss = AverageValueMeter()
        with open(logname, 'a') as f: #open and append
                f.write(str(self.network) + '\n')

        # =============start of the learning loop ======================================== #
        for epoch in range(self.nepoch):
            #TRAIN MODE
            train_loss.reset()
            self.network.train()
            
            # learning rate schedule
            if epoch%100==0 and epoch > 0:
                optimizer = optim.Adam(self.network.parameters(), lr = lrate/2.0)

            for i, data in enumerate(dataloader, 0):
                optimizer.zero_grad()
                morph, val = data
                #print(morph.shape)
                morph = morph.cuda()
                if self.classify:
                    val = val.float().cuda()
                else:
                    val = val.float().unsqueeze(1).cuda()
                pred_val = self.network(morph) #forward pass
                if self.classify:
                    loss_fn = torch.nn.BCEWithLogitsLoss()
                else:
                    loss_fn = torch.nn.SmoothL1Loss()
                #penalty = false_neg_penalty(pred_val,val)
                loss_net = loss_fn(pred_val,val)
                loss_net.backward()
                train_loss.update(loss_net.item())
                optimizer.step() #gradient update
                print('[%d: %d/%d] train loss:  %f ' %(epoch, i, len(dataset)/self.batch_size, loss_net.item()))

            # #VALIDATION MODE
            # if not self.train or epoch == (self.nepoch-1) or epoch == 0:
            #     val_loss.reset()
            #     dataset_test.ValueMeter.reset()
            #     self.network.eval()
            #     with torch.no_grad():
            #         for i, data in enumerate(dataloader_test, 0):
            #             optimizer.zero_grad()
            #             morph, val = data
            #             morph = morph.cuda()
            #             val = val.float().cuda()
            #             pred_val = self.network(morph) #forward pass
            #             loss_fn = torch.nn.SmoothL1Loss()
            #             loss_net = loss_fn(pred_val,val)
            #             val_loss.update(loss_net.item())
            #             print('[%d: %d/%d] val loss:  %f ' %(epoch, i, len(dataset_test)/self.batch_size, loss_net.item()))

            #     #dump stats in log file
            #     log_table = {
            #       "train_loss" : train_loss.avg,
            #       "val_loss" : val_loss.avg,
            #       "epoch" : epoch,
            #       "lr" : lrate,
            #     }
            # else:
            #     #dump stats in log file
            #     log_table = {
            #       "train_loss" : train_loss.avg,
            #       "epoch" : epoch,
            #       "lr" : lrate,
            #     }

            # with open(logname, 'a') as f: #open and append
            #     f.write('json_stats: ' + json.dumps(log_table) + '\n')

            #save last network
            print('saving net...')
            torch.save(self.network.state_dict(), '%s/network.pth' % (dir_name))

    def predict_morph_values(self):
        self.network.eval()
        with torch.no_grad():
            for data in self.morph_data:
                key = rotate_to_ref(data["morph"])
                if key == data["morph"]:
                    #morph = torch.Tensor(key).unsqueeze(0).cuda()
                    morph = thruster_key_to_morph(key).unsqueeze(0).cuda()
                    pred_val = self.network(morph)
                    data["pred_val"] = pred_val.tolist()[0]
                    data["pred_history"].append(pred_val.tolist()[0])
                else:
                    data["pred_val"] = -np.inf
                    data["pred_history"].append(-np.inf)    

    def train_sklearn(self):
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        morphs = []
        labels = []
        for data in self.morph_data:
            if len(data["val_means"]) > 0:
                morphs.append(data["morph"]) 
                labels.append(np.argmax(val_to_class(data["val_means"][-1])))
        morphs = np.array(morphs)
        labels = np.array(labels)
        clf = RandomForestClassifier(random_state=0)
        clf.fit(morphs,labels)

        for data in self.morph_data:
            key = rotate_to_ref(data["morph"])
            if key == data["morph"]:
                morph = np.array(data["morph"]).reshape(1,-1)
                data["pred_val"] = float(clf.predict(morph)[0])

    def save_to_file(self,path=None):
        if path == None:
            path = self.morph_path
        morph_dict = json.dumps(self.morph_data)
        fp = open(path,"w")
        fp.write(morph_dict)
        fp.close()
        print('Saved morphology dict to file')

if __name__ == "__main__":
    #morph_path = "./old_morph_datas/morph_data_10_14_2.json"
    morph_path = "./conv_only/morph_data_conv_only_recent_100.json"
    Morphologies = Morphologies(train=False,nepoch=100,morph_path = morph_path,classify=True)
    now = datetime.datetime.now()
    save_path = now.isoformat().replace(":","_")
    Morphologies.top_dir = os.path.join('train_log', save_path)
    if not os.path.exists(Morphologies.top_dir):
        os.mkdir(Morphologies.top_dir)   
    with open(morph_path) as f:
        morph_dict = json.load(f)
    Morphologies.morph_data = morph_dict
    #Morphologies.train_sklearn()
    Morphologies.train_conv_net()
    Morphologies.predict_morph_values()
    Morphologies.save_to_file(path='./conv_only/old_redo_class.json')

    #key_str = Morphologies.choose_morphologies()
    #print(key_str)


    #print(Morphologies.morph_data[0])
    #f = open('morph_data.json','r')
    #b = json.load(f)
    #print(b[0])
    # morph = np.array([
    #     [[0,0,0],
    #     [255,0,255],
    #     [0,0,0]],

    #     [[0,255,0],
    #     [0,0,0],
    #     [0,255,0]]
    #     ])
    # key = morphologies.morph_to_thruster_key(morph)
    # print(key)
    # print(morphologies.thruster_key_to_morph(key))
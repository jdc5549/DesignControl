import torch
import torch.nn as nn
import numpy as np

def morph_to_thruster_key(morph):
    # convert = [[7,0,1],[6,None,2],[5,4,3]]
    # key = np.zeros(8)
    # for i in range(2):
    #     for j in range(3):
    #         for k in range(3):
    #             if morph[i][j][k] != 0:
    #                 if i == 0:
    #                     key[convert[j][k]] = -1
    #                 else:
    #                     key[convert[j][k]] = 1
    convert = [[7,0,1],[6,None,2],[5,4,3]]
    key = np.zeros(8)
    for j in range(3):
        for k in range(3):
        	if convert[j][k] != None:
        		key[convert[j][k]] = morph[j][k]/255
    return key.tolist()

def thruster_key_to_morph(key):
    # convert = [[0,1],[0,2],[1,2],[2,2],[2,1],[2,0],[1,0],[0,0]]
    # morph = torch.zeros([2,3,3])
    # for i in range(len(key)):
    #     if key[i] == 1:
    #         place = convert[i]
    #         morph[1,place[0],place[1]] = 255
    #     if key[i] == -1:
    #         place = convert[i]
    #         morph[0,place[0],place[1]] = 255
    convert = [[0,1],[0,2],[1,2],[2,2],[2,1],[2,0],[1,0],[0,0]]
    morph = torch.zeros([3,3])
    for i in range(len(key)):
        place = convert[i]
        morph[place[0],place[1]] = 255*key[i]
    return morph 

def thruster_keys_to_string(keys):
    thrusters = ''
    for i in range(len(keys)):
        thrusters += '['
        for j in range(7):
            thrusters += str(keys[i][j]) + ' '
        thrusters += str(keys[i][7]) + ']'
        if(i < len(keys)-1):
            thrusters += ','   
    return thrusters

def get_morph_rotations(mask):
    rot_list = [mask]
    for j in range(7):
        for i in range(2):
            old_mask = mask.clone()
            mask[i,0,0] = old_mask[i,1,0]
            mask[i,0,1] = old_mask[i,0,0]
            mask[i,0,2] = old_mask[i,0,1]
            mask[i,1,2] = old_mask[i,0,2]
            mask[i,2,2] = old_mask[i,1,2]
            mask[i,2,1] = old_mask[i,2,2]
            mask[i,2,0] = old_mask[i,2,1]
            mask[i,1,0] = old_mask[i,2,0]
        rot_list.append(mask.clone())
    rot_list = torch.stack(rot_list)
    return rot_list 

def rotate_to_ref(key):
	rot_list = get_morph_key_rotations(key)
	for rot in rot_list:
		if rot[0] == 1:
			return rot
	for rot in rot_list:
		if rot[0] == -1:
			return rot
	return key

def get_morph_key_rotations(key):
	rot_list = [key]
	key_copy = key.copy()
	for i in range(7):
		key_copy.append(key_copy.pop(0))
		rot_list.append(key_copy)
		key_copy = key_copy.copy()
	return rot_list

#initialize the weighs of the network for Convolutional layers and batchnorm layers
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def val_to_class(val):
    out = None
    # if val <= 0.2:
    #     out = torch.tensor([1,0])
    # else:
    #     out = torch.tensor([0,1])
    #     #out = 0
    if val < 0:
        out = torch.tensor([1,0,0,0,0])
    elif val > 0 and val <= 0.2:
        out = torch.tensor([0,1,0,0,0])
        #out = 1
    if val > 0.2 and val <= 0.4:
        out = torch.tensor([0,0,1,0,0])
        #out = 2
    if val > 0.4 and val <= 0.6:
        out = torch.tensor([0,0,0,1,0])
        #out = 3
    if val > 0.6:
        out = torch.tensor([0,0,0,0,1])
        #out = 4
    return out
    # if val > 0.8 && val <= 1:
    #     out = 5

class AverageValueMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
	    self.reset()

	def reset(self):
	    self.val = 0
	    self.avg = 0
	    self.sum = 0
	    self.count = 0.0

	def update(self, val, n=1):
	    self.val = val
	    self.sum += val * n
	    self.count += n
	    self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, phase):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if (epoch%phase==(phase-1)):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/10.

def false_neg_penalty(pred,label):
    # pred = torch.argmax(pred)
    # label = torch.argmax(label)
    penalty = 0
    lam = 100
    for i in range(len(pred)):
        if torch.argmax(pred[i]) == 0 and torch.argmax(label[i]) == 1:
            penalty += lam
    return penalty
        

if __name__ == '__main__':
	key = [1,0,-1,0,1,0,-1,0]
	print(key)
	morph = thruster_key_to_morph(key)
	print(morph)
	new_key = morph_to_thruster_key(morph)
	print(new_key)

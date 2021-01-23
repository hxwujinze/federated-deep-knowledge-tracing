import sys
sys.path.append('../')
from model.RNNModel import RNNModel
from data.dataloader import getDataLoader
from Constant import Constants as C
import torch.optim as optim
from evaluation import eval
import torch
import math
import copy
trainLoader, testLoade,dis,split = getDataLoader()
rnn = RNNModel(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT).cuda(3)
loss_func = eval.lossFunc()

w_locals = [copy.deepcopy(rnn.state_dict())]*C.USERS

for epoch in range(0,C.EPOCH):
    print('epoch: ' + str(epoch))
    rnn, _, w_locals = eval.train_epoch(rnn, trainLoader, loss_func,dis,epoch,w_locals)
    auc = eval.test_epoch(rnn, testLoade, loss_func,split)


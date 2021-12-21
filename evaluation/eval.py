import sys
sys.path.append('../')
import tqdm
import torch
import torch.nn as nn
from sklearn import metrics
from Constant import Constants as C
import copy
import numpy as np
import random
import torch.optim as optim
import math
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
name = ['rnn']
def FedAvg(w,weights):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        if 1:
            w_avg[k] = w_avg[k].fill_(0)
    for k in w_avg.keys():
        if 1:
            for i in range(0, len(w)):
                w_avg[k] += w[i][k]*weights[i]/np.sum(weights)

    return w_avg

def Apply(g_model,local):
    
    w = g_model
    l_w = local
    for k in w.keys():
       if 1:
           A = g_model[k].cpu().view(1,-1).detach().numpy()
           B = local[k].cpu().view(1,-1).detach().numpy()
           q = 1 - cdist(A,B,metric='cosine')[0][0]
           l_w[k] =  q*l_w[k]+w[k]*(1-q)
    return l_w

def performance(ground_truth, prediction,reference):

    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().numpy(), prediction.detach().numpy())
    auc = metrics.auc(fpr, tpr)

    f1 = metrics.f1_score(ground_truth.detach().numpy(), torch.round(prediction).detach().numpy())
    recall = metrics.recall_score(ground_truth.detach().numpy(), torch.round(prediction).detach().numpy())
    precision = metrics.precision_score(ground_truth.detach().numpy(), torch.round(prediction).detach().numpy())
    mse = torch.sqrt(torch.sum((ground_truth - prediction)**2)/len(prediction)).detach().numpy()
    r = 0
    p = 0
    print(str(t)+'auc: ' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) + ' precision: ' + str(precision)+' mse: '+str(mse)+' r: '+str(r)+' p: '+str(p))

    return auc

class lossFunc(nn.Module):
    def __init__(self):
        super(lossFunc, self).__init__()

    def forward(self, pred, batch):
        loss = torch.sum(torch.zeros([1],requires_grad=True))
        for student in range(pred.shape[0]):
            delta = batch[student][:,0:C.NUM_OF_QUESTIONS] + batch[student][:,C.NUM_OF_QUESTIONS:]
            temp = pred[student][:C.MAX_STEP - 1].mm(delta[1:].t())
            index = torch.LongTensor([[i for i in range(C.MAX_STEP - 1)]])
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:C.NUM_OF_QUESTIONS] - batch[student][:, C.NUM_OF_QUESTIONS:]).sum(1) + 1)/2)[1:]
            for i in range(len(p)):
                if p[i] > 0:
                    loss = loss - (a[i]*torch.log(p[i]) + (1-a[i])*torch.log(1-p[i]))
     
        return loss

def train_epoch(model, Loader, loss_func,dis,epoch, m_locals,aucs):
    w_locals = []
    Init_w = copy.deepcopy(model.state_dict())
    
    E = 5
    newdis = []#calculate by your self

    mdis = np.array(newdis)
    mdis = mdis / np.sum(newdis)
    for trainLoader in Loader:
        batch_num = 0
        #model.load_state_dict(Apply(Init_w,m_locals[Loader.index(trainLoader)]))
        optimizer = optim.Adam(model.parameters(), lr=C.LR, amsgrad=False)
        
        for e in range(E):
            for batch in tqdm.tqdm(trainLoader, desc='\r\nTraining:    ', mininterval=2):
                batch_num += 1
                pred = model(batch.cuda(3)) 
                loss = loss_func(pred.cpu(), batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        w_locals.append(copy.deepcopy(model.state_dict()))
        model.load_state_dict(Init_w)
        
    w_glob = FedAvg(w_locals,newdis)
    model.load_state_dict(w_glob)
    return model, optimizer, w_locals

def test_epoch(model, testLoader, loss_func,split):

        ground_truth = torch.Tensor([])
        prediction = torch.Tensor([])
        for batch in tqdm.tqdm(testLoader, desc='\r\nTesting:    ', mininterval=2):
            pred = model(batch.cuda(3)).cpu()
            for student in range(pred.shape[0]):
                delta = batch[student][:,0:C.NUM_OF_QUESTIONS] + batch[student][:,C.NUM_OF_QUESTIONS:]
                temp = pred[student][:C.MAX_STEP - 1].mm(delta[1:].t())
                index = torch.LongTensor([[i for i in range(C.MAX_STEP - 1)]])
                p = temp.gather(0, index)[0]
                a = (((batch[student][:, 0:C.NUM_OF_QUESTIONS] - batch[student][:, C.NUM_OF_QUESTIONS:]).sum(1) + 1)/2)[1:]
                for i in range(len(p)):
                    if p[i] > 0:
                        prediction = torch.cat([prediction,p[i:i+1]])
                        ground_truth = torch.cat([ground_truth, a[i:i+1]])
         
        ref = performance(ground_truth, prediction,0)
    return ref





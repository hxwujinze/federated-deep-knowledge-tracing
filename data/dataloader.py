import sys
sys.path.append('../')
import torch
import torch.utils.data as Data
from Constant import Constants as C
from data.readdata import DataReader
import numpy as np
def getDataLoader():
   
    train,test,dis = DataReader('../dataset/student_log.json',C.MAX_STEP, C.NUM_OF_QUESTIONS,C.RATE).getData()
    trainLoader = []
    testLoader = []

    for item in train:
        dtrain = torch.FloatTensor(np.array(item).astype(float).tolist())
        trainLoader.append(Data.DataLoader(dtrain, batch_size= C.BATCH_SIZE, shuffle=True))
    dtest = torch.FloatTensor(test.tolist())
    testLoader = Data.DataLoader(dtest, batch_size= C.BATCH_SIZE, shuffle=False)
    

    return trainLoader, testLoader, dis


#trainLoader, testLoader = getDataLoader()

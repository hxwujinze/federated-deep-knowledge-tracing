import numpy as np
import itertools
import json
import os
import math
class DataReader():
    def __init__(self, recorde_path,maxstep, numofques,rate):
        self.path =recorde_path
        self.maxstep = maxstep
        self.numofques = numofques
        self.rate =rate
    def getData(self):
        print('loading train data...')
        AllData = [[],[]]
        splitD = []
        DIS = []
        know = [0]*self.numofques
        questoknow = {}            
        
        with open(self.path, 'r') as data:

            zero = [0 for i in range(self.numofques * 2)]
            ques = []
            ans = []
            id = 23098
            mlen = 0
            Data = []
            pp = 0
            school = 73
            schools = [73]
            if 1:
                for line in data:
                    item = json.loads(line)
                    if school != item['school_id']:
                        schools.append(school)

                        school = item['school_id']
                        if id != item['student_id']:
                            id = item['student_id']
                            mlen = len(ques)
                            pp +=1

                            slices = mlen//self.maxstep + (1 if mlen%self.maxstep > 0 else 0)
                            for i in range(slices):
                                temp = temp = np.zeros(shape=[self.maxstep, 2 * self.numofques])
                                if mlen > 0:
                                    if mlen >= self.maxstep:
                                        l = self.maxstep
                                    else:
                                        l = mlen
                                    for j in range(l):
                                        if ans[i*self.maxstep + j] >=0.5:
                                            temp[j][ques[i*self.maxstep + j]] = 1
                                        else:
                                            temp[j][ques[i*self.maxstep + j] + self.numofques] = 1
                                    mlen = mlen - self.maxstep
                                Data.append(temp.tolist())

                            ques = []
                            ans = []

                        TEST = Data[:int(len(Data)*self.rate)]
                        TRAIN = Data[int(len(Data)*self.rate):]

                        AllData[0].append(TRAIN)
                        AllData[1]+=TEST                 
                        DIS.append(math.ceil(len(Data)/64))

                        zero = [0 for i in range(self.numofques * 2)]
                        ques = []
                        ans = []
                        id = 0
                        mlen = 0
                        Data = []
                        pp = 0

                    if id != item['student_id']:
                        id = item['student_id']
                        mlen = len(ques)
                        pp +=1

                        slices = mlen//self.maxstep + (1 if mlen%self.maxstep > 0 else 0)
                        for i in range(slices):
                            temp = temp = np.zeros(shape=[self.maxstep, 2 * self.numofques])
                            if mlen > 0:
                                if mlen >= self.maxstep:
                                    l = self.maxstep
                                else:
                                    l = mlen
                                for j in range(l):
                                    if ans[i*self.maxstep + j] >=0.5:
                                        temp[j][ques[i*self.maxstep + j]] = 1
                                    else:
                                        temp[j][ques[i*self.maxstep + j] + self.numofques] = 1
                                mlen = mlen - self.maxstep
                            Data.append(temp.tolist())

                        ques = []
                        ans = []

                    ques.append(int(item['know']))
                    ans.append(int(item['grade']))
                        
                

                mlen = len(ques)
                pp +=1
                recode = {}
                if 1:
                    slices = mlen//self.maxstep + (1 if mlen%self.maxstep > 0 else 0)
                    for i in range(slices):
                        temp = temp = np.zeros(shape=[self.maxstep, 2 * self.numofques])
                        if mlen > 0:
                            if mlen >= self.maxstep:
                                l = self.maxstep
                            else:
                                l = mlen
                            for j in range(l):
                                if ans[i*self.maxstep + j] >=0.5:
                                    temp[j][ques[i*self.maxstep + j]] = 1
                                else:
                                    temp[j][ques[i*self.maxstep + j] + self.numofques] = 1
                            mlen = mlen - self.maxstep
                        Data.append(temp.tolist())

                    TEST = Data[:int(len(Data)*self.rate)]
 
                    TRAIN = Data[int(len(Data)*self.rate):]
                    AllData[0].append(TRAIN)
                    AllData[1]+=TEST

                    DIS.append(math.ceil(len(Data)/64))

        print("Train:"+str(np.array(AllData[0]).shape)+"\tTest:"+str(np.array(AllData[1]).shape))
        return AllData[0], np.array(AllData[1]),DIS






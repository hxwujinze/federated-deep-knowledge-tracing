# -*- coding: utf-8 -*-
import sys
import numpy as np
import random
from math import exp, log
from sklearn.metrics import roc_auc_score
import os
import json

def random_split(percentage,path):
    # split train data and test data
    data = []
    with open(path,'r') as file:
        for item in file:
            item = json.loads(item)
            data.append([item['student_number'],item['question_number'],int(item['score']) / int(item['total_score'])])
    train=list()
    test=list()
    student = []
    know = []
    distrik = []
    testindex = np.random.choice(range(len(data)),int(len(data)*percentage))
    
    for index in range(0,len(data)):
        if data[index][0] not in student:
            student.append(data[index][0])
        data[index][0] = student.index(data[index][0])

        if data[index][1] not in know:
            know.append(data[index][1])
            distrik.append(0)
        distrik[know.index(data[index][1])] += 1
        data[index][1] = know.index(data[index][1])
        if index in testindex:
            test.append(data[index])
        else:
            train.append(data[index])

    train=np.array(train)
    test=np.array(test)

    
    return train,test,len(student),len(know),np.array(distrik)/len(data)

def train_model(M,N,alpha,percentage,path,thres):
    train,test,M,N,distrik = random_split(percentage,path)
    A= np.random.rand(N)
    B= np.random.rand(N)
    D=1.7
    Theta= np.random.rand(M)
    likelihood_value_old= 0
    steps = 1000
    maxbatch=10
    threshold = 0.5
    maxA = A.copy()
    maxB = B.copy()
    maxAUC = 0.6
    mstr = ""
    for step in np.arange(steps):
        user=[int(t[0]) for t in train]
        item=[int(t[1]) for t in train]
        rating=np.array([float(t[2]) for t in train])
        predict = []
        for i in np.arange(len(user)):
            try:
                predict.append(  (1.0/(1.0+exp(-D*A[item[i]]*(Theta[user[i]]-B[item[i]]))))  )
            except:
                predict.append(0.001)
        predict = np.array(predict)

        likelihood_value =sum([(rating[i]* log(predict[i])+(1.0-rating[i])*log(1.001-predict[i])) for i in range(len(predict))])
        print("step= ",step,"likelihood_value = ",likelihood_value)
        for batch in np.arange(maxbatch):
            batch_index=np.random.choice(np.arange(0, len(train)), int(np.around((1.0/maxbatch) * len(train))),replace=False)
            a_gradient = np.zeros(N)
            b_gradient = np.zeros(N)
            theta_gradient = np.zeros(M)
            # A 、B and Theta gradient update
            for i in batch_index:
                user_index=int(train[i][0])
                item_index=int(train[i][1])
                rating=float(train[i][2])
                try:
                    temp=rating-1.0/(1.0+exp(-D*A[item_index]*(Theta[user_index]-B[item_index])))
                except:
                    temp = rating
                a_gradient[item_index]+=temp*D
                b_gradient[item_index]+=-temp*D*A[item_index]
                theta_gradient[user_index]=temp*D*A[item_index]
            A+= alpha*a_gradient
            B+= alpha*b_gradient
            Theta+=alpha*theta_gradient
        if step>1:

            if 1:

                user = [int(t[0]) for t in test]
                item = [int(t[1]) for t in test]
                rating =np.array([float(t[2]) for t in test])
                predict = []
                for i in np.arange(len(user)):
                    predict.append(  (1.0/(1.0+exp(-D*A[item[i]]*(Theta[user[i]]-B[item[i]])))))
                test_set_error =predict - rating
                mae_test= np.sum(abs(test_set_error)) / (len(test_set_error))
                rmse_test= np.sum(pow(test_set_error, 2)) /(len(test_set_error))
                rmse_test= np.sqrt(rmse_test)
                bi_rating = np.array([ 1 if score > threshold else 0 for score in rating])
                bi_predict =  np.array([ 1 if score > threshold else 0 for score in predict])

                acc_test = 1 - np.sum(abs(bi_rating-bi_predict))/(len(test_set_error))
                auc_test =roc_auc_score(bi_rating,predict)
                # ------------diversity test----------

                if auc_test > maxAUC:
                    maxAUC = auc_test
                    maxA = A.copy()
                    maxB = B.copy()
                    mstr ="test_acc:"+str(acc_test)+ "\t test_auc:"+str(auc_test)+"\t test_mae:"+str(mae_test)+ "\t test_rmse:"+str(rmse_test)
                
            else:
                likelihood_value_old = likelihood_value
        else:
            likelihood_value_old = likelihood_value
    if maxAUC >thres:
        print('True')
        print(mstr)
        return maxA,maxB,True,distrik
    else:
        print('False')
        return [],[],False,distrik
print("***************")



rootdir = ''

filelist = os.listdir(rootdir) 
statics = np.loadtxt('staticsifly.txt',dtype=int,delimiter=' ')
weight = []
C = False 
for fileindex in range(0,len(filelist)):
    C = False
    thres = 0.6
    print(rootdir+str(fileindex)+'.data')
    M = statics[fileindex][0]
    N = statics[fileindex][3]
    while not C:
        A,B,C,distrik =train_model(M,N,0.001,0.1,rootdir+str(fileindex)+'.data',thres)
        thres -= 0.005
    ALLIS = []
    for i in range(0,N):
        a = A[i]
        b = B[i]
        thetaS = list(np.arange(-5.0, 5.0, 0.01))
        IS = []
        for theta in thetaS:
            IS.append(1.7**2*a**2/( np.exp(1.7*a*(theta-b))* ((1 + np.exp(-1.7*a*(theta-b)))**2) ))
        ALLIS.append(np.array(IS)*distrik[i])
    info = np.array(ALLIS)
    infot = np.sum(info,axis=0)*N
    print(max(infot))
    weight.append(max(infot))
print(weight)











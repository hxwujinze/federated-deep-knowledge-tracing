import numpy as np
import os
import json
rootdir = './question/'

filelist = os.listdir(rootdir) 



def CTT(count,grade,distrik):
    Pvec = []
    Dvec = []
    R = 0
    k = len(count[0])
    varlist = []
    dk = []

    for knowindex in range(len(count[0])):
        gradevec = grade.T[knowindex].tolist()
        varlist.append(np.var(gradevec))
        gradevec = sorted(filter(lambda n: n != -1, gradevec))

        sec = int(len(gradevec)*0.27)

        if sec != 0 :
            Llist = gradevec[:sec]
            Hlist = gradevec[-sec:]
        
            PL = np.sum(Llist)/len(Llist)
            PH = np.sum(Hlist)/len(Hlist)
            Pvec.append((PH+PL)/2)
            Dvec.append(PH-PL)
            dk.append(distrik[knowindex])

    dk = np.array(dk)

    R = (k/(k-1))*(1-np.sum(varlist*distrik)*len(distrik)/np.var(np.sum(grade,axis=1).tolist()))
    return  np.sum(-np.log(np.abs(np.array(Pvec)-0.5))*dk),np.sum(np.array(Dvec)*dk),R
statics = np.loadtxt('statics.txt',dtype=int,delimiter=' ')
weight = []
for fileindex in range(0,len(filelist)):
    data = np.loadtxt(rootdir+str(fileindex)+'.data')
    
    knowlist = []
    studentlist = []
    recode = []
    count = np.zeros((statics[fileindex][0], statics[fileindex][3]))
    grade = np.zeros((statics[fileindex][0], statics[fileindex][3]))
    distrik = []
    allcount = 0
    data = np.loadtxt(rootdir+str(fileindex)+'.data')
    if 1:
        for item in data:
            allcount +=1
            student = item[0]
            know = item[1]
            gradei = item[2]
            if know not in knowlist:
                
                knowlist.append(know)
                distrik.append(0)
            distrik[knowlist.index(know)]+=1
            if student not in studentlist:
                studentlist.append(student)
            count[studentlist.index(student)][knowlist.index(know)] += 1
            grade[studentlist.index(student)][knowlist.index(know)] += gradei
    for i in range(0,len(studentlist)):
        for j in range(0,len(knowlist)):
            if count[i][j] == 0:
                grade[i][j] = -1
            else:
                grade[i][j] /= count[i][j]
    
    Pvec,Dvec,R = CTT(count,grade,np.array(distrik)/allcount)

    weight.append(Pvec*Dvec*R)

weight = np.array(weight)

print(weight.tolist())
    
        


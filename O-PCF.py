
#from __future__ import print_function
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
import random
import sys
import cplex
from cplex.exceptions import CplexError
import numpy as np
from gurobipy import *
from cvxopt import matrix, solvers
import math
import csv
import arff
import time
import copy
style.use('ggplot')
from sklearn import metrics




def traTestBol(a,indxForTenFold,wFold,TotalFold):

    dTest=[]
    dTrain=[]

    IOindx = np.ones(len(a))

    for i in range(0,len(indxForTenFold)):
        for j in range(0,len(indxForTenFold[i])):
            if j%TotalFold==wFold-1:
                dTest.append(a[indxForTenFold[i][j]])
                IOindx[indxForTenFold[i][j]]=0
            else:
                dTrain.append(a[indxForTenFold[i][j]])


    dTest=np.array(dTest)
    dTrain=np.array(dTrain)



    return dTrain, dTest,IOindx





def prepIndx(a):

    sinifS = int(np.max(a[:,-1]))
    rowS=len(a)
    col=len(a[0])

    indxes=[]

    for i in range(0,sinifS):
        indxes.append([])

    for i in range(0,rowS):
        indxes[int(a[i,col-1]-1)].append(i)

    return indxes



def PCF(Ajr, cjr,status):

    # Create optimization model
    m = Model('PCF')

    # Create variables
    gamma = m.addVar(vtype=GRB.CONTINUOUS, name='gamma')
    w = range(nn)
    for a in range(nn):
        w[a] = m.addVar(vtype=GRB.CONTINUOUS,ub=1, name='w[%s]' % a)

    ksi = m.addVar(vtype=GRB.CONTINUOUS, lb=1,ub=10, name='ksi')

    m.update()
    hataA = {}


    for i in range(len(Ajr)):
        hataA[i] = m.addVar(vtype=GRB.CONTINUOUS,lb=0, name='hataA[%s]' % i)


        m.update()
        m.addConstr(quicksum((Ajr[i][j] - cjr[j]) * w[j] for j in range(len(cjr))) + (ksi * quicksum(math.fabs(Ajr[i][j] - cjr[j]) for j in range(len(cjr)))) - gamma <= hataA[i])




    m.update()





    m.setObjective((quicksum (quicksum((Ajr[i][j] - cjr[j]) * -w[j] for j in range(len(cjr))) - (ksi * quicksum(math.fabs(Ajr[i][j] - cjr[j]) for j in range(len(cjr)))) + gamma for i in range(len(Ajr)))) +2.0*(quicksum(hataA[k] for k in range(len(hataA)))), GRB.MINIMIZE)

    m.update()
    # Compute optimal solution
    m.optimize()
    m.write('model.sol')
    status.append(m.Status)
    ww=[]
    for i in range(len(cjr)):
        ww.append(w[i].X)

    return {'s':status,'w': ww, 'gamma': gamma.x, 'ksi': ksi.x, 'c':cjr}

def findgj(A, labels, centroids,status):

    gj=[]



    Aj=[]
    for i in range(len(centroids)):
        Aj.append([])

    for index in range(len(labels)):
        Aj[labels[index]].append(A[index])

    r=0
    for Ajr in Aj:
        sonuc=PCF(Ajr, centroids[r],status)
        status=sonuc['s']
        gj.append(sonuc)
        r=r+1

    return status,gj

def pcfDeger(w, ksi, gamma, c, x):
    deger = np.dot(w,x-c) + ksi*np.sum(abs(x-c)) -gamma
    return deger

def sinifBul(data):
    sinifTahmini=[]
    g_deger=[]
    for d in data:
        t=1
        enkDeger=float('inf')
        gj_deger=[]
        for gj in g:
            gjr_deger=[]
            for gjr in gj:
                fonkDeger = pcfDeger(gjr['w'],gjr['ksi'],gjr['gamma'],gjr['c'],d[0:-1])
                gjr_deger.append(fonkDeger)
                if (fonkDeger<enkDeger):
                    enkDeger=fonkDeger
                    sinifT=t
            t=t+1
            gj_deger.append(gjr_deger)
        g_deger.append(gj_deger)
        sinifTahmini.append(sinifT)
    return sinifTahmini

def ROCCurve(VAte, VBte):

    scores=[]
    scores.append(VAte)
    scores.append(VBte)

    labels=[]
    labels.append(np.ones(len(VAte)))
    labels.append(np.zeros(len(VBte)))

    labels=np.concatenate((labels[0],labels[1]),axis=0)
    scores=np.concatenate((scores[0],scores[1]),axis=0)

    fpr, tpr, thresholds = metrics.roc_curve(labels, -scores, pos_label=1)


    aucVal=metrics.auc(fpr, tpr)



    return  fpr, tpr, aucVal


def egitimOraniniHesapla(gj, sinifEtiket, dataTrain):
    dogruSayisiA=0.0
    dogruSayisiB=0.0
    say=0.0   # B sayisi
    for d in dataTrain:
        enkDeger=float('inf')
        for gjr in gj:
            fonkDeger = pcfDeger(gjr['w'],gjr['ksi'],gjr['gamma'],gjr['c'],d[0:-1])
            if (fonkDeger<enkDeger):
                enkDeger=fonkDeger
        if (enkDeger<0):
            if d[-1]==sinifEtiket:
                dogruSayisiA=dogruSayisiA+1
            else:
                say+=1
        else:
            if d[-1]!=sinifEtiket:
                dogruSayisiB=dogruSayisiB+1
                say+=1
    egitimOraniA=float(dogruSayisiA)/(len(dataTrain)-say)
    egitimOraniB=float(dogruSayisiB)/say
    egitimOrani=(float(dogruSayisiA)+float(dogruSayisiB))/len(dataTrain)
    return egitimOrani
def PCFValues(M, g):
    val=np.zeros(len(M))
    say=0
    for d in M:
        tV=999999999999999999999
        for gjr in g:
            fonkDeger = pcfDeger(gjr['w'],gjr['ksi'],gjr['gamma'],gjr['c'],d[0:-1])
            if (fonkDeger<tV):
                tV=copy.deepcopy(fonkDeger)
        val[say]=copy.deepcopy(tV)
        say=say+1

    return val

def arffOku(dosya):
    d = arff.load(open(dosya, 'rb'))
    #print d
    #print d['attributes']
    ozellikSayisi= len(d['attributes'])
    v=[]
    for dd in d['data']:
        satir=[]
        for ddd in dd:
            satir.append(float(ddd))
        v.append(satir)
    v=np.array(v)
    return v

def veriOku(dosya):
    dosya = open(dosya)
    okuyucu = csv.reader(dosya, quotechar=',')
    data = []
    # problem verileri okunuyor...
    for row in okuyucu:
        satirVeri = []
        for deger in row:
            satirVeri.append(float(deger))

        data.append(satirVeri)

    data=np.array(data)
    return data

start_time = time.time()
data = veriOku('.../w7aTraining.csv')
data2 = veriOku('...w7aTesting.csv')




#print data
egtDO=[]
testDO=[]

## PARAMETERS ##
cSayisi=7 #cluster size
tekrar=20 # repeat

######################

mm=len(data)  # row size
nn=len(data[0])-1 # feature size
sinifSayisi = int(np.max(data[:,-1])) # classes must be 1 to n in the last column ...................................


status = []
print "okundu"

totalAtest=0
totalAtrain=0
totalB=0
Allaucs=np.zeros(tekrar)

for hh in range(tekrar):


    #--------k-Means  Temelli PCF Algoritmasi ....-------------------------------------------------Alg1_start----


    g=[]



    AjTr = []
    AjTe = []
    Bj = []
    for d in data:
        if d[-1] == 1:
            AjTr.append(d)
        else:
            Bj.append(d)



    for d in data2:
        if d[-1] == 1:
            AjTe.append(d)
        else:
            Bj.append(d)



    AjTe=np.array(AjTe)
    AjTr=np.array(AjTr)
    Bj=np.array(Bj)




    kmeans = KMeans(init='k-means++', n_clusters=cSayisi, n_init=100)
    kmeans.fit(AjTr[:,:nn])


    centroids = kmeans.cluster_centers_
    clusters = kmeans.labels_

    status,gj=findgj(AjTr[:,:nn], clusters, centroids,status)

    g.append(gj)


    #-------------------------------------------------------------------------------------------------Alg1_end '''

    j=1
    for gj in g:
        r=1
        print j,". SINIF ICIN SINIFLANDIRICI"
        for gjr in gj:
            print j, "siniflandiricisinin", r,". cluster'i B'den ayiran fonksiyon parametreleri"
            print gjr['ksi']
            print gjr['gamma']
            print gjr['w']
            print gjr['c']
            print "\n"
            r=r+1
        j=j+1

    valuesBj=PCFValues(Bj,g[0])
    valuesAjTr=PCFValues(AjTr,g[0])
    valuesAjTe=PCFValues(AjTe,g[0])

    minVB=np.min(valuesBj)
    maxVB=np.max(valuesBj)

    minVA=np.min(valuesAjTe)
    maxVA=np.max(valuesAjTe)

    minV=min(minVA,minVB)
    maxV=max(maxVA,maxVB)

    fpr, tpr, auc =ROCCurve(valuesAjTe,valuesBj)

    print auc

    Allaucs[hh]=copy.deepcopy(auc)

print np.sum(Allaucs)/tekrar
print("----")
print(Allaucs)
print("--- %s seconds ---" % (time.time() - start_time))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('The ROC curve of Web-7a')
plt.plot(fpr,tpr, 'b--')
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))







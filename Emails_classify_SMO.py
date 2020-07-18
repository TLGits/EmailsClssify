#-*-coding:utf-8-*-

from numpy import *
import numpy as np

#定义存储变量的类
class opStruct():
    def __init__(self,dataMatIn,classLabels,C,toler,kTup,kTup1):
        self.X = dataMatIn                        #数据
        self.labelMat = classLabels               #标签
        self.C = C                                #容忍度
        self.toler = toler                        #误差的容忍度
        self.m = shape(dataMatIn)[0]              #数据的个数
        self.alphas = mat(zeros((self.m,1)))      #alpha 值，每个数据对应一个alpha
        self.b = 0                                # 常数项
        self.eCache = mat(zeros((self.m,2)))      #保存误差和下表
        self.k = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.k[:,i] = kernelTrans(self.X,self.X[i,:],kTup,kTup1) 

def getrightfile(filename1,filename2):
    """
    载入数据
    """
    a = np.load(filename1)
    b = np.load(filename2)
    dataMatIn = a.tolist()
    classLabels = b.tolist()
    return dataMatIn,classLabels

def kernelTrans(X,A,kTup,kTup1):
    """
    加入核函数
    """
    m,n = shape(X)
    k = mat(zeros((m,1)))
    if kTup == 'lin':
        k = X * A.T
    elif kTup == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A #每一行减去A，在自己乘
            k[j] = deltaRow * deltaRow.T
        k = exp(k/(-1 * kTup1 ** 2)) #就是利用的公式
    return k

def clipAlpha(ai,H,L):
    """
    保证alpha必须在范围内
    """
    if ai > H:
        ai = H
    elif ai < L :
        ai = L
    return ai

def selectJrand(i,oS):
    """
    随机选择第二个不同的alpha
    """
    j = i
    while i == j:
        j = int(np.random.uniform(0,oS.m))
    return j

def calcEk(oS,k):
    """
    计算误差
    """
    fXk = float((multiply(oS.alphas,oS.labelMat)).T*oS.k[:,k] + oS.b) #预测值
    Ek = fXk - oS.labelMat[k] #误差值
    return Ek

def selectJ(i,oS,Ei):
    """
    选择第二个alpha 并且相差最大的
    """
    maxK = -1
    maxDelaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcaheList = nonzero(oS.eCache[:,0].A)[0]
    if len(validEcaheList) > 0:
        for k in validEcaheList:
            if k == i: #取不同的 alpha
                continue
            Ek = calcEk(oS,k) #计算k的与测试与真实值之间的误差
            deltaE = abs(Ei - Ek) #找与Ei 距离最远的
            if maxDelaE < deltaE:
                maxDelaE = deltaE  
                maxK = k     #与Ei差别最大的K
                Ej = Ek      #K的误差
        return maxK,Ej
    else:
        j = selectJrand(i,oS)
        Ej = calcEk(oS,j) #计算预测值和真实值的误差
    return j,Ej

def updateEk(oS,k):
    """
    更新误差
    """
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]

def innerL(i,oS):
    """
    SMO 优化
    """
    Ei = calcEk(oS,i)
    #在误差允许的范围外，如果小于规定的误差，就不需要更新了
    if ((oS.labelMat[i] * Ei ) <= oS.toler and oS.alphas[i] <= oS.C) or\
            ((oS.labelMat[i] * Ei) <= oS.toler and oS.alphas[i] >= 0):
        j,Ej = selectJ(i,oS,Ei)  #选择另一个alphaj和预测值与真实值的差
        alphaIold = oS.alphas[i].copy() #复制alpha，因为后边会用到
        alphaJold = oS.alphas[j].copy()
 
        if (oS.labelMat[i] != oS.labelMat[j]): #两个类别不一样 一个正类 一个负类
            L = max(0,oS.labelMat[j] - oS.labelMat[i])  # 约束条件 博客里有
            H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C,oS.alphas[j] + oS.alphas[i])
 
        if L == H:
            print('L == H')
            return 0
            
        #利用核函数
        eta = 2.0 * oS.k[i,j] - oS.k[i,i] - oS.k[j,j]
        if eta > 0:
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta  #就是按最后的公式求解
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)  #在L，H范围内
        updateEk(oS,j)
 
        if (oS.alphas[j] - alphaJold) < 0.0001:
            return 0
 
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS,i)

        #利用核函数的
        b1 = oS.b - Ei - oS.labelMat[i] * oS.k[i,i] * (oS.alphas[i] - alphaIold) - oS.labelMat[j] * oS.k[i,j] * (oS.alphas[j] - alphaJold)
        b2 = oS.b - Ej - oS.labelMat[i] * oS.k[i,j] * (oS.alphas[i] - alphaIold) - oS.labelMat[j] * oS.k[j,j] * (oS.alphas[j] - alphaJold)
 
        #跟新b
        if oS.alphas[i] < oS.C and oS.alphas[i] > 0:
            oS.b = b1
        elif oS.alphas[j] < oS.C and oS.alphas[j] > 0:
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

def calcWs(alpha,dataArr,classLabels):
    """
    计算alpha 获得分类的权重向量
    :param alpha:
    :param dataArr: 训练数据
    :param classLabels: 训练标签
    :return:
    """
    X = mat(dataArr)
    labelMat = mat(classLabels).T #变成列向量
    m,n = shape(X)
    w  =zeros((n,1)) #w的个数与 数据的维数一样
    for i in range(m):
        w += multiply(alpha[i] * labelMat[i],X[i,:].T) #alpha[i] * labelMat[i]就是一个常数  X[i,:]每（行）个数据，因为w为列向量，所以需要转置
    return w

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup ='lin',kTup1=0):
    """
    核心主函数
    :param dataMatIn: 训练数据
    :param classLabels: 训练标签
    :param C: 常量
    :param toler:容错度
    :param maxIter: 最大迭代次数
    :param kTup: 核函数类型参数
    :param kTup1: 核函数类型参数
    :return:
    """
    oS = opStruct(mat(dataMatIn),mat(classLabels).T,C,toler,kTup,kTup1)
    iter = 0
    entireSet = True
    alphaPairedChanged = 0
    while (iter < maxIter) and ((alphaPairedChanged > 0) or (entireSet)):
        alphaPairedChanged = 0
        if entireSet:
            # 遍历所有的数据 进行更新
            for i in range(oS.m):
                alphaPairedChanged += innerL(i,oS)
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]
            for i in nonBoundIs:
                alphaPairedChanged += innerL(i,oS)
            iter += 1
 
        if entireSet:
            entireSet = False
        elif (alphaPairedChanged == 0):
            entireSet = True
    return oS.b,oS.alphas

if __name__ == '__main__':
    dataMatIn,classLabels = getrightfile('smo_mail_list.npy','smo_label_list.npy')
    b,alphas =smoP(dataMatIn,classLabels,C=0.6,toler=0.001,maxIter=40,kTup = 'lin',kTup1=1)
    print(b,alphas)

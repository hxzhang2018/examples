import os
import random
from clusterfile import ClusterFile;
from cluster import Cluster
import numpy as np


class DoClusterWork(object):
    def __init__(self,path,clusterNum):
        self.__path = path
        self.__vocabulary = []
        self.__clusterFile = []
        self.__cluster = []

        self.createClusterFile()
        self.generateVocabulary()
        self.calClusterFileTdm()
        self.createFirstCluster(clusterNum)

        self.doCluster()
        self.outputCluster()

    #产生聚簇文件对象
    def createClusterFile(self):
        fileList = os.listdir(self.__path)

        for fileName in fileList:
            print(fileName)
            #必须是文件,只有一层
            clusterFile = ClusterFile(self.__path,fileName)
            self.__clusterFile.append(clusterFile)


    #产生词汇表
    def generateVocabulary(self):

        #加载去停词
        stopWordsFile = open("hlt_stop_words.txt", encoding='utf-8')
        stopWords = stopWordsFile.read().splitlines()
        stopWords.append('\n')
        stopWords.append(' ')
        stopWordsFile.close()

        vocabulary = set()
        for clusterFile in self.__clusterFile:
            clusterFile.cutWords(stopWords);
            for word in clusterFile.getWords():
                vocabulary.add(word)
        self.__vocabulary = list(vocabulary)

    def calClusterFileTdm(self):

        #计算idf
        idf = [0] * len(self.__vocabulary)
        clusterFileNum = len(self.__clusterFile)
        for idx in range(clusterFileNum):
            for word in set(self.__clusterFile[idx].getWords()):
                idf[self.__vocabulary.index(word)] +=1

        #每个聚簇文件计算tdm
        for clusterFile in self.__clusterFile:
            clusterFile.calTdm(clusterFileNum,self.__vocabulary,idf)

    #第一次产生聚簇
    def createFirstCluster(self,clusterNum):
        fileClusterNum = len(self.__clusterFile)
        assert fileClusterNum >= clusterNum,"簇数一定要小于文件数 "
        if fileClusterNum >= clusterNum:
            #fileClusterList = random.sample(range(fileClusterNum),clusterNum)
            fileClusterList = [0,5,8,19]
            clusterIndex = 0;
            for index in fileClusterList:
                cluster = Cluster(clusterIndex)
                cluster.addClusterFile(self.__clusterFile[index])
                self.__cluster.append(cluster)
                clusterIndex += 1

    #计算距离cos
    def calVectorDis(self,A,B):
        aLen = len(A)
        bLen = len(B)
        assert aLen == bLen ,"长度必须相等"
        if aLen == bLen:
            sAB = 0
            sAA = 0
            sBB = 0
            for idx in range(aLen):
                sAB += A[idx] * B[idx]
                sAA += A[idx] * A[idx]
                sBB += B[idx] * B[idx]
            return float(sAB)/(np.sqrt(sAA) * np.sqrt(sBB)) ;

    #聚簇
    def doCluster(self):

        change = True;
        while change:
            #计算中心
            for cluster in self.__cluster:
                cluster.calcClusterCenter()
                cluster.removeAllClusterFiles();

            change = False;#假定本次族不再发生变化

            #将ClusterFile划分到簇中
            for clusterFile in self.__clusterFile:
                clusterLen = len(self.__cluster)
                dis = [0] * clusterLen
                for idx in range(clusterLen):
                    dis[idx] = self.calVectorDis(clusterFile.getTdm(),self.__cluster[idx].getClusterCenter())

                #找出dis中值最大的索引，也就是clusterfile 应该划分到目标簇
                clusterIdx = dis.index(max(dis))

                fileClusterIndexOld = clusterFile.getClusterIndex();

                if clusterIdx != fileClusterIndexOld: #不在以前的簇中,要继续
                    change = True

                self.__cluster[clusterIdx].addClusterFile(clusterFile)

    def outputCluster(self):
        for cluster in self.__cluster:
            cluster.outputClusterFile();


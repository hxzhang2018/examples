import jieba
import numpy as np

#聚类文本文件
class ClusterFile(object):
    def __init__(self,path,fileName):
        self.__path = path
        self.__fileName = fileName
        self.__words = [] #单词
        self.__tdm = []
        self.__clusterIndex = -1;#所属簇索引

    def getPath(self):
        return self.__path

    def getFileName(self):
        return self.__fileName

    def getWords(self):
        return self.__words

    def cutWords(self,stopWords):
        file = open(self.__path + self.__fileName,encoding="utf-8")
        contents = file.read()
        words = jieba.cut(contents)
        for word in words:
            if word not in stopWords:
                self.__words.append(word)
        file.close()

    def calTdm(self,docLen,vocabulary,idf):
        vocabularyLen = len(idf)
        wordLen = len(self.__words)
        wordFreq = [0] * vocabularyLen
        for word in self.__words:
            wordFreq[vocabulary.index(word)] += 1

        self.__tdm = [0] * vocabularyLen

        for idx in range(vocabularyLen):
            self.__tdm[idx] = float(wordFreq[idx])/wordLen * np.log(float(docLen)/(idf[idx]+1))

    def getClusterIndex(self):
        return self.__clusterIndex

    def setClusterIndex(self,clusterIndex):
        self.__clusterIndex = clusterIndex

    def getTdm(self):
        return self.__tdm





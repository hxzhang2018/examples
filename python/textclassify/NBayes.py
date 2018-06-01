import numpy as np

class NBayes(object):
    #vocabulary 二维列表#分词后的文档
    #lables 对应每个文本的分类，是个外部导入的列表,和文本长度一样
    def __init__(self,documents,labels):
        assert (len(documents) == len(labels)),"文档数必须和列表数一致"
        self.documents = documents;
        self.vocabulary = []
        self.lables = labels
        self.idf = 0
        self.tf = 0
        self.tdm = 0
        self.Pcates = {} #类别及概率

        #生成字典 vocabulary
        self.genVocabulary()

        # 计算每个分类在数据集中的概率：P(yi) Pcates
        self.calProp()

        #生成词频空间向量
        optimizal = True
        if(optimizal):
            self.calTfIdf() #计算Tf-Idf
        else:
            self.calWordFreq()#计算普通词频

        self.calTdm()

    def getDocmenstCount(self):
        return len(self.lables)

    def trainSet(self):
        pass;

    #生成词典
    def genVocabulary(self):
        voca = set()
        [voca.add(word) for doc in self.documents for word in doc]
        self.vocabulary = list(voca)

    def getVocabulary(self):
        return self.vocabulary


    # 计算每个分类在数据集中的概率：P(yi)
    def calProp(self):
        lableTmps = set(self.lables)
        for label in lableTmps:
            self.Pcates[label] = float(self.lables.count(label))/(len(self.lables))

    def getProp(self):
        return self.Pcates;

    #生成普通词频向量
    def calWordFreq(self):
        docCount = self.getDocmenstCount();
        vocaCount = len(self.vocabulary)

        self.tf = np.zeros([docCount,vocaCount])
        self.idf = np.zeros([1,vocaCount])
        for idx in range(docCount):
            for word in self.documents[idx]:
                self.tf[idx,self.vocabulary.index(word)] +=1

            # 计算每个单词在文档中出现的次数
            for word in set(self.documents[idx]):
                self.idf[0,self.vocabulary.index(word)] +=1

    def calTfIdf(self):
        docCount = self.getDocmenstCount()
        vocaCount = len(self.vocabulary)
        self.tf = np.zeros([docCount,vocaCount])
        self.idf = np.zeros([1,vocaCount])

        for idx in range(docCount):
            for word in self.documents[idx]:
                self.tf[idx,self.vocabulary.index(word)] += 1

            # 消除不同句长导致的偏差
            self.tf[idx] = (self.tf[idx])/len(self.documents[idx])

            for word in set(self.documents[idx]):
                self.idf[0,self.vocabulary.index(word)] +=1

        self.idf = np.log(float(docCount) / (self.idf+1))
        self.tf = np.multiply(self.tf, self.idf)  # 矩阵与向量的点乘

        print("tf-idf successed!")

    #计算tdm值 # P(x|yi)
    def calTdm(self):
        vocaCount = len(self.vocabulary)
        cateCount = len(self.Pcates)
        self.tdm = np.zeros([cateCount,vocaCount])
        sumList = np.zeros([cateCount,1])
        docCount = self.getDocmenstCount()
        for idx in range(docCount):
            self.tdm[self.lables[idx]] += self.tf[idx]
            sumList[self.lables[idx]] = np.sum(self.tdm[self.lables[idx]])

        self.tdm = self.tdm / sumList #归一化？？？


    #def 测试集映射到词典中
    def map2Vacab(self,testData):
        testDataFreq = np.zeros([1,len(self.vocabulary)])
        for word in testData:
            if word in self.vocabulary:
                testDataFreq[0,self.vocabulary.index(word)] +=1
        return  testDataFreq

    # 输出分类类别
    def predict(self, testData):
        testDataFreq = self.map2Vacab(testData)
        if np.shape(testDataFreq)[1] != len(self.vocabulary):
            print
            "输入错误"
            exit(0)
        predvalue = 0
        predclass = ""
        for tdm_vect, keyclass in zip(self.tdm, self.Pcates):
            # P(x|yi)P(yi)
            temp = np.sum(testDataFreq * tdm_vect * self.Pcates[keyclass])
            if temp > predvalue:
                predvalue = temp
                predclass = keyclass
        return predclass


    #这种算法可能会导致无法判断出属于哪一类的问题，因为可能会产生predvalue的值在哪一类中都为0的情况
    def predictOther(self, testData):
        testDataFreq = self.map2Vacab(testData)
        if np.shape(testDataFreq)[1] != len(self.vocabulary):
            print
            "输入错误"
            exit(0)
        predvalue = 0
        predclass = ""
        for tdm_vect, keyclass in zip(self.tdm, self.Pcates):
            # P(x|yi)P(yi)
            temp = 1
            for idx in range(len(testDataFreq[0])):
                if testDataFreq[0,idx] >0 :
                    temp *= tdm_vect[idx]
                else:
                    temp *= (1-tdm_vect[idx])

            temp *= self.Pcates[keyclass]

            if temp > predvalue:
                temp = predvalue
                predclass = keyclass
        return predclass
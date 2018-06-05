
class Cluster(object):
    def __init__(self,clusterIndex):
        self.__clusterCenter=[] #聚簇中心
        self.__clusterFiles = [] #簇中所包含的文件
        self.__clusterIndex = clusterIndex

    def getClusterCenter(self):
        return self.__clusterCenter

    def addClusterFile(self,clusterFile):
        clusterFile.setClusterIndex(self.__clusterIndex)
        self.__clusterFiles.append(clusterFile)

    def getClusterFiles(self):
        return self.__clusterFiles;

    def removeAllClusterFiles(self):
        self.__clusterFiles.clear()


    #计算簇中心
    def calcClusterCenter(self):
        self.__clusterCenter.clear();
        fileCount = len(self.__clusterFiles)
        if fileCount > 0 :
            tdmCount = len(self.__clusterFiles[0].getTdm())
            assert tdmCount > 0,"tdm向量维数必须大于0"
            self.__clusterCenter = [0]  * tdmCount
            for fileIndex in range(fileCount):
                tdmCountTmp = len(self.__clusterFiles[fileIndex].getTdm())
                assert tdmCount == tdmCountTmp,"维度必须相等"
                for idx in range(tdmCount):
                    self.__clusterCenter[idx] += self.__clusterFiles[fileIndex].getTdm()[idx]

            #求平均值
            for idx in range(tdmCount):
                self.__clusterCenter[idx] /= fileCount


    def outputClusterFile(self):
        print("%d类文件有： " % (self.__clusterIndex+1))
        for clusterFile in self.__clusterFiles:
            print(clusterFile.getFileName())


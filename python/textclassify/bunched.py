import os
import jieba
from sklearn.datasets.base import Bunch
import pickle
from file import  File
import pickle

class Bunched(object):
    def __init__(self,segPath,wordBagPath):
        self.__segPath = segPath
        self.__wordBagPath = wordBagPath

    def bunched(self):

        file  = File()
        bunch = Bunch(targetName=[],label=[],fileName=[],contents=[])
        cateList = os.listdir(self.__segPath)
        bunch.targetName.extend(cateList)

        for myDir in cateList:
            dirPath = self.__segPath + myDir + "/"
            fileList = os.listdir(dirPath)

            for fileName in fileList:
                fileFullName = dirPath + fileName
                bunch.label.append(myDir)
                bunch.fileName.append(fileFullName)
                bunch.contents.append(file.readFile(fileFullName).strip())

        #序列化到文件中
        dumpFile = open(self.__wordBagPath,"wb")
        pickle.dump(bunch,dumpFile)

        dumpFile.close()

        print("构建文本对象结束")


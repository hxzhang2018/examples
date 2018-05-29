from file import  File
import os
import jieba

class CutWords(object):
    def __init__(self,cutWordSrcDir,cutWordSaveDir):
        self.__cutWordSrcDir = cutWordSrcDir
        self.__cutWordSaveDir = cutWordSaveDir

    #分词操作
    def cut(self):
        srcDirList = os.listdir(self.__cutWordSrcDir)
        file = File()
        for myDir in srcDirList:
            #创建保存目录
            saveDir = self.__cutWordSaveDir +  myDir + "/"
            if not os.path.exists(saveDir):
                os.mkdir(saveDir)

            #对目录中的每个文件进行分词操作
            srcDir = self.__cutWordSrcDir + myDir + "/"
            fileList = os.listdir(srcDir)
            for fileName in fileList:
                filePath = srcDir + fileName
                content = file.readFile(filePath)

                # 删除换行和多余的空格
                content = content.replace("\r\n", "")

                contentSeg = jieba.cut(content)
                file.writeFile(saveDir + fileName," ".join(contentSeg))

    print("分词已完成")

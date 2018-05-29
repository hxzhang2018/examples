
#文件操作类
class File(object):

   def readFile(self,path):
       file = open(path,"r",encoding="utf-8")
       content = file.read()
       file.close()
       return content

   def writeFile(self,path,content):
       file = open(path,"w",encoding="utf-8")
       file.write(content)
       file.close()
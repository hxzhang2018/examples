from cutwords import CutWords

from bunched import  Bunched


def main():
    corpus_path = "F:/pythonDemo/textclassify/train_corpus_small/" #未分词语料库文件路径
    seg_path = "F:/pythonDemo/textclassify/train_corpus_seg/" #分词后的语料库文件路径

    word_bag_path = "F:/pythonDemo/textclassify/train_word_bag/train_set.dat"#文本树对象保存路径

    #分词
    cutWords = CutWords(corpus_path,seg_path)
    cutWords.cut()

    #构建文本树对象
    bunched = Bunched(seg_path,word_bag_path)
    bunched.bunched()


if __name__== "__main__":
    main()





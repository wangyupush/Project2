# -*- coding: utf-8 -*- 
import os
import pickle
import numpy as np
import operate_data as od
import ml_model as ml
import warnings
import sys
import xlrd
import xlwt
warnings.filterwarnings('ignore')

VECTOR_MODE = {'onehot': 0, 'wordfreq': 1, 'twovec': 2, 'tfidf': 3, 'outofdict': 4}

def save_model(best_vector,best_model):
 
    od.loadStopwords()
    od.loadEmotionwords()
    od.loadWords(od.stopList)
    od.loadDocument(od.stopList)
    xpath = os.path.join('result', 'vector', 'resultX.npz')
    ypath = os.path.join('result', 'vector', 'resultY.npz')
    resultX = np.load(xpath)
    resultY = np.load(ypath)
    new_x, new_y = od.twoTag(resultX[best_vector], resultY[best_vector])
    model_saved = ml.naiveBayes(new_x, new_y)
    path = os.path.join('model','wordfreq_naiveBayes.ml')
    with open(path,'wb') as f:
        pickle.dump(model_saved,f)
    print("Save over")

class Predictor(object):
   
    def __init__(self):
        self._model = None
        self.news = None
        self.__tag = None
        self._vec = None
        self.mode = None

    def load_model(self,path=None):
        if not path:
            path = os.path.join('model','wordfreq_naiveBayes.ml')

        with open(path,'rb') as f:
            self._model = pickle.load(f)

    def set_mode(self,mode):
        if isinstance(mode,int):
            assert mode in VECTOR_MODE.values(), "没有这种vector方式"
        if isinstance(mode,str):
            assert mode in VECTOR_MODE.keys(), "没有这种vector方式"
            mode = VECTOR_MODE[mode]
        self.mode = mode

    def set_news(self,news):
        if not len(news):
            print("请输入有效的新闻文本,谢谢")
            return
        self.news = news

    def trans_vec(self):
        vec_list = od.words2Vec(self.news,od.emotionList,od.stopList,od.posList,od.negList,mode=self.mode)
        self._vec = np.array(vec_list).reshape(1,-1)

    # 调用的时候计算函数
    def __call__(self, *args, **kwargs):
        self.__tag = self._model.predict(self._vec)
        return self.__tag

    def get_tag(self):
        return self.__tag


if __name__=='__main__':
    best_vector = "wordfreq"
    best_model = 1
    save_model(best_vector, best_model)

    predictor = Predictor()
    predictor.load_model()
    predictor.set_mode(mode="wordfreq") 
    
    data = xlrd.open_workbook('fulldata.xls')
    table=data.sheet_by_index(0)
    nrows = table.nrows
    ncols = table.ncols
    contents = table.col_values(4)
    row=0
    xf=0
    for news in contents:
        print(news)
        # 这是您的新闻样本
        predictor.set_news(news=news)
        predictor.trans_vec()
        tag = predictor()
        print("打标的结果是：",tag)
        # data.write(row,6,tag)
        if tag == -1:
            sqlfile=open("C:\\Users\\lenovo\\Desktop\\test\\result\\negtive\\"+ str(row)+'.txt','w')
            sqlfile.writelines(str(table.row_values(row)))
            sqlfile.close()
            row=row+1
        elif tag == 1:
            sqlfile=open("C:\\Users\\lenovo\\Desktop\\test\\result\\postive\\"+ str(row) +'.txt','w')
            sqlfile.writelines(str(table.row_values(row)))
            sqlfile.close()
            row=row+1
        else:
            sqlfile=open("C:\\Users\\lenovo\\Desktop\\test\\result\\neutral\\"+ str(row) +'.txt','w')
            sqlfile.writelines(str(table.row_values(row)))
            sqlfile.close()
            row=row+1

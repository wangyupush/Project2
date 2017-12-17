# Project2
## 实验任务
### 对上市公司的财经新闻进行情感分析
## 实验设计思路
### step1 处理训练集,将训练集文本转换成向量形式
#### 首先将训练文本预处理，这里我将所有训练文本转换为UTF-8模式，然后对文本进行分词,文本向量化时，主要采用方式TF-idf
##### TF—IDF部分的程序设计
###### TF-IDF原理公式 TFIDF = TF * IDF;
![tf](https://github.com/wangyupush/Project2/blob/master/tf.png)
![idf](https://github.com/wangyupush/Project2/blob/master/idf.png)
###### TF—IDF部分的代码
![tfidf](https://github.com/wangyupush/Project2/blob/master/tfidf.png)
### step2 文本向量化之后，进行模型训练,这里我使用了开源机器学习库：sklearn,可以支持5种分类方法:KNN、逻辑回归、决策树、支持向量机和朴素贝叶斯
#### 调用代码如下：
##### KNN:
![knn](https://github.com/wangyupush/Project2/blob/master/knn.png)
##### 逻辑回归：
![logic](https://github.com/wangyupush/Project2/blob/master/logic.png)
##### 决策树：
![tree](https://github.com/wangyupush/Project2/blob/master/tree.png)
##### 支持向量机：
![svm](https://github.com/wangyupush/Project2/blob/master/svm.png)
##### 朴素贝叶斯：
![bayes](https://github.com/wangyupush/Project2/blob/master/bayes.png)
### step3 在进行分类预测时，可以选择文本向量化的方式以及机器学习模型，将输入文件转换成excel文件，按输出的tag标签的值，分别存在postive、negtive和neutral的文件夹中

## 程序不足：主要在tf-idf部分，这部分复杂度为O(2*n*n)，所以在运行时需要很长很长时间，在分类预测时，也是同样的问题，运行的效率很慢，后期可能会尝试添加多线程来加快效率。

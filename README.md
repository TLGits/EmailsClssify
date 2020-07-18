# EmailsClssify
tf-idf方法文本特征提取，通过SVM分类


  
Document Classification (Email Classfy)

文件：
install_database.py：安装分词处理工具脚本；

Emails_classify_Proprocessing.py：数据预处理脚本；

Emails_classify_svm.py：svm分类脚本；

Emails_classify_SMO.py: SMO分类文件 

smo_label_list.npy：SMO标签文件

smo_mail_list.npy：SMO邮件内容文件

requirements.txt：安装依赖

lemmatization_preed_data.csv：词形还原特征文件

stemming_preed_data.csv：词干提取特征文件

report_linear_lem.txt：词形还原特征+linear核SVM分类输出文件

report_linear_stem.txt：词干提取特征+linear核SVM分类输出文件

report_rbf_lem.txt：词形还原特征+rbf核SVM分类输出文件

report_rbf_stem.txt：词干提取特征+rbf核SVM分类输出文件


1. 提取特征方法
    读取文章内容：从第四行——末尾，避免前两行邮件收发人信息影响文件内容；

    去除标点符号：使用正则表达式去除标点符号，并将每个单词用一个空格分开；

    词干提取：将所有的单词转化为小写，利用nltk包进行词干提取；

    词形还原：判断单词词性，利用nltk包根据词性将其还原成原始形态，例如running——run；

    去除停用词：下载nltk包中的标准stopwords，对每一个单词进行判断，例如是删除'the'等词汇；

    特征提取：采用tf-idf方法，计算在每一个邮件中，每一个单词及其相对应的value
    
2. SVM分类主函数
    SVM分类，调用sklearn-learn中的SVM分类。采用svm.SVC 分类方法，核函数可以通过运行框进行自主选择，一般默认为‘RBF’。运行缓存定位800MB，尽量加快训练速度。
    通过数据读取，数据格式划分，建立模型，模型拟合，计算评价指标，完成整个训练。

3. 程序运行方法：
    所有程序均在Python3.6环境下运行

    1）安装依赖：pip install -r requirements.txt

    2)  安装分词处理工具：python install_database.py

    3)  数据预处理：python Emails_classify_Proprocessing.py

	|  Short   | Long  |  Default  |  Description  |

	|  -h  | --help |  ----  |  Show help  |

	| -m  | --method |  stemming  |  method 词干提取stemming或词形还原lemmatization

       例：python Emails_classify_Proprocessing.py -m lemmatization：采用词形还原方法变换；
       生成 method+"_preed_data.csv" 特征文件包含对每一个文件中相应的词value大小
    4) SVM分类：python Emails_classify_svm.py
    
	|  Short   | Long  |  Default  |  Description  |
	
	|  -h  | --help |  ----  |  Show help  |
	
	| -m  | --method |  rbf  |  method 选择核函数，(linear or rbf or poly or sigmoid)
	
	| -f  | --file |  stemming  |  file 选择采用数据集类型(stemming or lemmatization)
       例：python Emails_classify_svm.py -m linear ：采用线性核函数；
       
    5）SMO算法：python Emails_classify_SMO.py 可以看到相应的a，b的输出
    

4. 结果保存：
      如果要保存输出结果，可采用python Emails_classify_svm.py -m linear -f lemmatization> report_linear.txt结果保存在report_linear.txt 文本中。

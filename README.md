# Intent Classification Baseline

[中文语音助手意图识别基线系统](https://github.com/)，外部专有名词知识、训练模型和预测样本的流程。

By W. Y. Shen 

工程：https://github.com/GhibliField/Intent-Classification-Baseline-System

## 使用方式

下载工程

```text
git clone https://github.com/GhibliField/Intent-Classification-Baseline-System.git
```



安装Python依赖库

```text
pip install -r requirements.txt
```


## 示例工程



运行服务端：

```text
python ServerDemo.py 
```

预测：

```text
python test.py 
```


## 工程架构

### 流程图

![架构](doc/arc.png)

### 文件夹结构

```text
├── config.py
├── data
│   ├── devel.json
│   └── train.json
├── models
│   ├── ltp_data_v3.4.0/
│   ├── some.pkl
│   └── ...
├── README.md
├── RequestHandler.py
├── requirements.txt
├── ServerDemo.py
├── test.py
├── trainers
│   ├── classifiers.py
│   ├── NB-SVM_with_chi2.ipynb
│   ├── SVM_epg-tvchannel.ipynb
│   ├── SVM_video-cinemas.ipynb
│   ├── SVM_video-epg.ipynb
│   ├── SVM_video-music.ipynb
│   └── SVM_website-app.ipynb
└── utils
    ├── predicates.tsv
    ├── pretty_sure.tsv
    ├── proper_noun/
    ├── proper_noun_all.tsv
    ├── rules.tsv
    ├── stopword_2792.txt
    └── utils.py

```

## 主要组件

- data: 训练数据
- model: 持久化了的分类模型与必要的对象
- trainer: notebook脚本,用于训练不同的分类器
- utils: 外部词库,方法等
- config.py: 配置路径参数, **请务必修改路径**
- RequestHandler.py: 测试流程
- ServerDemo.py: Flask 测试服务端
- test.py: 测试脚本

## 感谢

- [MrGemy95/Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template)
- [lrei/NB-SVM](https://github.com/lrei/nbsvm.git)
- [WindInWillows/SMP2018-ECDT-TASK1](https://github.com/WindInWillows/SMP2018-ECDT-TASK1)





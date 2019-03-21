# YOLO-V3-loss-IOU-avg-Recall-
ubuntu16.04 darknet网络，绘制yolov3,yolov3-tiny等网络训练过程中参数可视化的loss以及iou
可视化中间参数需要用到训练时保存的log文件（命令中的路径根据自己实际修改）： ./darknet detector train pds/fish/cfg/fish.data pds/fish/cfg/yolov3-fish.cfg darknet53.conv.74 2>1 | tee visualization/train_yolov3.log 
在使用脚本绘制变化曲线之前，需要先使用extract_log.py脚本，格式化log,用生成的新的log文件供可视化工具绘图，格式化log的extract_log.py脚本如下（和生成的log文件同一目录）：
# coding=utf-8
# 该文件用来提取训练log，去除不可解析的log后使log文件格式化，生成新的log文件供可视化工具绘图
 
import inspect
import os
import random
import sys
def extract_log(log_file,new_log_file,key_word):
    with open(log_file, 'r') as f:
      with open(new_log_file, 'w') as train_log:
  #f = open(log_file)
    #train_log = open(new_log_file, 'w')
        for line in f:
    # 去除多gpu的同步log
          if 'Syncing' in line:
            continue
    # 去除除零错误的log
          if 'nan' in line:
            continue
          if key_word in line:
            train_log.write(line)
    f.close()
    train_log.close()
 
extract_log('train_yolov3.log','train_log_loss.txt','images')
extract_log('train_yolov3.log','train_log_iou.txt','IOU')

运行之后，会解析log文件的loss行和iou行得到两个txt文件

使用train_loss_visualization.py脚本可以绘制loss变化曲线 
train_loss_visualization.py脚本如下（也是同一目录新建py文件）：

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
 
lines =5124    #改为自己生成的train_log_loss.txt中的行数
result = pd.read_csv('train_log_loss.txt', skiprows=[x for x in range(lines) if ((x%10!=9) |(x<1000))] ,error_bad_lines=False, names=['loss', 'avg', 'rate', 'seconds', 'images'])
result.head()
 
result['loss']=result['loss'].str.split(' ').str.get(1)
result['avg']=result['avg'].str.split(' ').str.get(1)
result['rate']=result['rate'].str.split(' ').str.get(1)
result['seconds']=result['seconds'].str.split(' ').str.get(1)
result['images']=result['images'].str.split(' ').str.get(1)
result.head()
result.tail()
 
# print(result.head())
# print(result.tail())
# print(result.dtypes)
 
print(result['loss'])
print(result['avg'])
print(result['rate'])
print(result['seconds'])
print(result['images'])
 
result['loss']=pd.to_numeric(result['loss'])
result['avg']=pd.to_numeric(result['avg'])
result['rate']=pd.to_numeric(result['rate'])
result['seconds']=pd.to_numeric(result['seconds'])
result['images']=pd.to_numeric(result['images'])
result.dtypes
 
 
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(result['avg'].values,label='avg_loss')
# ax.plot(result['loss'].values,label='loss')
ax.legend(loc='best')  #图列自适应位置
ax.set_title('The loss curves')
ax.set_xlabel('batches')
fig.savefig('avg_loss')
# fig.savefig('loss')

修改train_loss_visualization.py中lines为train_log_loss.txt行数，并根据需要修改要跳过的行数：

skiprows=[x for x in range(lines) if ((x%10!=9) |(x<1000))]
此处我改成了iou的

运行train_loss_visualization.py会在脚本所在路径生成avg_loss.png。
可以通过分析损失变化曲线，修改cfg中的学习率变化策略。

除了可视化loss，还可以可视化Avg IOU，Avg Recall等参数 
可视化’Region Avg IOU’, ‘Class’, ‘Obj’, ‘No Obj’, ‘Avg Recall’,’count’这些参数可以使用脚本train_iou_visualization.py，使用方式和train_loss_visualization.py相同，train_iou_visualization.py脚本如下（#lines根据train_log_iou.txt的行数修改）：

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
 
lines = 122956    #根据train_log_iou.txt的行数修改
result = pd.read_csv('train_log_iou.txt', skiprows=[x for x in range(lines) if (x%10==0 or x%10==9) ] ,error_bad_lines=False, names=['Region Avg IOU', 'Class', 'Obj', 'No Obj', 'Avg Recall','count'])
result.head()
 
result['Region Avg IOU']=result['Region Avg IOU'].str.split(': ').str.get(1)
result['Class']=result['Class'].str.split(': ').str.get(1)
result['Obj']=result['Obj'].str.split(': ').str.get(1)
result['No Obj']=result['No Obj'].str.split(': ').str.get(1)
result['Avg Recall']=result['Avg Recall'].str.split(': ').str.get(1)
result['count']=result['count'].str.split(': ').str.get(1)
result.head()
result.tail()
 
# print(result.head())
# print(result.tail())
# print(result.dtypes)
print(result['Region Avg IOU'])
 
result['Region Avg IOU']=pd.to_numeric(result['Region Avg IOU'])
result['Class']=pd.to_numeric(result['Class'])
result['Obj']=pd.to_numeric(result['Obj'])
result['No Obj']=pd.to_numeric(result['No Obj'])
result['Avg Recall']=pd.to_numeric(result['Avg Recall'])
result['count']=pd.to_numeric(result['count'])
result.dtypes
 
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(result['Region Avg IOU'].values,label='Region Avg IOU')
# ax.plot(result['Class'].values,label='Class')
# ax.plot(result['Obj'].values,label='Obj')
# ax.plot(result['No Obj'].values,label='No Obj')
# ax.plot(result['Avg Recall'].values,label='Avg Recall')
# ax.plot(result['count'].values,label='count')
ax.legend(loc='best')
# ax.set_title('The Region Avg IOU curves')
ax.set_title('The Region Avg IOU curves')
ax.set_xlabel('batches')
# fig.savefig('Avg IOU')
fig.savefig('Region Avg IOU')

运行train_iou_visualization.py会在脚本所在路径生成相应的曲线图。

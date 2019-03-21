# YOLO-V3-loss-IOU-avg-Recall-
ubuntu16.04 darknet网络，绘制yolov3,yolov3-tiny等网络训练过程中参数可视化的loss以及iou
可视化中间参数需要用到训练时保存的log文件（命令中的路径根据自己实际修改）： ./darknet detector train pds/fish/cfg/fish.data pds/fish/cfg/yolov3-fish.cfg darknet53.conv.74 2>1 | tee visualization/train_yolov3.log 
在使用脚本绘制变化曲线之前，需要先使用extract_log.py脚本，格式化log,用生成的新的log文件供可视化工具绘图，格式化log的extract_log.py脚本如下（和生成的log文件同一目录）：
运行之后，会解析log文件的loss行和iou行得到两个txt文件

使用train_loss_visualization.py脚本可以绘制loss变化曲线 
train_loss_visualization.py脚本如下（也是同一目录新建py文件）：
修改train_loss_visualization.py中lines为train_log_loss.txt行数，并根据需要修改要跳过的行数：

skiprows=[x for x in range(lines) if ((x%10!=9) |(x<1000))]
此处我改成了iou的

运行train_loss_visualization.py会在脚本所在路径生成avg_loss.png。
可以通过分析损失变化曲线，修改cfg中的学习率变化策略。

除了可视化loss，还可以可视化Avg IOU，Avg Recall等参数 
可视化’Region Avg IOU’, ‘Class’, ‘Obj’, ‘No Obj’, ‘Avg Recall’,’count’这些参数可以使用脚本train_iou_visualization.py，使用方式和train_loss_visualization.py相同，train_iou_visualization.py脚本如下（#lines根据train_log_iou.txt的行数修改）：
运行train_iou_visualization.py会在脚本所在路径生成相应的曲线图。

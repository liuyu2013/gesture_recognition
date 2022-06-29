#手势识别
from ast import arg
import cv2
import cvzone
from matplotlib.cbook import pts_to_midstep
import numpy as np
from cvzone.HandTrackingModule import HandDetector  # 导入手部检测模块
import math
import random
import mediapipe as mp
import suanpan
from suanpan.app import app
from suanpan.app.arguments import String, Json, Int, Float, Bool
from suanpan.log import logger

from components.until.snake import SnakeGameClass
@app.input(String(key="inputData1", alias="msgin", default="Suanpan"))

@app.param(Int(key="param0", alias="camera"))
@app.param(Int(key="param1", alias="width"))
@app.param(Int(key="param2", alias="hight"))
@app.param(Int(key="param3", alias="max_num_hands"))
@app.param(Float(key="param4", alias="min_detection_confidence"))
@app.param(Bool(key="param5", alias="save_video"))
@app.param(String(key="param6", alias="save_path"))

@app.output(String(key="outputData1", alias="result"))
def hands_sanke(context):
    args = context.args
    #（1）获取摄像头
    camera = 0
    if args.camera is not None:
        camera = args.camera
    cap = cv2.VideoCapture(camera) # 0代表电脑自带的摄像头
    # 设置显示摄像头的分辨率
    w, h = args.param1, args.param2
    cap.set(3, w)  # 窗口宽1280
    cap.set(4, h)   # 窗口高720

    # 是否需要保存该视频
    # 定义编解码器并创建VideoWriter对象
    is_save_video = False if args.param5 is None else True
    max_num_hands = args.param3 if args.param3 is not None else 1
    min_detection_confidence = args.param4 if args.param4 is not None else 0.8

    if is_save_video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_video = cv2.VideoWriter(args.param6, fourcc, 20.0, (w,h))

    #（2）模型配置
    hands = mp.solutions.hands
    hand_landmark = hands.Hands(min_detection_confidence=min_detection_confidence, max_num_hands=max_num_hands) #只检测一只手
    draw = mp.solutions.drawing_utils

    game = SnakeGameClass()

    while True:
        #（3）图像处理
        # 每次读取一帧相机图像，返回是否读取成功success，读取的帧图像frm
        success, frm = cap.read()
        # 图像翻转，使图像和自己呈镜像关系
        frm = cv2.flip(frm, 1)# 0代表上下翻转，1代表左右翻转

        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

        # 检测手部关键点。返回手部信息hands，绘制关键点后的图像img
        op = hand_landmark.process(rgb)
        #（4）关键点处理 
        # print(op.multi_hand_landmarks) #长度为1
        pointIndex = [0,0]
        if op.multi_hand_landmarks:    #如果检测到手
            for i in op.multi_hand_landmarks:
                draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)   #绘制手的21个关键点，在原图上标记
                pointIndex = [int(i.landmark[8].x*w), int(i.landmark[8].y*h)]  #只获取食指指尖关键点的（x,y）坐标,一定要乘以设置的面板长、高
                # cv2.imshow('frm', frm)
                frm = game.update(frm, pointIndex)
        #（5）显示图像,保存视频
        if is_save_video:
            out_video.write(frm)
        cv2.imshow('frm', frm)  # 输入图像显示窗口的名称及图像
        k = cv2.waitKey(1)  # 每帧滞留1毫秒后消失
        logger.info(game.gameover)
        if k == ord('r'):  # 键盘'r'键代表重新开始游戏
            game.gameover = False
            game.score = 0  # 积分器
            game.points = []  # 蛇的身体的节点坐标
            game.lengths = []  # 蛇身各个节点之间的坐标
            game.currentLength = 0  # 当前蛇身长度
            game.allowedLength = 150  # 没吃东西时，蛇的总长度
            game.previousHead = (0,0)  # 前一个蛇头节点的坐标                
            game.randomFoodLocation()  # 随机改变食物的位置
        if k & 0xFF == 27:  # 键盘ESC键退出程序
            break
    # 释放视频资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    suanpan.run(app)

#通过设定不同的手势完成音量的控制
#手势识别
from ast import arg
import cv2
from matplotlib.cbook import pts_to_midstep
import numpy as np
import math
import random
import mediapipe as mp
import suanpan
from suanpan.app import app
from suanpan.app.arguments import String, Json, Int, Float, Bool
from suanpan.log import logger
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

@app.input(String(key="inputData1", alias="msgin", default="Suanpan"))
@app.param(Int(key="param0", alias="camera"))
@app.param(Int(key="param1", alias="width"))
@app.param(Int(key="param2", alias="hight"))
@app.param(Int(key="param3", alias="max_num_hands"))
@app.param(Float(key="param4", alias="min_detection_confidence"))
@app.param(Bool(key="param5", alias="save_video"))
@app.param(String(key="param6", alias="save_path"))
@app.output(Json(key="outputData1", alias="result"))
def control_volume(context):
    args = context.args
    #（1）获取摄像头
    camera = 0
    if args.camera is not None:
        camera = args.camera
    cap = cv2.VideoCapture(camera) # 0代表电脑自带的摄像头
    # 设置显示摄像头的分辨率
    w, h = args.param1, args.param2
    # w, h = 1280,720
    cap.set(3, w)  # 分辨率宽
    cap.set(4, h)   # 分辨率高

    #是否需要保存该视频
    #定义编解码器并创建VideoWriter对象
    is_save_video = args.param5 if args.param5 is not None else False
    max_num_hands = args.param3 if args.param3 is not None else 1
    min_detection_confidence = args.param4 if args.param4 is not None else 0.8

    if is_save_video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_video = cv2.VideoWriter(args.param6, fourcc, 20.0, (w,h))

    #（2）模型配置
    hands = mp.solutions.hands
    hand_landmark = hands.Hands(min_detection_confidence=min_detection_confidence, max_num_hands=max_num_hands) #只检测一只手
    draw = mp.solutions.drawing_utils

    # 识别系统音量设备
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
    volume = cast(interface,POINTER(IAudioEndpointVolume))
 
    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]
    # 用食指+大拇指来上下控制
    while True:
        #（3）图像处理
        # 每次读取一帧相机图像，返回是否读取成功success，读取的帧图像img
        success, frm = cap.read()
        
        # 图像翻转，使图像和自己呈镜像关系
        frm = cv2.flip(frm, 1)# 0代表上下翻转，1代表左右翻转
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        # 检测手部关键点。返回手部信息hands，绘制关键点后的图像img
        op = hand_landmark.process(rgb)

        #（4）关键点处理 
        # print(op.multi_hand_landmarks) #长度为1
        if op.multi_hand_landmarks:    #如果检测到手
            for i in op.multi_hand_landmarks:
                draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)   #绘制手的21个关键点，在原图上标记
                x1, y1 = int(i.landmark[4].x*w), int(i.landmark[4].y*h)  #只获取食指指尖关键点的（x,y）坐标,一定要乘以设置的面板长、高
                x2, y2 = int(i.landmark[8].x*w), int(i.landmark[8].y*h)  #只获取食指指尖关键点的（x,y）坐标,一定要乘以设置的面板长、高
                cx, cy = (x1+x2)//2, (y1+y2)//2

                cv2.circle(frm,(x1,y1),15,(255,0,255),cv2.FILLED)
                cv2.circle(frm,(x2,y2),15,(255,0,255), cv2.FILLED)
                cv2.line(frm,(x1,y1),(x2,y2),(255,0,255),3)
                cv2.circle(frm, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                length = math.hypot(x2-x1,y2-y1)

                # 线段长度最大300，最小50，转换到音量范围，最小-65，最大0
                # 将线段长度变量length从[50,300]转变成[-65,0]
                vol = np.interp(length, [50,300], [minVol,maxVol])
                logger.info('vol:', vol, 'length:', length)
                volBar = np.interp(length, [50,300], [400,150])
                #设置电脑主音量
                volume.SetMasterVolumeLevel(vol,None)
                if length < 50:  # 距离小于50改变中心圆颜色绿色
                    cv2.circle(frm, (cx,cy), 12, (0,255,0), cv2.FILLED)
                cv2.rectangle(frm, (50,150), (85,400), (0,0,255), 3)
                # 用音量的幅度作为填充矩形条的高度，像素坐标是整数
                cv2.rectangle(frm, (50,int(volBar)), (85,400), (0,0,255), cv2.FILLED)
                text_vol = 100 * (volBar-150)/(400-150)   # 音量归一化再变成百分数
                cv2.putText(frm, f'{str(int(text_vol))}%', (50-5,150-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        #（5）显示图像，保存图像
        if is_save_video:
            out_video.write(frm)
        cv2.imshow('frm', frm)  # 输入图像显示窗口的名称及图像
        
        k = cv2.waitKey(1)  # 每帧滞留1毫秒后消失
        if k & 0xFF == 27:  # 键盘ESC键退出程序
            break
    # 释放视频资源
    cap.release()
    cv2.destroyAllWindows() 
    result = args.save_path if is_save_video else "success"
    app.send({"out1":result})

if __name__ == "__main__":
    suanpan.run(app)
    # control_volume("")

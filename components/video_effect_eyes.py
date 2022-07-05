import numpy as np
import statistics
import math
from ast import arg
import cv2
from matplotlib.cbook import pts_to_midstep
import numpy as np
import math
import mediapipe as mp
import suanpan
from suanpan.app import app
from suanpan.app.arguments import String, Json, Int, Float, Bool
from suanpan.log import logger

@app.input(String(key="inputData1", alias="msgin", default="Suanpan"))
@app.param(Int(key="param0", alias="camera"))
@app.param(Int(key="param1", alias="width"))
@app.param(Int(key="param2", alias="hight"))
@app.param(Float(key="param3", alias="min_tracking_confidence"))
@app.param(Float(key="param4", alias="min_detection_confidence"))
@app.param(Bool(key="param5", alias="save_video"))
@app.param(String(key="param6", alias="save_path"))
@app.param(String(key="param7", alias="image_path"))
@app.param(Float(key="param8", alias="factor"))
@app.output(String(key="outputData1", alias="result"))

def video_effect_eyes(context):
    args = context.args
    #（1）获取摄像头
    camera = 0
    if args.camera is not None:
        camera = args.camera
    cap = cv2.VideoCapture(camera) # 0代表电脑自带的摄像头
    # 设置显示摄像头的分辨率
    w, h = args.param1, args.param2
    # w,h = 1280,720
    cap.set(3, w)  # 窗口宽1280
    cap.set(4, h)   # 窗口高720

    # 是否需要保存该视频
    # 定义编解码器并创建VideoWriter对象
    is_save_video = args.param5 if args.param5 is not None else False
    min_tracking_confidence = args.param3 if args.param3 is not None else 0.5
    min_detection_confidence = args.param4 if args.param4 is not None else 0.5

    if is_save_video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_video = cv2.VideoWriter(args.param6, fourcc, 20.0, (w,h))

    #（2）模型配置
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # 依照MediaPipe提供的「人脸网格标注图」绘制眼睛周围的点
    left_eye_list = [35, 124, 46, 53, 52, 65, 55, 8, 285, 295, 282, 283, 276, 353, 265]
    right_eye_list = [31, 228, 229, 230, 231, 232, 233, 244, 261, 448, 449, 450,451, 452, 453, 464, 245, 122, 6, 351, 465]
    eye_list = left_eye_list + right_eye_list


    # 载入特效图片
    image_path = args.param7
    # image_path = "D:\site_tools\gesture\gesture_recognition\ly_test\glass.png"
    eye_normal = cv2.imread(image_path)

    # 开始检测
    with mp_face_mesh.FaceMesh(
        min_detection_confidence = min_detection_confidence,
        min_tracking_confidence = min_tracking_confidence) as face_mesh:
        while cap.isOpened():
            #取得视频的一帧
            success, frm = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            h, w, d = frm.shape #取得图像的宽，长
            #转为rgb
            image = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            if results.multi_face_landmarks:
                # 遍历左眼、右眼坐标
                for face_landmarks in results.multi_face_landmarks:
                    eye_point = []
                    eye_width = []
                    eye_height = []
                    for index in eye_list:
                        x = int(face_landmarks.landmark[index].x * w) #点的原始坐标
                        y = int(face_landmarks.landmark[index].y * h)
                        eye_point.append([x, y])
                        if index == 35 or index == 265:  #表示眼睛宽
                            eye_width.append([x, y])
                        if index == 52 or index == 230:  #表示眼睛高
                            eye_height.append([x, y])
                    if len(eye_width) == 2:
                        eye_wlen = int(math.pow(math.pow((eye_width[0][0] - eye_width[1][0]), 2) + math.pow((eye_width[0][1] - eye_width[1][1]), 2), 0.5)) #眼睛的长度14

                try:
                # 將取得的个点坐标statistics計算出眼睛中心坐标
                    points = eye_point
                    center = [statistics.mean(i) for i in zip(*points)]  #获取眼睛的中心坐标
                    
                    # print(eye_len)
                    height, width, _ = eye_normal.shape 

                    # 通过检测的人体眼睛长度以及高度来resize读取的特效图片
                    factor = args.factor if args.factor is None else 1.5 #缩放因子
                    # factor = 1.5
                    eye = cv2.resize(eye_normal, (int(eye_wlen * factor), int(height * eye_wlen * factor / width))) #整个的长度，高度等比例缩放
                

                    eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY) #转成rgb
                    _, eye_mask = cv2.threshold(eye_gray, 10, 255, cv2.THRESH_BINARY_INV) #提取图片重要的轮廓

                    mask_inv = cv2.bitwise_not(eye_mask)

                    img_height, img_width, _ = eye.shape
                    x, y = int(center[0]-img_width/2), int(center[1]-img_height/2) #确定插入到人脸的开始坐标
                    eye_area = frm[y: y+img_height, x: x+img_width] #在人脸上确定图片的位置
                    #在原图上抠出来需要预留的logo
                    eye_area_no_eye = cv2.bitwise_and(eye_area, eye_area, mask=mask_inv)

                    #抠出logo的本身图
                    eye_log = cv2.bitwise_and(eye, eye, mask=eye_mask)  #跟原图刚好相反


                    final_eye = cv2.add(eye_area_no_eye, eye_log)

                    frm[y: y+img_height, x: x+img_width] = final_eye
                except:
                    pass
            if is_save_video:
                out_video.write(frm)

            cv2.imshow("MediaPipe FaceMesh", frm)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        result = args.save_path if is_save_video else "success"
        app.send({"out1":result})


if __name__ == "__main__":
    suanpan.run(app)
    # video_effect_eyes("")


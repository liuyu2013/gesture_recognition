import cv2
import mediapipe as mp
import numpy as np
import suanpan
from suanpan.app import app
from suanpan.app.arguments import String, Json, Int, Float, Bool
from suanpan.log import logger

@app.input(String(key="inputData1", alias="msgin", default="Suanpan"))
@app.param(Float(key="param0", alias="min_tracking_confidence"))
@app.param(Float(key="param1", alias="min_detection_confidence"))
@app.param(String(key="param2", alias="image_path"))
@app.param(String(key="param3", alias="effect_path"))
@app.param(Float(key="param4", alias="factor"))
@app.param(Int(key="param5", alias="threshold"))
@app.param(Int(key="param6", alias="dw"))
@app.param(Int(key="param7", alias="dh"))
@app.param(String(key="param8", alias="save_path"))
@app.output(String(key="outputData1", alias="result"))
def image_effect_head(context):
    args = context.args
    # 绘图/标注方法、人脸标注网格方法
    min_tracking_confidence = args.param0 if args.param0 is not None else 0.5
    min_detection_confidence = args.param1 if args.param1 is not None else 0.5
    effect_path = args.effect_path
    # effect_path = "D:\site_tools\gesture\gesture_recognition\ly_test\hat.jpg"

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    head_list = [33,130, 243, 359, 463, 164, 263]  #左右眼角5个关键点
    frm = cv2.imread(args.image_path,cv2.IMREAD_REDUCED_COLOR_2)  # 读取人脸图像
    # frm = cv2.imread("D:\site_tools\gesture\gesture_recognition\ly_test\peoples.png",cv2.IMREAD_REDUCED_COLOR_2)  # 读取人脸图像
    #载入特效图片
    hat = cv2.imread(effect_path, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("no_sunglasses_face", hat)
    # r,g,b,a = cv2.split(hat)
    # rgb_hat = cv2.merge((r,g,b))
    rgb_hat = hat

    save_path = args.save_path
    # save_path = "output1.png"
    # 开始检测
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence) as face_mesh:
        h, w, d = frm.shape #取得图像的宽，长
        image = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            # 歷遍左眼各點取得目前座標與眼睛大小
            for face_landmarks in results.multi_face_landmarks:
                point1 = []
                point2 = []
                # x,y = int(face_landmarks.landmark[234].x * w), int(face_landmarks.landmark[10].y * h), 
                for index in head_list:
                    x = int(face_landmarks.landmark[index].x * w) #点的原始坐标
                    y = int(face_landmarks.landmark[index].y * h)
                    if index == 33:  #左眼眼角
                        point1.append([x, y])
                    if index == 263:  #右眼眼角
                        point2.append([x, y])
                    
                    
                # 求两点中心
                eyes_center = ((point1[0][0]+point2[0][0])//2,(point1[0][1]+point2[0][1])//2)

                #  根据人脸大小调整帽子大小
                factor = args.factor if args.factor is None else 1.5 #缩放因子
                # factor = 1.5
                fw = int(face_landmarks.landmark[454].x * w) -int(face_landmarks.landmark[234].x * w) #d.right()-d.left()
                resized_hat_h = int(round(rgb_hat.shape[0]*fw/rgb_hat.shape[1]*factor))
                resized_hat_w = int(round(rgb_hat.shape[1]*fw/rgb_hat.shape[1]*factor))
                y =  int(face_landmarks.landmark[10].y * h)  #脸部的top
                if resized_hat_h > y:
                    resized_hat_h = y-1

                # 根据人脸大小调整帽子大小
                resized_hat = cv2.resize(rgb_hat,(resized_hat_w,resized_hat_h))
                # print(resized_hat.shape)
                hat_gray = cv2.cvtColor(resized_hat, cv2.COLOR_BGR2GRAY) #转成rgb

                threshold = args.threshold #颜色越鲜明值可设置小，越不容易分清，则越大
                # threshold = 240
                _, hat_mask = cv2.threshold(hat_gray, threshold, 255, cv2.THRESH_BINARY_INV)

                mask_inv = cv2.bitwise_not(hat_mask)

                #确定帽子相对人脸的偏移位置
                dh = args.dh if args.dh is not None else 0 
                dw = args.dw if args.dw is not None else 0 
                # dh, dw = 10,0
                #原图中提取存放帽子的区域
                hat_area = frm[y+dh-resized_hat_h:y+dh, (eyes_center[0]-resized_hat_w//2 + dw):(eyes_center[0]+resized_hat_w//2 + dw)]
                # cv2.imshow("MediaPipe FaceMesh", hat_area)
                
                hat_area = hat_area.astype(float)
                mask_inv = cv2.merge((mask_inv, mask_inv, mask_inv))
                alpha = mask_inv.astype(float)/255
                alpha = cv2.resize(alpha,(hat_area.shape[1],hat_area.shape[0]))

                bg = cv2.multiply(alpha, hat_area)
                bg = bg.astype('uint8')
                
                #帽子logo抠图
                img2_fg = cv2.bitwise_and(resized_hat, resized_hat, mask = hat_mask)

                img2_fg = cv2.resize(img2_fg, (hat_area.shape[1], hat_area.shape[0]))
                add_hat = cv2.add(bg, img2_fg)

                frm[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//2 + dw):(eyes_center[0]+resized_hat_w//2 + dw)] = add_hat

        cv2.imwrite(save_path, frm)  # 保存图像文件     
        cv2.imshow("MediaPipe FaceMesh", frm)
        cv2.waitKey()  # 按下任何键盘按键后
    cv2.destroyAllWindows()
    app.send({"out1":args.save_path})
if __name__ == "__main__":
    suanpan.run(app)
    # image_effect_head("")
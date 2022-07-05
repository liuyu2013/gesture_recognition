import cv2
import numpy as np
from util.HandTrackingModule import HandDetector
import time
import autopy
import suanpan
from suanpan.app import app
from suanpan.app.arguments import String, Json, Int, Float, Bool
from suanpan.log import logger


@app.input(String(key="inputData1", alias="msgin", default="Suanpan"))
@app.param(Int(key="param0", alias="camera"))
@app.param(Int(key="param1", alias="width"))
@app.param(Int(key="param2", alias="hight"))
@app.param(Int(key="param3", alias="max_num_hands"))
@app.param(Float(key="param4", alias="min_detection_confidence"))
@app.param(Bool(key="param5", alias="save_video"))
@app.param(String(key="param6", alias="save_path"))
@app.output(Json(key="outputData1", alias="result"))
def virtual_mouse(context):
    args = context.args
    #（1）获取摄像头
    camera = 0
    if args.camera is not None:
        camera = args.camera
    cap = cv2.VideoCapture(camera) # 0代表电脑自带的摄像头
    cap = cv2.VideoCapture(0)
    # 设置显示摄像头的分辨率
    w, h = args.param1, args.param2 #640, 480
    # w, h = 640, 480
    cap.set(3, w)  # 分辨率宽
    cap.set(4, h)   # 分辨率高

    #是否需要保存该视频
    #定义编解码器并创建VideoWriter对象
    is_save_video = args.param5 if args.param5 is not None else False
    max_num_hands = args.param3 if args.param3 is not None else 2
    min_detection_confidence = args.param4 if args.param4 is not None else 0.8

    frameR = 100     #Frame Reduction
    smoothening = 7  #random value
    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    if is_save_video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_video = cv2.VideoWriter(args.param6, fourcc, 20.0, (w,h))

    detector = HandDetector(detectionCon=min_detection_confidence, maxHands=max_num_hands) # 0.8 , 2
    # detector = HandDetector(detectionCon=0.8, maxHands=2) # 0.8 , 2
    wScr, hScr = autopy.screen.size()
    # 用食指+中指来控制，食指单个表示移动，点击时中指再抬起此时两个手指操作点击
    while True:
        # Step1: Find the landmarks
        success, frm = cap.read()
        hands, frm = detector.findHands(frm)
        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right


            # Step2: Get the tip of the index and middle finger
            if len(lmList1) != 0:
                x1, y1 = lmList1[8]
                x2, y2 = lmList1[12]
                # Step3: Check which fingers are up
                fingers1 = detector.fingersUp(hand1)
                cv2.rectangle(frm, (frameR, frameR), (w - frameR, h - frameR),
                            (255, 0, 255), 2)

                # Step4: Only Index Finger: Moving Mode
                if fingers1[1] == 1 and fingers1[2] == 0:

                    # Step5: Convert the coordinates
                    x3 = np.interp(x1, (frameR, w-frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, h-frameR), (0, hScr))
                    # Step6: Smooth Values
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening
                    # Step7: Move Mouse
                    autopy.mouse.move(wScr - clocX, clocY)
                    cv2.circle(frm, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    plocX, plocY = clocX, clocY

                # Step8: Both Index and middle are up: Clicking Mode
                if fingers1[1] == 1 and fingers1[2] == 1:
                    # Step9: Find distance between fingers
                    length, lineInfo,frm = detector.findDistance(lmList1[8], lmList1[12], frm)
                    # Step10: Click mouse if distance short
                    if length < 40:
                        cv2.circle(frm, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                        autopy.mouse.click()
        # Step11: Frame rate
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(frm, str(int(fps)), (28, 58), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 8), 3)
        # Step12: Display
        #显示图像，保存图像
        if is_save_video:
            out_video.write(frm)
        cv2.imshow("Image", frm)

        k = cv2.waitKey(1)
        if k & 0xFF == 27:
            break
    #释放资源
    cap.release()
    cv2.destroyAllWindows()

    result = args.save_path if is_save_video else "success"
    app.send({"out1":result})

if __name__ == "__main__":
    suanpan.run(app)
    # virtual_mouse("")
from attr import s
import mediapipe as mp
import cv2
import numpy as np
import time
import suanpan
from suanpan.app import app
from suanpan.app.arguments import String, Json, Int, Float, Bool
from suanpan.log import logger

from components.until.paint_function import getTool, index_raised

@app.input(String(key="inputData1", alias="msgin", default="Suanpan"))
@app.param(Int(key="param0", alias="camera"))
@app.param(Int(key="param1", alias="width"))
@app.param(Int(key="param2", alias="hight"))
@app.param(Int(key="param3", alias="max_num_hands"))
@app.param(Float(key="param4", alias="min_detection_confidence"))
@app.param(Bool(key="param5", alias="save_video"))
@app.param(String(key="param6", alias="save_path"))
@app.output(String(key="outputData1", alias="result"))

def paint(context):
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

	#模型配置
	hands = mp.solutions.hands
	hand_landmark = hands.Hands(min_detection_confidence=min_detection_confidence, max_num_hands=max_num_hands) #只检测一只手
	draw = mp.solutions.drawing_utils

	# drawing tools
	#contants
	ml = 250
	max_x, max_y = 250+ml, 50
	curr_tool = "select tool"
	time_init = True
	rad = 40
	var_inits = False
	thick = 4
	prevx, prevy = 0,0

	tools = cv2.imread("tools.png")
	tools = tools.astype('uint8')

	mask = np.ones((h,w))*255
	mask = mask.astype('uint8')

	while True:
		success, frm = cap.read()
		frm = cv2.flip(frm, 1)

		rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

		op = hand_landmark.process(rgb)

		if op.multi_hand_landmarks:
			for i in op.multi_hand_landmarks:
				draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
				x, y = int(i.landmark[8].x*w), int(i.landmark[8].y*h)

				if x < max_x and y < max_y and x > ml:
					if time_init:
						ctime = time.time()
						time_init = False
					ptime = time.time()

					cv2.circle(frm, (x, y), rad, (0,255,255), 2)
					rad -= 1

					if (ptime - ctime) > 0.8:
						curr_tool = getTool(x, ml)
						print("your current tool set to : ", curr_tool)
						time_init = True
						rad = 40

				else:
					time_init = True
					rad = 40

				if curr_tool == "draw":
					xi, yi = int(i.landmark[12].x*w), int(i.landmark[12].y*h)
					y9  = int(i.landmark[9].y*h)

					if index_raised(yi, y9):
						cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
						prevx, prevy = x, y

					else:
						prevx = x
						prevy = y

				elif curr_tool == "line":
					xi, yi = int(i.landmark[12].x*w), int(i.landmark[12].y*h)
					y9  = int(i.landmark[9].y*h)

					if index_raised(yi, y9):
						if not(var_inits):
							xii, yii = x, y
							var_inits = True

						cv2.line(frm, (xii, yii), (x, y), (50,152,255), thick)

					else:
						if var_inits:
							cv2.line(mask, (xii, yii), (x, y), 0, thick)
							var_inits = False

				elif curr_tool == "rectangle":
					xi, yi = int(i.landmark[12].x*w), int(i.landmark[12].y*h)
					y9  = int(i.landmark[9].y*h)

					if index_raised(yi, y9):
						if not(var_inits):
							xii, yii = x, y
							var_inits = True

						cv2.rectangle(frm, (xii, yii), (x, y), (0,255,255), thick)

					else:
						if var_inits:
							cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
							var_inits = False

				elif curr_tool == "circle":
					xi, yi = int(i.landmark[12].x*w), int(i.landmark[12].y*h)
					y9  = int(i.landmark[9].y*h)

					if index_raised(yi, y9):
						if not(var_inits):
							xii, yii = x, y
							var_inits = True

						cv2.circle(frm, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (255,255,0), thick)

					else:
						if var_inits:
							cv2.circle(mask, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (0,255,0), thick)
							var_inits = False

				elif curr_tool == "erase":
					xi, yi = int(i.landmark[12].x*w), int(i.landmark[12].y*h)
					y9  = int(i.landmark[9].y*h)

					if index_raised(yi, y9):
						cv2.circle(frm, (x, y), 30, (0,0,0), -1)
						cv2.circle(mask, (x, y), 30, 255, -1)

		op = cv2.bitwise_and(frm, frm, mask=mask)
		frm[:, :, 1] = op[:, :, 1]
		frm[:, :, 2] = op[:, :, 2]

		frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

		cv2.putText(frm, curr_tool, (270+ml,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
		#（5）显示图像,保存视频
		if is_save_video:
			out_video.write(frm)
		cv2.imshow("paint app", frm)
		k = cv2.waitKey(1)  # 每帧滞留1毫秒后消失
		if k & 0xFF == 27:  # 键盘ESC键退出程序
			break
	cv2.destroyAllWindows()
	cap.release()

if __name__ == "__main__":
	suanpan.run(app)
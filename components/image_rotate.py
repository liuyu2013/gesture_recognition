from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
import suanpan
from suanpan.app import app
from suanpan.app.arguments import String, Json, Int, Float, Bool
from suanpan.log import logger

@app.input(String(key="inputData1", alias="msgin", default="Suanpan"))
@app.param(Int(key="param0", alias="image_path"))
@app.param(Float(key="param1", alias="save_path"))
@app.param(Int(key="param2", alias="angle"))
@app.output(String(key="outputData1", alias="result"))
def image_rotate(context):
    args = context.args
    # 绘图/标注方法、人脸标注网格方法
    tp=Image.open(args.image_path)
    tp.rotate(args.angle,expand=True).save(args.save_path)#旋转45度
    cv2.waitKey()  # 按下任何键盘按键后
if __name__ == "__main__":
    suanpan.run(app)


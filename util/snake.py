#手势识别
import cv2
import cvzone
from matplotlib.cbook import pts_to_midstep
import numpy as np
import math
import random
import mediapipe as mp
 
# 构造一个贪吃蛇移动的类
class SnakeGameClass:
 
    #（一）初始化
    def __init__(self):
 
        self.score = 0  # 积分器
        self.points = []  # 蛇的身体的节点坐标
        self.lengths = []  # 蛇身各个节点之间的坐标
        self.currentLength = 0  # 当前蛇身长度
        self.allowedLength = 150  # 没吃东西时，蛇的总长度
        self.previousHead = (0,0)  # 前一个蛇头节点的坐标
 
        self.foodPoint = (0,0)  # 食物的起始位置
        self.randomFoodLocation()  # 随机改变食物的位置
        self.gameover = False  # 蛇头撞到蛇身，变成True，游戏结束
 
    #（二）食物随机出现的位置
    def randomFoodLocation(self):
        # x在100至1000之间，y在100至600之间，随机取一个整数
        self.foodPoint = random.randint(100, 1000),  random.randint(100, 600)
 
    #（三）更新增加蛇身长度
    def update(self, imgMain, currentHead): # 输入图像，当前蛇头的坐标
 
        # 游戏结束，显示文本
        if self.gameover:
            cvzone.putTextRect(imgMain, 'GameOver', [400,300], 5, 3, colorR=(0,255,255), colorT=(0,0,255))
            cvzone.putTextRect(imgMain, f'Score:{self.score}', [450,400], 5, 3, colorR=(255,255,0))
            cvzone.putTextRect(imgMain, f"Press Key 'R' to Restart", [230,500], 4, 3, colorR=(0,255,0), colorT=(255,0,0))
            cvzone.putTextRect(imgMain, f"Press Key 'esc' to exit the game", [130,600], 4, 3, colorR=(0,255,0), colorT=(255,0,0))
        else:
            px, py = self.previousHead  # 获得前一个蛇头的x和y坐标
            cx, cy = currentHead  # 当前蛇头节点的x和y坐标
            
            # 添加当前蛇头的坐标到蛇身节点坐标列表中
            self.points.append([cx,cy])
 
            # 计算两个节点之间的距离
            distance = math.hypot(cx-px, cy-py)  # 计算平方和开根
            # 将节点之间的距离添加到蛇身节点距离列表中
            self.lengths.append(distance)
            # 增加当前蛇身长度
            self.currentLength += distance
 
            # 更新蛇头坐标
            self.previousHead = (cx,cy)
 
            #（四）减少蛇尾长度，即移动过程中蛇头到蛇尾的长度不大于150
            if self.currentLength > self.allowedLength:
 
                # 遍历所有的节点线段长度。新更新的蛇头索引在列表后面，蛇尾的索引在列表前面
                for i, length in enumerate(self.lengths):
 
                    # 从蛇尾到蛇头依次减线段长度，得到的长度是否满足要求
                    self.currentLength -= length
 
                    # 从列表中删除蛇尾端的线段长度，以及蛇尾节点
                    self.lengths.pop(i)
                    self.points.pop(i)
 
                    # 如果当前蛇身长度小于规定长度，满足要求，退出循环
                    if self.currentLength < self.allowedLength:
                        break
            
            #（五）绘制得分板
            cvzone.putTextRect(imgMain, f'Score:{self.score}', [50,80], 4, 3, colorR=(255,255,0))
 
            #（六）检查蛇是否吃了食物
            rx, ry = self.foodPoint  # 得到食物的中心点坐标位置
            
            # 绘制矩形作为蛇的食物
            cv2.rectangle(imgMain, (rx-20, ry-20), (rx+20, ry+20), (255,255,0), cv2.FILLED)
            cv2.rectangle(imgMain, (rx-20, ry-20), (rx+20, ry+20), (0,255,255), 5)        
            cv2.rectangle(imgMain, (rx-5, ry-5), (rx+5, ry+5), (0,0,255), cv2.FILLED)  
 
            # 检查指尖(即蛇头cx,cy)是否在矩形内部
            if rx-20 < cx < rx+20 and ry-20< cy < ry+20:
 
                # 随机更换食物的位置
                self.randomFoodLocation()
 
                # 增加蛇身的限制长度，每吃1个食物就能变长50
                self.allowedLength += 50
 
                # 吃食物的计数加一
                self.score += 1
 
                print('eat!', f'score:{self.score}')
 
            #（七）绘制蛇
            # 当节点列表中有值了，才能绘制
            if self.points:
 
                # 遍历蛇身节点坐标
                for i, point in enumerate(self.points):  
                    # 绘制前后两个节点之间的连线
                    if i != 0:
                        cv2.line(imgMain, tuple(self.points[i-1]), tuple(self.points[i]), (0,255,0), 20)
                        cv2.line(imgMain, tuple(self.points[i-1]), tuple(self.points[i]), (0,0,255), 15)
 
                # 在蛇头的位置画个圆
                cv2.circle(imgMain, tuple(self.points[-1]), 20, (255,255,0), cv2.FILLED)
                cv2.circle(imgMain, tuple(self.points[-1]), 18, (255,0,0), 3)            
                cv2.circle(imgMain, tuple(self.points[-1]), 5, (0,0,0), cv2.FILLED)
 
            #（八）检查蛇头碰撞到自身        
            for point in self.points[:-2]:  # 不算蛇头到自身的距离
 
                # 计算蛇头和每个节点之间的距离
                dist = math.hypot(cx-point[0], cy-point[1])
 
                # 如果距离小于1.8，那么就证明碰撞了
                if dist < 1.8:
                    # 游戏结束
                    self.gameover = True
 
        # 返回更新后的图像
        return imgMain
 
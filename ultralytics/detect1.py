import argparse
import os
import sys

import pylab as plt
import matplotlib
import cv2
import torch
from detect import detect
import PyQt5


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.weights='/home/gibert1/yolov7/runs/train/exp10/weights/best.pt'#runs/train/exp9/weights/best.pt'#'/home/gibert1/yolov7/runs/train/exp10/weights/best.pt'#old_backup/runs-2023-0116/train/exp4/weights/best.pt'#/home/gibert1/yolov7/runs/train/exp2/weights/best.pt'#'/home/gibert1/yolov7/weights/yolov7.pt'
    opt.img_size=1920
    #opt.source = "/home/gibert1/Downloads/cat-vs-dog.jpg"
    opt.view_img=True
    opt.save_txt=False
    opt.save_conf=False
    opt.classes=None#None才會標示框線
    opt.no_trace=False
    opt.nosave=False#要儲存，才會畫框線
    opt.project='runs/detect'
    opt.name='exp3'
    opt.exist_ok=True #True才不會一直新增目錄
    opt.device='0'#'cpu'#'0'#使用GPU 0裝置 #'cpu'#
    opt.augment=False
    opt.conf_thres=0.25
    opt.iou_thres=0.45
    opt.agnostic_nms=False
    opt.update=False
    if not os.path.isfile('/home/gibert1/yolov7/V_20221031_142549_ES5.mp4'):#test_case_0129-1.mp4'):
            print("\n-------Err: no such file or sd-card not found!!\n")
            sys.exit(1)

    opt.source = '/home/gibert1/yolov7/V_20221031_142549_ES5.mp4'#test_case_0129-1.mp4'
    img = detect(myopt=opt)
    with torch.no_grad():
            img=detect(True,myopt=opt)
        #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # matplotlib.use('qt5agg') #需於偵測完，才能更改模式
        #plt.imshow(img)

    # plt.imsave("logo.png",img)
    # plt.axis("off")
    # plt.show()


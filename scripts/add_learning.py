#!/usr/bin/env python3
from nav_cloning_pytorch import *
import cv2
import csv
from skimage.transform import resize
import time
import os
import sys

class cource_following_learning_node:
    def __init__(self):
        self.dl = deep_learning(n_action=1)
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        # os.makedirs("/home/y-takahashi/catkin_ws/src/nav_cloning/data/loss/" + self.start_time)
        self.model_num = str(sys.argv[1])
        self.pro = "00_01_02"
        self.learn_no = 4000
        self.pos_no = 0
        self.data = 2775 + 1
        self.img_count = 0
        self.ang_count = 0
        self.save_path = ("/home/kazuki/takashi_ws/src/nav_cloning/data/model/"+str(self.pro)+"/"+str(self.learn_no)+"/model"+str(self.model_num)+".pt")
        self.ang_path = ("/home/kazuki/takashi_ws/src/nav_cloning/data/ang/"+str(self.pro)+"/")
        self.img_right_path = ("/home/kazuki/takashi_ws/src/nav_cloning/data/img/"+str(self.pro)+"/right")
        self.img_path = ("/home/kazuki/takashi_ws/src/nav_cloning/data/img/"+str(self.pro)+"/center")
        self.img_left_path = ("/home/kazuki/takashi_ws/src/nav_cloning/data/img/"+str(self.pro)+"/left")
        os.makedirs("/home/kazuki/takashi_ws/src/nav_cloning/data/model/"+str(self.pro)+"/"+str(self.learn_no), exist_ok=True)
        os.makedirs("/home/kazuki/takashi_ws/src/nav_cloning/data/loss/"+str(self.pro)+"/"+str(self.learn_no), exist_ok=True)
        
        # self.dl.save("/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/")

    def learn(self):
        ang_list = []
        img_right_list = []
        img_list = []
        img_left_list = []
        for i in range(self.data):
            if i % 5 == 1 or i % 5 == 4:
                pass
            else:
                for j in ["-5", "0", "+5"]:
                    img_right = cv2.imread(self.img_right_path + str(i) + "_" + j + ".jpg")
                    img = cv2.imread(self.img_path + str(i) + "_" + j + ".jpg")
                    img_left = cv2.imread(self.img_left_path + str(i) + "_" + j + ".jpg")
                    img_right_list.append(img_right)
                    img_list.append(img)
                    img_left_list.append(img_left)
                    self.img_count += 1
        print("img", self.img_count)

        with open(self.ang_path + 'ang.csv', 'r') as f:
            for row in csv.reader(f):
                no, tar_ang = row
                if float(no) % 5 == 1 or float(no) % 5 == 4:
                    pass
                else:
                    ang_list.append(float(tar_ang))
                    self.ang_count += 1
        print("ang", self.ang_count)
        
        for k in range(self.img_count):
            img_right = img_right_list[k]
            img = img_list[k]
            img_left = img_left_list[k]
            target_ang = ang_list[k]

            img_right = resize(img_right, (48, 64), mode='constant')
            r, g, b = cv2.split(img_right)
            imgobj_right = np.asanyarray([r, g, b])

            img = resize(img, (48, 64), mode='constant')
            r, g, b = cv2.split(img)
            imgobj = np.asanyarray([r, g, b])

            img_left = resize(img_left, (48, 64), mode='constant')
            r, g, b = cv2.split(img_left)
            imgobj_left = np.asanyarray([r, g, b])

            # self.dl.make_dataset(img_right, target_ang + 0.2)
            dataset = self.dl.make_dataset(img, target_ang)
            # self.dl.make_dataset(img_left, target_ang - 0.2)
            print("dataset", k)
        print("batch_size", self.img_count)

        for l in range(self.learn_no):
            loss = self.dl.trains(self.img_count)
            print("train", l)
            with open("/home/kazuki/takashi_ws/src/nav_cloning/data/loss/"+str(self.pro)+"/"+str(self.learn_no)+"/"+str(self.model_num)+".csv", 'a') as fw:
                writer = csv.writer(fw, lineterminator='\n')
                line = [str(loss)]
                writer.writerow(line)
        self.dl.save(self.save_path)
        sys.exit()

if __name__ == '__main__':
    rg = cource_following_learning_node()
    rg.learn()

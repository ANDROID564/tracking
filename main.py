from models import *
from utils import *
import math
import datetime
import csv
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from termcolor import colored
import tensorflow as tf
import sys
from os.path import isfile, join
import matplotlib.pyplot as plt
import collections


# load weights and set defaults
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4
counter1 = 0
person = 0
car = 0
motorbike = 0
bus = 0
truck = 0
person1 = 0
car1 = 0
motorbike1 = 0
bus1 = 0
truck1 = 0
final = []
final1 = []
totalperson=0
totaltruck=0
totalcar=0
totalmotorbike=0
totalbus=0
Day_Nig=0
old_obj_id=0
old_obj_id1=0
old_obj_id2=0
old_obj_id3=0
c=0 
perso=0
pp=0
perso1=0
pp1=0
pp11=0
perso2=0
pp2=0
Total_Carbon_Footprint=0
old_obj_id_list_PP=[]
old_obj_id_list_UP=[]
old_obj_id_list_DN=[]
old_obj_id_list_DN1=[[]]
traffic_up=0
traffic_dn=0
UP=0
tttt=0
p4=[]
obj_id1=[]


# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


def plot_graph(fin_timer,ket2,identity):
  
    #exit()
    
    xs = [x[0] for x in ket2]
    ys = [x[1] for x in ket2]
  
    

    j=1
    y_dif=[]
    x_dif=[]
    angle_dif=[]
    dist=[]
    y_dif1=[]
    x_dif1=[]
    dist2=[]
    dist3=0
    timer_f=0
    #print('ttttt')
    for i in range(len(xs)):
        if(j<len(xs)):
            #print('hhhhhhh')
            y=ys[j]-ys[i]
            y_dif.append(abs(y))
            x=xs[j]-xs[i]
            x_dif.append(abs(x))
        
            dist_y=y**2
            dist_x=x**2
            dist1=abs(dist_y+dist_x)
            dist2=math.sqrt(dist1)
            #dist.append(dist2)
            dist3+=dist2
            timer_f=fin_timer[i]+timer_f
            j=j+1
    speed=dist2/timer_f
    print('speed meter/second for id',identity,'is',speed*10)


videopath = 'input.mp4'

threshold=30

import cv2
from sort import *
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort() 

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800,600))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame=vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
#print ("Video size", vw,vh)
outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,20.0,(vw,vh))
#outvideo = cv2.VideoWriter(videopath.replace(".avi", "-det.avi"),fourcc,20.0,(vw,vh))



frames = 0

s_time=time.time()
q1=1
timer=[]
while(True):
    
    ret, frame = vid.read()
    if not ret:
        break
    frames += 1

    frames1=frames
    list1=[]
    
    #print(frames)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    # Bounding Box Detection=================================================================================    
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu().numpy())

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            cv2.rectangle(frame, (x1, y1-20), (x1+len(cls)*19, y1), color, -1)
            cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

            k1=int(((x1)+(x1+box_w))/2)
            k2=int(((y1)+(y1+box_h))/2)
            p1=(k1,k2)

            #if cls == 'person' :#and (int(obj_id) not in old_obj_id_list_PP) :
            #    person += 1
          
            if cls == 'car' :

         
                p4.append(p1)

            
                #print("the centroid is p3,obj_id,class",p1,obj_id,cls)
                prof = {}
                prof[obj_id] = p1
                list1.append(prof)

                #print("list1",list1) 
                
                car += 1
                #print("********* The next frame calculation starts ********")
                #print("old_obj_id_list_DN1",old_obj_id_list_DN1) 
        #print(prof)
        #if frames==15*q1:

        q1+=1

    cv2.imshow('Stream', frame)
    outvideo.write(frame)
   #print("########here the frame got processed##################",frames)
    old_obj_id_list_DN1.append(list1)  
    e_time=time.time()
    d_time=e_time-s_time
    timer.append(d_time)
    #print('timer',timer)
    s_time=time.time()           
    
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break
#print("old_obj_id_list_DN1",old_obj_id_list_DN1)

#exit()

old=old_obj_id_list_DN1
i2=[]
i31=[[]]
fin_timer=[]
len1=range(len(old)-1)
len4=range(len(old))
#print(len(old),len(timer))
#exit()
for i in len4:	
    len3=range(len(old[i]))
    #print('len3=range(len(old))',len3)
    for m in len3:
        j1=i+1
        #print('j1',j1)
        if(j1<=len(old)-1):
            for j in len4[j1:]:
                #print('len(old[j])',len(old[j]))
                len2=range(len(old[j]))
                len5=range(len(old[j]))
				
                for i12 in len5:
                #print('i12',i12)
                    for k in (len2):

                        if list(old[i][m].keys())==list(old[j][k].keys()):
                            #print("########")
                            i2.append((old[j][k]))
                            #i2.append(old[i][m])
                            #print ('old[i][m]',old[i][m])
                            old[j].remove(old[j][k])
                            #print('old[j]in end',old[j])
                            len5=range(len(old[j]))

                            len2=range(len(old[j]))
                            fin_timer.append(timer[i])

                            #print('len5',len5)
                            #fin_timer.append(timer[i])
                            #t1=n_time[i]
                            #fin_timer.append(t1)
                            break


#print('i2',i2)
#print('fin_timer',fin_timer)
year=dict()
year1=[]
leni2=len(i2)
year2=[]
#print(leni2)
ket2=[]

for line1 in range(leni2):
    #print(line.keys())
    if i2[line1].keys()  not in year2:
        identity=list(i2[line1].keys())
        #print('identity',identity)
        year2.append(i2[line1].keys())
        #print(year2) 
        for line2 in range(leni2):
            
            #print('i2[line1].keys()',i2[line1].keys())    
            if i2[line1].keys() == i2[line2].keys():
                #print('i2[line2].keys()',i2[line2].keys())
                #print('line',i2[line2].keys())
                year1.append(list(i2[line2].values()))
                #del i2[line2].keys()
                #i2.remove(i2[line2])
                

        
        
        #print('the values of year1 are',(year1))      
        for d2 in range(len(year1)):
            #print(year1[d2][0])
            ket2.append(year1[d2][0])

        #print('id and ket2',identity,ket2)
        if len(ket2)>1:
            plot_graph(fin_timer,ket2,identity)

            myFile = open('activity.csv', 'w')  
		    
        else:
            print('run the video more .....')
        del ket2[:] 
        #del ket1[:]
        del year1[:]



cv2.destroyAllWindows()
outvideo.release()
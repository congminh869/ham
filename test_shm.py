import sysv_ipc
import cv2
import numpy as np
import hashlib
import json
import time
import shutil
import redis
import time
import subprocess
import threading
from shm.SHMReader_3 import Channel, DataImage, SHMReader
import multiprocessing
from multiprocessing import Process, Pipe, Manager, Condition, Event

       
KEY_SHM = 100
SIZE_SHM = 10
WID_IMG = 1920
HEI_IMG = 1080
DEP_IMG = 3      

def ThreadShow(pipe):
    channel_shm = Channel(KEY_SHM)
    channel_shm.size_shm = SIZE_SHM
    channel_shm.width_img = WID_IMG
    channel_shm.height_img = HEI_IMG
    channel_shm.depth_img = DEP_IMG
    
    shm = SHMReader(channel_shm)
    list_video  = []
    time_1 =time.time()
    count = 0
    while True:
        ret, arr = shm.Read()
        if ret > 0:
            for f in arr:
                if len(f) > 0:
                    count+=1;
                    resized = cv2.resize(f[0].image, (640,360), interpolation = cv2.INTER_AREA)
                    str_show = "Test Show_" + str(f[0].cid);
                    cv2.imshow(str_show, resized)
                    cv2.waitKey(1)
        if time.time() - time_1 >=1:
            print("Frame per 1 second : ", count)
            time_1 =time.time()
            count=0
        
#Main
if __name__ == "__main__":
    parent_conn, child_conn = Pipe()

    thread1 = Process(target=ThreadShow,args=(parent_conn,)) 
    thread1.daemon = False
    thread1.start()
    
    # thread2 = Process(target=ThreadShow,args=(parent_conn,)) 
    # thread2.daemon = False
    # thread2.start()



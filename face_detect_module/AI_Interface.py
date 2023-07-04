import cv2
import sys 
import numpy
import os
import time
from datetime import datetime
import yaml
from pathlib import Path
import cv2
import FaceDetect
import numpy
import time

class DtectBox:
    def __init__(self):
        self.bbox = None # [x1, y1, x2, y2]
        self.name_class = None # string
        self.id_tracking = None #int
        self.class_conf = None  # float
        '''list: 0:chuadung   
        1:mu_bh_trang 
        2:mu_bh_vang  
        3:mu_thuong   
        4:ao_dacam    
        5:ao_trang    
        6:ao_khac 
        7:quan_dacam  
        8:quan_khac   
        9:balo'''
        self.class_ids = None #for detect uniform  list
        self.check_class_ids = None

        
class DataTracking:
    def __init__(self, frame, dtectBoxs, cid, type_id, count, frame_src=None):
        #DataTracking(frame, dtectBoxs, cid, type_id, count)
        self.frame_src = frame_src
        self.frame = frame
        self.dtectBoxs = dtectBoxs # list

        self.cid = cid #int
        self.type_id = type_id# int
        self.count = count # int


class AIInterface: 
    config_file = ""
    name = ""
    type_ai = ""
    ai_id = -1
    
    size_shm = 10
    width_img = 1920
    height_img = 1080 
    depth_img = 3
     
    def __init__(self): 
        self.config_file = ""
        self.name = ""
        self.type_ai = ""
        self.ai_id = -1

    def Init(self):
        print(self.name , "Init")
        
    def Detect(self, img, ares):
        ret = {"warning": True}
        print(self.name , "Detect")
        return ret

class FaceData:
    def __init__(self): 
        self.face_id = -1
        self.time_begin = -1
        self.time_update = -1
        self.landmarks = []
        self.face_img = None

class AI_FaceDetect(AIInterface):
    def __init__(self): 
        self.max_camera = -1
        self.face_detecter = None

        #List face result as dict:
        #   With element as:
        #       Key: <camera id>
        #       Value: [<FaceData object> , ..]
        self.list_face_ret = {}

    def Init(self):
        self.face_detecter = FaceDetect.FaceDetectRetinaTRT(os.path.abspath(os.getcwd())+'/config/config_FaceRetina.txt')

    def PushToResult(self, ret):
        if ret.camera_id in self.list_face_ret:
            f_existed = False
            for i in range(len(self.list_face_ret[ret.camera_id])):
                if self.list_face_ret[ret.camera_id][i].face_id == ret.id_tracking:
                    f_existed = True
                    self.list_face_ret[ret.camera_id][i].time_update = time.time()
                    self.list_face_ret[ret.camera_id][i].face_image = numpy.asarray(ret.img_face)
                    self.list_face_ret[ret.camera_id][i].landmarks = []
                    for point in ret.landmark_points:
                        self.list_face_ret[ret.camera_id][i].landmarks.append([point.x, point.y])
                    break
            if f_existed is False:
                face_data = FaceData()
                face_data.face_id = ret.id_tracking
                face_data.time_begin = time.time()
                face_data.time_update = time.time()
                face_data.face_image = numpy.asarray(ret.img_face)
                face_data.landmarks = []
                for point in ret.landmark_points:
                    face_data.landmarks.append([point.x, point.y])
                self.list_face_ret[ret.camera_id].append(face_data)
        else:
            face_data = FaceData()
            face_data.face_id = ret.id_tracking
            face_data.time_begin = time.time()
            face_data.time_update = time.time()
            face_data.face_image = numpy.asarray(ret.img_face)
            face_data.landmarks = []
            for point in ret.landmark_points:
                face_data.landmarks.append([point.x, point.y])
            self.list_face_ret[ret.camera_id] = []
            self.list_face_ret[ret.camera_id].append(face_data)
    
    def Detect(self, dataImages, coordinate_rois, labels_allow_helmet = None,  labels_allow_uniform = None):
        imgs_out = []

        for dataImage in dataImages:
            np_image_data = numpy.asarray(dataImage.image)
            m = FaceDetect.Mat3b.from_array(np_image_data)
            ret = self.face_detecter.Detect(dataImage.cid, m)

            #Draw img
            for i in ret:
                dataImage.image = cv2.rectangle(dataImage.image, (int(i.bbox.x), int(i.bbox.y)), (int(i.bbox.width + i.bbox.x), int(i.bbox.height + i.bbox.y)), (255, 0, 0), 2)
                imgs_out.append(dataImage)
                self.PushToResult(i)

        return None, imgs_out

    def GetResult(self):
        ret = []
        list_keys = list(self.list_face_ret.keys())

        #Get result
        for key in list_keys:
            i = 0
            while True:
                if i >= len(self.list_face_ret[key]):
                    break
                #If timeout update new data
                if time.time() - self.list_face_ret[key][i].time_update > 10:
                    ret.append(face_data)
                    self.list_face_ret[key].pop(i)
                    i = i - 1
                #If face is exist long -> reget face 
                elif self.list_face_ret[key][i].time_update - self.list_face_ret[key][i].time_begin > 10:
                    ret.append(face_data)
                    self.list_face_ret[key][i].time_begin = self.list_face_ret[key][i].time_update
                i+=1

                #Remove camera have data is none
                if len(self.list_face_ret[key]) == 0:
                    self.list_face_ret.pop(key)
        return ret



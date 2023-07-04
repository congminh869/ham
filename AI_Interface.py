import cv2
import sys 
import numpy
import numpy as np
import os
import time
from datetime import datetime
import yaml
from pathlib import Path

from include.main import main_TBA, main_tunnel, main_fence, main_high_volt_switch, \
                        main_belt, main_clock, load_sort, main_personHoldThingDetect,\
                         main_test, CountPerson
#region Phat hien vat the cao
from include.personHoldThingDetect import personHoldThing_All
from include.yolo import YOLOv562
from include.model_paddle_class import paddleClas, infer_paddleClass
from include.CheckSort import NCheckViolateID
from include.personbelt import PersonBelt
from include.event import ConvertEvent, ConvertEventFence
from include.personHoldThing import BGS_HB
from include.personFalse import PersonFalse
import face_detect_module.FaceDetect as FaceDetect
import dlib
from face_utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image, extract_lankmarks5

#debug log
from inspect import currentframe, getframeinfo
import datetime

import cv2
# import face_detect_module.FaceDetect as FaceDetect
import numpy
import time
import copy

# from helmet_processes import to_batches, helmet_scaling # import processes helmets
# from CheckViolate import CheckViolate

debug_log = False

def Debug_log(cf, filename, name = None):
    if debug_log:
        #Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        ct = datetime.datetime.now()
        print(f'================ [{ct}] file {filename} , line : {cf.f_lineno} {name}')

from License_Plate.include.onebatch import License_Plate





GS_HangRao_Nguoi = 1
GS_HangRao_DamChay = 2
GS_KhuVuc_ThiCong = 3
GS_KhuVuc_HanChe = 4
GS_ThietBi_DongHo = 5
GS_ThietBi_DaoCachLy = 6
GS_DoBaoHo_DaiAnToan = 7
GS_NhietDo_2Diem = 8
GS_NhietDo_1Vung = 9
ND_BienSo = 10
ND_KhuonMat = 11

def show(list_dataTrackings):
    cid = 0
    imgs = []
    for idx, dataTrackings in enumerate(list_dataTrackings):
        for i in range(len(dataTrackings)):
            if dataTrackings[i].cid == cid:
                imgs.append(cv2.resize(dataTrackings[i].frame, (450,350), interpolation = cv2.INTER_AREA))
    
    # if mode==0:
    zero_img = np.zeros((350,450,3), np.uint8)

    im_h1 = cv2.hconcat([imgs[0], imgs[1], imgs[2]])
    if len(imgs)==6:
        im_h2 = cv2.hconcat([imgs[3], imgs[4], imgs[5]])
    else:
        im_h2 = cv2.hconcat([imgs[3], imgs[4], zero_img])
    img = cv2.vconcat([im_h1, im_h2])

    cv2.imshow('TBA', img)
    if cv2.waitKey(1) == 27:
        sys.exit()
    # video_write.write(img)
    # if mode==1:
    #     zero_img = np.zeros((350,450,3), np.uint8)
    #     im_h1 = cv2.hconcat([imgs[0], imgs[1], zero_img])
    #     im_h2 = cv2.hconcat([imgs[2], imgs[3], zero_img])
    #     img = cv2.vconcat([im_h1, im_h2])

    #     cv2.imshow('turnel', img)
    #     cv2.waitKey(1)
    #     video_write.write(img)

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
        with open(os.path.abspath(os.getcwd()) +'/config/config_main_include.yaml', 'r') as file:
            self.configuration = yaml.safe_load(file)
        
    def Init(self):
        print(self.name , "Init")
        
    def Detect(self, img, ares):
        ret = {"warning": True}
        img_out = None
        print(self.name , "Detect")
        return ret, img_out

class AI_TBA(AIInterface):     
    def __init__(self): 
        super().__init__()
        self.config_file = ""
        self.name = ""
        self.type_ai = ""
        self.ai_id = -1
        self.list_sort = {}
        self.max_camera = -1
        
        self.time_kill_sort = 600
        self.time_kill_id_seconds=600
        self.time_check_seconds=180

        with open(os.path.abspath(os.getcwd())+'/config/config_TBA.yaml', 'r') as file:
            self.configuration = yaml.safe_load(file)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        
    def Init(self):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        #print("load TBA".center(150,"*"))
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        print('LOAD MODEL PERSON')
        weights_person = path = str(Path(os.path.abspath(os.getcwd())).parent.absolute()) + "/" + self.configuration['weights_person']['weights'] #'./weight/crowdhuman_yolov5m.pt'
        classes=self.configuration['weights_person']['classes']
        device=self.configuration['weights_person']['device']
        iou_thres = self.configuration['weights_person']['iou_thres']
        conf_thres = self.configuration['weights_person']['conf_thres']
        img_size = self.configuration['weights_person']['img_size']
        max_det=self.configuration['weights_person']['max_det']
        agnostic_nms=self.configuration['weights_person']['agnostic_nms']
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        self.yolo_person = YOLOv562(weights_person, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)

        #load model detect hat
        print('LOAD MODEL HAT')
        weights_hat = str(Path(os.path.abspath(os.getcwd())).parent.absolute()) + "/" + self.configuration['weights_hat']['weights'] #'./weight/crowdhuman_yolov5m.pt'
        classes=self.configuration['weights_hat']['classes']
        device=self.configuration['weights_hat']['device']
        iou_thres = self.configuration['weights_hat']['iou_thres']
        conf_thres = self.configuration['weights_hat']['conf_thres']
        img_size = self.configuration['weights_hat']['img_size']
        max_det=self.configuration['weights_hat']['max_det']
        agnostic_nms=self.configuration['weights_hat']['agnostic_nms']
        self.yolo_hat = YOLOv562(weights_hat, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)
        
        #load model detect vehicle
        print('LOAD MODEL VEHICLE')
        weights_vehicle = str(Path(os.path.abspath(os.getcwd())).parent.absolute()) + "/" + self.configuration['weights_vehicle']['weights'] #'./weight/crowdhuman_yolov5m.pt'
        classes=self.configuration['weights_vehicle']['classes']
        device=self.configuration['weights_vehicle']['device']
        iou_thres = self.configuration['weights_vehicle']['iou_thres']
        conf_thres = self.configuration['weights_vehicle']['conf_thres']
        img_size = self.configuration['weights_vehicle']['img_size']
        max_det=self.configuration['weights_vehicle']['max_det']
        agnostic_nms=self.configuration['weights_vehicle']['agnostic_nms']
        self.yolo_vehicle = YOLOv562(weights_vehicle, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)

        #region khoi tao phat hien vat cao
        self.personHoldThingDetect_video_1 = personHoldThing_All()

        #check violate hat
        self.nCheckViolateID = NCheckViolateID()

        self.dict_video = {}
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        self.convertEvent = ConvertEvent(self.time_kill_id_seconds, self.time_check_seconds)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)

    def save_video(self, name_show):
        video = cv2.VideoWriter(os.path.abspath(os.getcwd())+'/output/'+name_show+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, (1350, 700))
        self.dict_video[name_show] = video
        
    def check_image(self, dataImages):
        print('len(dataImages) : ', len(dataImages))
        for idx, dataImage in enumerate(dataImages): 
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, idx)
            cid = dataImage.cid
            type_id = dataImage.type_id
            count = dataImage.count
            image = dataImage.image
            cv2.imwrite(f'./output/{str(idx)}_yolo.jpg', image)
            print(image)
            # dtectBoxs_person = []
            # dataTrackings.append(DataTracking(image, dtectBoxs_person, cid, type_id, count))

    def Detect(self, dataImages, coordinate_rois, labels_allow_helmet = None,  labels_allow_uniform = None):
        img_out = {}
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        #if coordinate_rois none => [[100, 100], [w - 100, h - 100]]
        # print('+++++++++++++++++++++++++++check image')
        #self.check_image(dataImages)
        # print('check image ', len(dataImages))
        for idx, dataImage in enumerate(dataImages):
            cid = dataImage.cid
            if cid not in self.list_sort:
                self.list_sort[cid] = load_sort(cid)
                self.save_video(str(cid))
        
        #check sort exit
        cid_dels = []
        # print('self.list_sort first :', self.list_sort)
        for cid in self.list_sort:
            time_check = time.time() - self.list_sort[cid].time_check
            # print(f'cid : {str(cid)}, time_check : {time_check}')
            if time_check >= self.time_kill_sort:
                # print('del sort 1')
                cid_dels.append(cid)
        for cid_del in cid_dels:
            # print('del sort 2')
            del self.list_sort[cid_del]

        # print('self.list_sort second :', self.list_sort)
        dict_data, img_out = main_TBA(dataImages, 
                                    coordinate_rois, 
                                    self.yolo_person, 
                                    self.yolo_vehicle, 
                                    self.yolo_hat, 
                                    self.personHoldThingDetect_video_1, 
                                    labels_allow_helmet, 
                                    self.list_sort, 
                                    self.nCheckViolateID,
                                    self.convertEvent)

        return dict_data, img_out

class AI_TUNNEL(AIInterface):     
    def __init__(self): 
        super().__init__()
        self.config_file = ""
        self.name = ""
        self.type_ai = ""
        self.ai_id = -1
        self.list_sort = {}
        self.max_camera = -1
        with open(os.path.abspath(os.getcwd()) +'/config/config_tunnel.yaml', 'r') as file:
            self.configuration = yaml.safe_load(file)
        
    def Init(self):
        #load model detect person
        print('LOAD MODEL PERSON')
        weights_person = str(Path(os.path.abspath(os.getcwd())).parent.absolute()) + "/" + self.configuration['weights_person']['weights'] #'./weight/crowdhuman_yolov5m.pt'
        classes=self.configuration['weights_person']['classes']
        device=self.configuration['weights_person']['device']
        iou_thres = self.configuration['weights_person']['iou_thres']
        conf_thres = self.configuration['weights_person']['conf_thres']
        img_size = self.configuration['weights_person']['img_size']
        max_det=self.configuration['weights_person']['max_det']
        agnostic_nms=self.configuration['weights_person']['agnostic_nms']
        engine=self.configuration['weights_person']['engine']
        self.yolo_person = YOLOv562(weights_person, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms, engine = engine)

        #load model detect hat
        print('LOAD MODEL HAT')
        weights_hat = str(Path(os.path.abspath(os.getcwd())).parent.absolute()) + "/" + self.configuration['weights_hat']['weights'] #'./weight/crowdhuman_yolov5m.pt'
        classes=self.configuration['weights_hat']['classes']
        device=self.configuration['weights_hat']['device']
        iou_thres = self.configuration['weights_hat']['iou_thres']
        conf_thres = self.configuration['weights_hat']['conf_thres']
        img_size = self.configuration['weights_hat']['img_size']
        max_det=self.configuration['weights_hat']['max_det']
        agnostic_nms=self.configuration['weights_hat']['agnostic_nms']
        # self.yolo_hat = YOLOv562(weights_hat, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)

        #load model detect uniform
        self.engine = paddleClas()

        #check violate hat
        self.nCheckViolateID = NCheckViolateID()

        self.time_kill_sort = 600

        self.personFalse = PersonFalse()
        
        
    def Detect(self, dataImages, coordinate_rois, labels_allow_helmet = None, labels_allow_uniform = None):
        #if coordinate_rois none => [[100, 100], [w - 100, h - 100]]

        img_out = None
        for idx, dataImage in enumerate(dataImages):
            cid = dataImage.cid
            if cid not in self.list_sort:
                self.list_sort[cid] = load_sort(cid)
        
        #check sort exit
        cid_dels = []
        # print('self.list_sort first :', self.list_sort)
        for cid in self.list_sort:
            time_check = time.time() - self.list_sort[cid].time_check
            # print(f'cid : {str(cid)}, time_check : {time_check}')
            if time_check >= self.time_kill_sort:
                # print('del sort 1')
                cid_dels.append(cid)
        for cid_del in cid_dels:
            # print('del sort 2')
            del self.list_sort[cid_del]

        # print('self.list_sort second :', self.list_sort)
        dict_data, img_out = main_tunnel(dataImages, 
                                coordinate_rois,
                                labels_allow_helmet, 
                                labels_allow_uniform,
                                self.list_sort,
                                self.nCheckViolateID,
                                self.yolo_person, 
                                self.yolo_person, 
                                self.engine,
                                self.personFalse
                                )

        return dict_data, img_out


class AI_HSV(AIInterface):     
    def __init__(self): 
        super().__init__()
        self.config_file = ""
        self.name = ""
        self.type_ai = ""
        self.ai_id = -1
        self.list_sort = {}
        self.max_camera = -1
        
    def Init(self):
        #load model detect High volt switch
        print('LOAD MODEL HIGH VOLT SWITCH')
        weights_HVS = str(Path(os.path.abspath(os.getcwd())).parent.absolute()) + "/" + self.configuration['weights_HVS']['weights'] #'./weight/crowdhuman_yolov5m.pt'
        classes=self.configuration['weights_HVS']['classes']
        device=self.configuration['weights_HVS']['device']
        iou_thres = self.configuration['weights_HVS']['iou_thres']
        conf_thres = self.configuration['weights_HVS']['conf_thres']
        img_size = self.configuration['weights_HVS']['img_size']
        max_det=self.configuration['weights_HVS']['max_det']
        agnostic_nms=self.configuration['weights_HVS']['agnostic_nms']
        self.yolo_HVS = YOLOv562(weights_HVS, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)
        # main_high_volt_switch(dataImages, coordinate_rois)
        
    def Detect(self, dataImages, coordinate_rois, labels_allow_helmet = None, labels_allow_uniform=None):
        img_out = None
        dict_data = main_high_volt_switch(dataImages, 
                                          coordinate_rois, 
                                          self.yolo_HVS)
        # print(dict_data)
        return dict_data, img_out
        

class AI_BELT(AIInterface):     
    def __init__(self): 
        super().__init__()
        self.config_file = ""
        self.name = ""
        self.type_ai = ""
        self.ai_id = -1
        self.list_sort = {}
        self.num_camera = -1
        with open(os.path.abspath(os.getcwd())+'/config/config_belt.yaml', 'r') as file:
            self.configuration = yaml.safe_load(file)
        
    def Init(self):
        #load model detect person
        print('LOAD MODEL PERSON')
        weights_person = str(Path(os.path.abspath(os.getcwd())).parent.absolute()) + "/" + self.configuration['weights_person']['weights'] #'./weight/crowdhuman_yolov5m.pt'
        #print(weights_person)
        classes=self.configuration['weights_person']['classes']
        device=self.configuration['weights_person']['device']
        iou_thres = self.configuration['weights_person']['iou_thres']
        conf_thres = self.configuration['weights_person']['conf_thres']
        img_size = self.configuration['weights_person']['img_size']
        max_det=self.configuration['weights_person']['max_det']
        agnostic_nms=self.configuration['weights_person']['agnostic_nms']
        self.yolo_person = YOLOv562(weights_person, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)

        #load model detect belt
        print('LOAD MODEL BELT')
        weights_belt = str(Path(os.path.abspath(os.getcwd())).parent.absolute()) + "/" + self.configuration['weights_belt']['weights'] #'./weight/crowdhuman_yolov5m.pt'
        classes=self.configuration['weights_belt']['classes']
        device=self.configuration['weights_belt']['device']
        iou_thres = self.configuration['weights_belt']['iou_thres']
        conf_thres = self.configuration['weights_belt']['conf_thres']
        img_size = self.configuration['weights_belt']['img_size']
        max_det=self.configuration['weights_belt']['max_det']
        agnostic_nms=self.configuration['weights_belt']['agnostic_nms']
        self.yolo_belt = YOLOv562(weights_belt, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)
        print('load model main_belt done')

        self.personBelt = PersonBelt(time.time())

        self.time_kill_sort = 600
        #print("\n\nPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP\n\n")
        

        
    def Detect(self, dataImages, coordinate_rois, labels_allow_helmet = None, labels_allow_uniform=None):
        img_out = None
        for idx, dataImage in enumerate(dataImages):
            cid = dataImage.cid
            if cid not in self.list_sort:
                self.list_sort[cid] = load_sort(cid)

        cid_dels = []
        for cid in self.list_sort:
            time_check = time.time() - self.list_sort[cid].time_check
            if time_check >= self.time_kill_sort:
                cid_dels.append(cid)
        for cid_del in cid_dels:
            del self.list_sort[cid_del]
        # print('second : ', self.list_sort)

        dict_data = main_belt(dataImages,   
                    coordinate_rois, 
                    self.yolo_belt, 
                    self.yolo_person, 
                    self.list_sort,
                    self.personBelt)

        # for i in dict_data:
        #     frame = dict_data[i]['img']
            # print("cid, data", i, dict_data[i]['data'])
            # resized = cv2.resize(frame, (640,360), interpolation = cv2.INTER_AREA)
            # str_show = "person" + str(i);
            # cv2.imshow(str_show, resized)
            # if cv2.waitKey(0) == 27:
            #     sys.exit()

        return dict_data, img_out

class AI_CLOCK(AIInterface):     
    def __init__(self): 
        super().__init__()
        self.config_file = ""
        self.name = ""
        self.type_ai = ""
        self.ai_id = -1
        self.list_sort = {}
        self.max_camera = -1
        
    def Init(self):
        #load model detect High volt switch
        #load model detect clock
        print('LOAD MODEL CLOCK')
        weights_clock = str(Path(os.path.abspath(os.getcwd())).parent.absolute()) + "/" + self.configuration['weights_clock']['weights'] #'./weight/crowdhuman_yolov5m.pt'
        classes=self.configuration['weights_clock']['classes']
        device=self.configuration['weights_clock']['device']
        iou_thres = self.configuration['weights_clock']['iou_thres']
        conf_thres = self.configuration['weights_clock']['conf_thres']
        img_size = self.configuration['weights_clock']['img_size']
        max_det=self.configuration['weights_clock']['max_det']
        agnostic_nms=self.configuration['weights_clock']['agnostic_nms']
        self.yolo_clock = YOLOv562(weights_clock, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)
        # main_clock(dataImages, coordinate_rois)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, device)
        
    def Detect(self, dataImages, coordinate_rois=None, labels_allow_helmet = None, labels_allow_uniform=None):
        img_out = None
        dict_data = main_clock(dataImages,
                               self.yolo_clock)
        # print(dict_data)
        return dict_data, img_out

class AI_FENCE(AIInterface):     
    def __init__(self): 
        super().__init__()
        self.config_file = ""
        self.name = ""
        self.type_ai = ""
        self.ai_id = -1
        self.list_sort = {}
        self.num_camera = -1

        self.time_reset=5

        with open(os.path.abspath(os.getcwd()) +'/config/config_fence.yaml', 'r') as file:
            self.configuration = yaml.safe_load(file)
        
    def Init(self):
        #load model detect person
        print('LOAD MODEL PERSON')
        weights_person = str(Path(os.path.abspath(os.getcwd())).parent.absolute()) + "/" +self.configuration['weights_person']['weights'] #'./weight/crowdhuman_yolov5m.pt'
        classes=self.configuration['weights_person']['classes']
        device=self.configuration['weights_person']['device']
        iou_thres = self.configuration['weights_person']['iou_thres']
        conf_thres = self.configuration['weights_person']['conf_thres']
        img_size = self.configuration['weights_person']['img_size']
        max_det=self.configuration['weights_person']['max_det']
        agnostic_nms=self.configuration['weights_person']['agnostic_nms']
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, device)
        self.yolo_person = YOLOv562(weights_person, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)

        ##########################load model
        print('LOAD MODEL FIRE AND SMOKE')
        #load model detect fire and smoke
        weights_FS = str(Path(os.path.abspath(os.getcwd())).parent.absolute()) + "/" + self.configuration['weights_FS']['weights'] #'./weight/crowdhuman_yolov5m.pt'
        classes=self.configuration['weights_FS']['classes']
        device=self.configuration['weights_FS']['device']
        iou_thres = self.configuration['weights_FS']['iou_thres']
        conf_thres = self.configuration['weights_FS']['conf_thres']
        img_size = self.configuration['weights_FS']['img_size']
        max_det=self.configuration['weights_FS']['max_det']
        agnostic_nms=self.configuration['weights_FS']['agnostic_nms']
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, device)
        self.yolo_FS = YOLOv562(weights_FS, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)
        print('load model done')
        ####################################
        # main_fence(dataImages)

        self.convertEventFence = ConvertEventFence(self.time_reset)
        

        
    def Detect(self, dataImages, coordinate_rois=None, labels_allow_helmet=None, labels_allow_uniform=None):
        img_out = {}
        dict_data, img_out = main_fence(dataImages, 
                                self.yolo_person, 
                                self.yolo_FS, 
                                self.convertEventFence,
                                coordinate_rois,
                                check_detect_person=True)

        return dict_data, img_out

class AI_LICENSE_PLATE(AIInterface):     
    def __init__(self): 
        super().__init__()
        self.config_file = ""
        self.name = ""
        self.type_ai = ""
        self.ai_id = -1
        self.list_sort = {}
        self.max_camera = 1
        
    def Init(self):
        # import os
        # import shutil
        # if os.path.isdir('event_plate'):
        #     shutil.rmtree('event_plate')
        #     os.mkdir('./event_plate/')

        #     os.mkdir('./event_plate/data_event/')
        #     os.mkdir('./event_plate/detect/')
        #     os.mkdir('./event_plate/event/')
        #     os.mkdir('./event_plate/main/')
        #     os.mkdir('./event_plate/reg/')
        # else:
        #     os.mkdir('./event_plate/')
        #     os.mkdir('./event_plate/data_event/')
        #     os.mkdir('./event_plate/detect/')
        #     os.mkdir('./event_plate/event/')
        #     os.mkdir('./event_plate/main/')
        #     os.mkdir('./event_plate/reg/')
        print('LOAD MODEL PERSON')
        self.license_Plate = License_Plate()
        
    def Detect(self, dataImages, coordinate_rois, labels_allow_helmet = None, labels_allow_uniform=None):
        #if coordinate_rois none => [[100, 100], [w - 100, h - 100]]
        # print('self.list_sort second :', self.list_sort)
        '''
            dataImages: len(dataImages) = 1
            dict_data = {"cid": {"img": image, "data": data},
            data = [{id: id, “lpNumber”: “84A89563” }, {id: id, “lpNumber”: “84A89563” }, {id: id, “lpNumber”: “84A89563” }...]
        '''
        img_out = None
        dict_data = {}
        for dataImg in dataImages:
            imgs_in = [dataImg]
            dict_data = self.license_Plate.detect(imgs_in, coordinate_rois, dict_data)                

        return dict_data, img_out

class AI_LP_LIVE(AIInterface):     
    def __init__(self): 
        super().__init__()
        self.config_file = ""
        self.name = ""
        self.type_ai = ""
        self.ai_id = -1
        self.list_sort = {}
        self.max_camera = 1
        
    def Init(self):
        print('LOAD MODEL PERSON')
        self.license_Plate = License_Plate()
        
    def Detect(self, dataImages, coordinate_rois, labels_allow_helmet = None, labels_allow_uniform=None):
        #if coordinate_rois none => [[100, 100], [w - 100, h - 100]]
        # print('self.list_sort second :', self.list_sort)
        '''
            dataImages: len(dataImages) = 1
            dict_data = {"cid": {"img": image, "data": data},
            data = [{id: id, “lpNumber”: “84A89563” }, {id: id, “lpNumber”: “84A89563” }, {id: id, “lpNumber”: “84A89563” }...]
        '''
        dict_data = {}
        for dataImg in dataImages:
            imgs_in = [dataImg]
            dict_data = self.license_Plate.detectLiveStr(imgs_in, coordinate_rois, dict_data)                

        return dict_data

class AI_PersonHoldThingDetect(AIInterface):     
    def __init__(self): 
        super().__init__()
        self.config_file = ""
        self.name = ""
        self.type_ai = ""
        self.ai_id = -1
        self.list_sort = {}
        self.max_camera = -1
        
        self.time_kill_sort = 600
        self.time_kill_id_seconds=600
        self.time_check_seconds=180
        self.coordinate_roi_border = 100

        with open(os.path.abspath(os.getcwd())+'/config/config_TBA.yaml', 'r') as file:
            self.configuration = yaml.safe_load(file)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        
    def Init(self):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        print('LOAD MODEL PERSON')
        weights_person = str(Path(os.path.abspath(os.getcwd())).parent.absolute()) + "/" + self.configuration['weights_person']['weights'] #'./weight/crowdhuman_yolov5m.pt'
        classes=self.configuration['weights_person']['classes']
        device=self.configuration['weights_person']['device']
        iou_thres = self.configuration['weights_person']['iou_thres']
        conf_thres = self.configuration['weights_person']['conf_thres']
        img_size = self.configuration['weights_person']['img_size']
        max_det=self.configuration['weights_person']['max_det']
        agnostic_nms=self.configuration['weights_person']['agnostic_nms']
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        self.yolo_person = YOLOv562(weights_person, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)

        #person Hold Thing detect
        self.bgshb = BGS_HB()


        self.convertEvent = ConvertEvent(self.time_kill_id_seconds, self.time_check_seconds)
        
    def Detect(self, dataImages, coordinate_rois, labels_allow_helmet = None,  labels_allow_uniform = None):
        #if coordinate_rois none => [[100, 100], [w - 100, h - 100]]

        img_out = None
        for idx, dataImage in enumerate(dataImages):
            cid = dataImage.cid
            if cid not in self.list_sort:
                self.list_sort[cid] = load_sort(cid)
        
        #check sort exit
        cid_dels = []
        # print('self.list_sort first :', self.list_sort)
        for cid in self.list_sort:
            time_check = time.time() - self.list_sort[cid].time_check
            # print(f'cid : {str(cid)}, time_check : {time_check}')
            if time_check >= self.time_kill_sort:
                # print('del sort 1')
                cid_dels.append(cid)
        for cid_del in cid_dels:
            # print('del sort 2')
            del self.list_sort[cid_del]

        # print('self.list_sort second :', self.list_sort)
        dict_data = main_personHoldThingDetect(dataImages, 
                                    coordinate_rois, 
                                    self.yolo_person, 
                                    self.list_sort, 
                                    self.convertEvent,
                                    self.coordinate_roi_border,
                                    self.bgshb)

        return dict_data, img_out   

class AI_COUNTPERSON(AIInterface):     
    def __init__(self): 
        super().__init__()
        self.config_file = ""
        self.name = ""
        self.type_ai = ""
        self.ai_id = -1
        self.list_sort = {}
        self.max_camera = -1
        with open(os.path.abspath(os.getcwd()) +'/config/config_tunnel.yaml', 'r') as file:
            self.configuration = yaml.safe_load(file)

        
        
    def Init(self):
        #load model detect person
        print('LOAD MODEL PERSON')
        weights_person = str(Path(os.path.abspath(os.getcwd())).parent.absolute()) + "/" + self.configuration['weights_person']['weights'] #'./weight/crowdhuman_yolov5m.pt'
        classes=self.configuration['weights_person']['classes']
        device=self.configuration['weights_person']['device']
        iou_thres = self.configuration['weights_person']['iou_thres']
        conf_thres = self.configuration['weights_person']['conf_thres']
        img_size = self.configuration['weights_person']['img_size']
        max_det=self.configuration['weights_person']['max_det']
        agnostic_nms=self.configuration['weights_person']['agnostic_nms']
        engine=self.configuration['weights_person']['engine']
        self.yolo_person = YOLOv562(weights_person, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms, engine = engine)

        self.time_kill_sort = 600

        # self.personFalse = PersonFalse()

        self.checkIDIn = {} # check id, if 40 not appear then remove {id: time}
        self.checkIDOut = {}

        self.person_in = {} # count person in {id: count}
        self.person_out = {}
        
        
    def Detect(self, dataImages, coordinates, labels_allow_helmet = None, labels_allow_uniform = None):
        #if coordinate_rois none => [[100, 100], [w - 100, h - 100]]

        coordinate_rois, coodiPersonIns, coodiPersonOuts = coordinates
        for idx, dataImage in enumerate(dataImages):
            cid = dataImage.cid
            if cid not in self.list_sort:
                self.list_sort[cid] = load_sort(cid)
        
        #check sort exit
        cid_dels = []
        # print('self.list_sort first :', self.list_sort)
        for cid in self.list_sort:
            time_check = time.time() - self.list_sort[cid].time_check
            # print(f'cid : {str(cid)}, time_check : {time_check}')
            if time_check >= self.time_kill_sort:
                # print('del sort 1')
                cid_dels.append(cid)
        for cid_del in cid_dels:
            # print('del sort 2')
            del self.list_sort[cid_del]

        # print('self.list_sort second :', self.list_sort)
        print('1 self.person_in, self.person_out : ', self.person_in, self.person_out)
        dict_data, self.person_in, self.person_out = CountPerson(dataImages, 
                              coordinate_rois,
                              coodiPersonIns,
                              coodiPersonOuts,
                              self.list_sort,
                              self.yolo_person,
                              self.person_in,
                              self.person_out,
                              self.checkIDIn,
                              self.checkIDOut)
        print('2 self.person_in, self.person_out : ', self.person_in, self.person_out)
        print('self.checkIDIn : ', self.checkIDIn)
        print('self.checkIDOut : ', self.checkIDOut)
        return dict_data, self.person_in , self.person_out

class AI_TESTSORT(AIInterface):     
    def __init__(self): 
        super().__init__()
        self.config_file = ""
        self.name = ""
        self.type_ai = ""
        self.ai_id = -1
        self.list_sort = {}
        self.max_camera = -1
        with open(os.path.abspath(os.getcwd()) +'/config/config_tunnel.yaml', 'r') as file:
            self.configuration = yaml.safe_load(file)
        
    def Init(self):
        #load model detect person
        print('LOAD MODEL PERSON')
        weights_person = str(Path(os.path.abspath(os.getcwd())).parent.absolute()) + "/" + self.configuration['weights_person']['weights'] #'./weight/crowdhuman_yolov5m.pt'
        classes=self.configuration['weights_person']['classes']
        device=self.configuration['weights_person']['device']
        iou_thres = self.configuration['weights_person']['iou_thres']
        conf_thres = self.configuration['weights_person']['conf_thres']
        img_size = self.configuration['weights_person']['img_size']
        max_det=self.configuration['weights_person']['max_det']
        agnostic_nms=self.configuration['weights_person']['agnostic_nms']
        engine=self.configuration['weights_person']['engine']
        self.yolo_person = YOLOv562(weights_person, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms, engine = engine)

        self.time_kill_sort = 600

        self.personFalse = PersonFalse()
        
        
    def Detect(self, dataImages, coordinate_rois, labels_allow_helmet = None, labels_allow_uniform = None):
        #if coordinate_rois none => [[100, 100], [w - 100, h - 100]]


        for idx, dataImage in enumerate(dataImages):
            cid = dataImage.cid
            if cid not in self.list_sort:
                self.list_sort[cid] = load_sort(cid)
        
        #check sort exit
        cid_dels = []
        # print('self.list_sort first :', self.list_sort)
        for cid in self.list_sort:
            time_check = time.time() - self.list_sort[cid].time_check
            # print(f'cid : {str(cid)}, time_check : {time_check}')
            if time_check >= self.time_kill_sort:
                # print('del sort 1')
                cid_dels.append(cid)
        for cid_del in cid_dels:
            # print('del sort 2')
            del self.list_sort[cid_del]

        # print('self.list_sort second :', self.list_sort)
        dict_data = main_test(dataImages, 
                              coordinate_rois,
                              self.list_sort,
                              self.yolo_person,
                              self.personFalse)

        return dict_data

      
class AITest(AIInterface):
    def __init__(self): 
        self.config_file = ""
        self.name = ""
        self.type_ai = ""
        self.ai_id = -1
        
    def Init(self):
        print(self.name , "-> Load model file")
        # time.sleep(3)
        # print(self.name , "-> Init default value")
        # time.sleep(1)
        
    def Detect(self, img, ares, a, b):
        img_out = None
        ret = {"obj": 4, "warning": True}
        time.sleep(0.1)

        return ret, img_out


class FaceData:
    def __init__(self): 
        self.face_id = -1
        self.time_begin = -1
        self.time_update = -1
        self.landmarks = []
        self.face_image = None
        self.cid = -1

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
        self.d_detector = dlib.get_frontal_face_detector()
        self.d_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


    def BestFace(self, img1, img2):
        face_img_resized1 = cv2.resize(img1, (112,112), interpolation = cv2.INTER_AREA)
        gray1 = cv2.cvtColor(face_img_resized1, cv2.COLOR_BGR2GRAY)
        laplacian_var1 = cv2.Laplacian(gray1, cv2.CV_64F).var()

        face_img_resized2 = cv2.resize(img2, (112,112), interpolation = cv2.INTER_AREA)
        gray2 = cv2.cvtColor(face_img_resized2, cv2.COLOR_BGR2GRAY)
        laplacian_var2 = cv2.Laplacian(gray2, cv2.CV_64F).var()

        h_img = cv2.hconcat([face_img_resized1, face_img_resized2])

        if laplacian_var2 > laplacian_var1:
            dets = self.d_detector(gray2, 1)
            if len(dets) > 0:
                shape = self.d_predictor(gray2, dets[0])
                lm5 = extract_lankmarks5(shape)
                return 1, face_img_resized2, lm5
            return 0, face_img_resized1, None
        else:
            return 0, face_img_resized1, None

    def PushToResult(self, ret):
        if ret.camera_id in self.list_face_ret:
            f_existed = False
            for i in range(len(self.list_face_ret[ret.camera_id])):
                if self.list_face_ret[ret.camera_id][i].face_id == ret.id_tracking:
                    f_existed = True
                    self.list_face_ret[ret.camera_id][i].time_update = time.time()
                    ret_f, best_face, lm5 = self.BestFace(self.list_face_ret[ret.camera_id][i].face_image, numpy.asarray(ret.img_face))
                    if ret_f > 0 and lm5 != None:
                        self.list_face_ret[ret.camera_id][i].face_image = best_face.copy()
                        self.list_face_ret[ret.camera_id][i].landmarks = []
                        for point in lm5:
                            self.list_face_ret[ret.camera_id][i].landmarks.append(point)
                    break
            if f_existed is False:
                face_data = FaceData()
                face_data.cid = ret.camera_id
                face_data.face_id = ret.id_tracking
                face_data.time_begin = time.time()
                face_data.time_update = time.time()
                image_face = cv2.resize(numpy.asarray(ret.img_face), (112,112), interpolation = cv2.INTER_AREA)
                face_data.face_image = image_face.copy()
                face_data.landmarks = []
                self.list_face_ret[ret.camera_id].append(face_data)

        else:
            face_data = FaceData()
            face_data.face_id = ret.id_tracking
            face_data.cid = ret.camera_id
            face_data.time_begin = time.time()
            face_data.time_update = time.time()
            image_face = cv2.resize(numpy.asarray(ret.img_face), (112,112), interpolation = cv2.INTER_AREA)
            face_data.face_image = image_face.copy()
            face_data.landmarks = []
            self.list_face_ret[ret.camera_id] = []
            self.list_face_ret[ret.camera_id].append(face_data)

    
    def Detect(self, dataImages, coordinate_rois, labels_allow_helmet = None,  labels_allow_uniform = None):
        imgs_out = {}
        for dataImage in dataImages:
            np_image_data = numpy.asarray(dataImage.image)
            m = FaceDetect.Mat3b.from_array(np_image_data)
            ret = self.face_detecter.Detect(dataImage.cid, m)
            #Draw img
            for i in ret:
                dataImage.image = cv2.rectangle(dataImage.image, (int(i.bbox.x), int(i.bbox.y)), (int(i.bbox.width + i.bbox.x), int(i.bbox.height + i.bbox.y)), (255, 0, 0), 2)
                dataImage.image = cv2.putText(dataImage.image, str(i.id_tracking), (int(i.bbox.x), int(i.bbox.y) - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
                self.PushToResult(i)
            imgs_out[dataImage.cid] = dataImage.image

        return None, imgs_out

    def GetResult(self):
        ret = []
        list_keys = list(self.list_face_ret.keys())

        #Get result
        for key in list_keys:
            i = 0
            while True:
                if key in self.list_face_ret:
                    if i >= len(self.list_face_ret[key]):
                        break
                    # print(self.list_face_ret[key][i].landmarks)
                   #If timeout update new data
                    if time.time() - self.list_face_ret[key][i].time_update > 10:
                        ret.append(copy.copy(self.list_face_ret[key][i]))
                        self.list_face_ret[key].pop(i)
                        i = i - 1
                    #If face is exist long -> reget face 
                    elif self.list_face_ret[key][i].time_update - self.list_face_ret[key][i].time_begin > 10:
                        ret.append(copy.copy(self.list_face_ret[key][i]))
                        self.list_face_ret[key][i].time_begin = self.list_face_ret[key][i].time_update
                    i+=1

                    #Remove camera have data is none
                    if len(self.list_face_ret[key]) == 0:
                        self.list_face_ret.pop(key)
                else:
                    break
        return ret



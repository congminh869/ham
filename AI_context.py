import time
import json
import datetime
import sys
import os
import numpy
import cv2
from pathlib import Path
# from AI_Interface import AI_FENCE,  AITest,
from AI_Interface import AI_FaceDetect, FaceData, AI_TBA, AI_TUNNEL, AI_LICENSE_PLATE, AI_HSV, AI_CLOCK, AI_FENCE, AI_BELT, AITest, AI_PersonHoldThingDetect
from shm.SHM import SHMWriter
from shm.SHMReader_3 import Channel, DataImage, SHMReader
from AI_Interface import AITest
import multiprocessing
from multiprocessing import Process, Pipe, Manager, Condition, Event
from cam_context import CameraContext, CameraProfileContext ,filter_similar_cameras, compare_list_cameras, get_change_list_cameras
from queue import Queue
import queue, threading
import struct
from signal import signal, SIGPIPE, SIG_DFL  
from redis_client import RedisClient
import redis
import json
signal(SIGPIPE,SIG_DFL)

NDTTEST = False
DEBUG_SHOW_IMG = False

def getInfoCameraTest(keyshm):
    FILE_CFG_TEST = "/home/mq/Desktop/AI_Middleware_Py2/config_test.txt"
    with open(FILE_CFG_TEST,'r+') as f:
        cfg_str = f.read()
        cfg_json = json.loads(cfg_str)
        for i in cfg_json:
            if i["key_redis"] == keyshm:
                return json.dumps(i)
    return ""

#Pop and Push to pipe	
def toPipe_string(pipe ,string):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    size = len(string)	
    res_string = bytes(string, 'utf-8')
    pipe.send_bytes(res_string)
    return
   
def fromPipe_string(pipe):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = None
    if pipe.poll(0.1):
	    encoded = pipe.recv_bytes()
    if encoded is None:
	    return ''
    string = encoded.decode('utf-8')
    return string

def putPipe_image(pipe , frame):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    h, w =  frame.image.shape[:2]
    shape = struct.pack('>IIQII',frame.cid, frame.type_id, frame.count, h, w)
    encoded = shape + frame.image.tobytes()
    pipe.send_bytes(encoded)
    return

def getPipe_image(pipe):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = None
    if pipe.poll(0.1):
        encoded = pipe.recv_bytes()
    if encoded is None:
        return None
    cid, type_id, count, h, w = struct.unpack('>IIQII',encoded[0:24])
    image = numpy.frombuffer(encoded, dtype=numpy.uint8, offset=24).reshape(h,w,3)
    frame = DataImage(cid, type_id, count, image.copy())
    return frame
    

class Channel:
    list_cid = []
    key_shm = -1
    size_shm = 5
    width_img = 1920
    height_img = 1080
    depth_img = 3
    
    def __init__(self, _key):
        self.key_shm = _key

class DataImage:
    cid = -1
    type_id = -1
    count = -1
    image = None
    
    def __init__(self, _cid, _type_id, _count, _image):
        self.cid = _cid
        self.type_id = _type_id
        self.count = _count
        self.image = _image   
        
#define id AI
GS_HangRao = 1 #
GS_KhuVuc_ThiCong = 2 # 
GS_KhuVuc_HanChe = 3 ## 
GS_ThietBi_DongHo = 4 #
GS_ThietBi_DaoCachLy = 5 #
GS_DoBaoHo_DaiAnToan = 6 #
GS_NhietDo_2Diem = 7 
GS_NhietDo_1Vung = 8
ND_BienSo = 9 
ND_KhuonMat = 10 
GS_NguoiMangVatThe = 11

TIME_P = 5
class CameraTaskAI:
    id_cam = -1
    name = ""
    isPTZ = False  
    isTour = False  
    time_begin = -1
    time_end = -1
    current_AI_type = ""
    current_ptz_id = ""
    current_ptz = []
    tour_presets = []
    current_are = []
    
    def isTimeOn(self):
        if self.isTour is True:
            return True
        else:
            t_now = time.time() + 7*3600 
            if t_now >= self.time_begin and t_now <= self.time_end:
                return True
            return False
            
    
    def Copy(self,src):
        self.id_cam = src.id_cam
        self.name = src.name
        self.isPTZ = src.isPTZ  
        self.isTour = src.isTour  
        self.time_begin = src.time_begin
        self.time_end = src.time_end
        self.current_AI_type = src.current_AI_type
        self.current_ptz_id = src.current_ptz_id
        self.current_ptz = src.current_ptz
        self.tour_presets = src.tour_presets
        self.current_are = src.current_are
    
    def Load(self, camctx):
        if camctx.id_cam > 0 :
            self.id_cam = camctx.id_cam
            self.name =  camctx.name
            self.isPTZ = camctx.isPTZ
            if self.isPTZ is True:
                self.current_ptz = camctx.current_ptz
                self.current_ptz_id = camctx.current_ptz_id                
            for ai in camctx.ais:
                f_break = False
                t_now = time.time() + 7*3600
                if ai.tour_name == "": #if Ai feature not set tour
                    self.isTour = False  
                    for preset in ai.presets:
                        if t_now >= preset.time_task.time_begin and t_now <= preset.time_task.time_end:
                            self.time_begin = preset.time_task.time_begin
                            self.time_end = preset.time_task.time_end
                            self.current_AI_type = ai.ai_type
                            f_break = True
                            break
                else: #if Ai feature set tour
                    self.isTour = True 
                    for time_e in ai.tour_times:
                        if t_now >= time_e - TIME_P and t_now <= time_e + TIME_P: 
                            self.isTour = True
                            tour_presets = ai.tour_presets 
                            f_break = True
                            break
                if f_break is True:
                    break

class AIContext:
    name = "" #
    type_ai = "" #
    key_shm = -1 #
    shm = None #
    process_id = -1 #
    max_cams = -1 #
    method_pointer = None
    method_pointer2 = None
    method_pointer_init = False
    process = None
    process2 = None
    #ai_tasks = []    
    list_camera = None
    camera_demo_json = ""

    host_server = ""
  
    def __init__(self):
        self.name = ""
        self.type_ai = ""
        self.key_shm = -1
        self.process_id = -1
        self.max_cams = -1
        self.method_pointer = None
        self.method_pointer2 = None
        self.method_pointer_init = False
        self.shm = Channel(self.key_shm)

        
    def Load(self, redis_c, ai_ctx, process_id):
        print(ai_ctx)        
        self.type_ai = ai_ctx["type_ai"]
        self.max_cams = ai_ctx["max_camera"]
        self.key_shm = ai_ctx["key_shm"]
        self.process_id = process_id
        manager= Manager()
        self.list_camera = manager.list()
        self.method_pointer_init = False
        
        print("========================================")
        print("--Type ai ", self.type_ai)
        print("--Key shm ai ", self.key_shm)
        print("========================================")
        if self.type_ai  == "GS_HangRao":
            self.method_pointer =AITest()  
        elif self.type_ai  == "GS_KhuVuc_ThiCong": #GS_KhuVuc_ThiCong
            self.method_pointer = AITest()  
        elif self.type_ai == "GS_KhuVuc_HanChe":
            self.method_pointer = AI_TUNNEL()   
        elif self.type_ai == "GS_ThietBi_DongHo":
            self.method_pointer = AITest()   
        elif self.type_ai == "GS_ThietBi_DaoCachLy":
            self.method_pointer = AITest() 
        elif self.type_ai == "GS_DoBaoHo_DaiAnToan":
            self.method_pointer =  AITest() #AITest()   
        elif self.type_ai == "ND_BienSo":
            self.method_pointer = AI_LICENSE_PLATE()  
        elif self.type_ai == "GS_NguoiMangVatThe":
            self.method_pointer = AITest()  
        elif self.type_ai == "ND_KhuonMat":
            self.method_pointer = AI_FaceDetect()  

        if self.method_pointer is None:
            return  
        if "list_camera" in ai_ctx:
            for i_ai_ctx in ai_ctx["list_camera"]:
                cam_ctx_json = redis_c.Get2(i_ai_ctx["key"])

                camctx = CameraContext()
                camctx.Load(cam_ctx_json)
                self.list_camera.append(camctx)
                # ai_task = CameraTaskAI()
                # ai_task.Load(camctx)
                # if ai_task.id_cam > 0:
                #     ipos = -1
                #     for i, task in enumerate(self.ai_tasks):
                #         if task.id_cam ==  ai_task.id_cam:
                #             ipos = i
                #             break
                #     if ipos >= 0:
                #         if ai_task.isTimeOn() is True:
                #             self.ai_tasks[ipos].Copy(ai_task)
                #         else:
                #             self.ai_tasks.pop(ipos)
                #     else:
                #         if ai_task.isTimeOn() is True:
                #             self.ai_tasks.append(ai_task)                           
                            
            
        self.name = self.type_ai + "_" + str(self.process_id)      
        self.shm = Channel(self.key_shm)     

        self.method_pointer.config_file = "./config_test.txt"
        self.method_pointer.name = self.name
        self.method_pointer.type_ai = self.type_ai
        if self.method_pointer2 is not None:
            self.method_pointer2.name = self.name
            self.method_pointer2.type_ai = self.type_ai

    def LoadCamDemo(self, redis_c, id_cam,  redis_demo):
        f_cam_existed = False
        camera_demo = redis_demo.Get3("demo")
        if len(camera_demo) > 0:
            camctx = CameraContext()
            camctx.Load(camera_demo)
            if camctx.id_cam == id_cam:
                key_cam = "vms:cameras:" + str(id_cam)
                cam_ctx_json = redis_c.Get2(key_cam) 
                # self.camera_demo[i].Load(cam_ctx_json)
                # self.list_camera.pop(i)
                f_cam_existed = True
        if f_cam_existed is True:
            return 

        key_cam = "vms:cameras:" + str(id_cam)
        cam_ctx_json = redis_c.Get2(key_cam)
        cam_ctx_json["ai_type_demo"] = self.type_ai
        redis_demo.Set("demo", json.dumps(cam_ctx_json))

    def ClearCamDemo(self, redis_demo):
        print("Camera clear ------------------------------------------->")
        redis_demo.Set("demo", "{}")       

    def LoadListCam(self, redis_c, ai_ctx):
        if "list_camera" in ai_ctx:
            #Check camera new or update
            for i_ai_ctx in ai_ctx["list_camera"]:
                f_cam_existed = False
                for i in range(len(self.list_camera)):
                    if i_ai_ctx["key"] == self.list_camera[i].key_redis:
                        cam_ctx_json = redis_c.Get2(i_ai_ctx["key"])  
                        self.list_camera[i].Load(cam_ctx_json)
                        f_cam_existed = True
                        break
                if f_cam_existed is True:
                    continue

                cam_ctx_json = redis_c.Get2(i_ai_ctx["key"])
                if cam_ctx_json != {}: 
                    camctx = CameraContext()
                    camctx.Load(cam_ctx_json)
                    self.list_camera.append(camctx)
                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Add camera : ", i_ai_ctx["key"] )

            #Check camera remove
            i = 0
            while True:
                if i >= len(self.list_camera) -1:
                    break
                f_cam_existed = False
                for i_ai_ctx in ai_ctx["list_camera"]:
                    if i_ai_ctx["key"] == self.list_camera[i].key_redis:
                        f_cam_existed = True
                        break
                if f_cam_existed is False:
                    self.list_camera.pop(i)
                    i = i -1
                i+=1

        for cam in self.list_camera:
            print(self.type_ai, "--->LIST CAMERA :", cam.id_cam)       
       
    
    def Run(self,pipe, pipe_demo):
        print(self.name, " -----------------------------> RUN ")
        self.process = Process(target=self.ProcessWorker, args=(pipe, pipe_demo, self.method_pointer)) 
        self.process.daemon = False
        self.process.start()

    def Restart(self,pipe, pipe_demo):
        print(self.name, " -----------------------------> RESTART ")
        self.process = Process(target=self.ProcessWorker, args=(pipe, pipe_demo, self.method_pointer)) 
        self.process.daemon = False
        self.process.start()
  
        
    def isRunning(self):
        return self.process.is_alive() 
        
    def Stop(self):
        self.process.terminate()

    def _convertRule2Rect(self, image, rule):
        if rule["type"] == "polygon":
            points_ret = []
            for point in rule["points"]:
                x = point[0] * image.shape[1]
                y = point[1] * image.shape[0]
                points_ret.append([x,y])
            arr = numpy.array(points_ret, numpy.float32)
            x,y,w,h = cv2.boundingRect(arr)
            return 0,x,y,w,h
        elif rule["type"] == "circle":
            x = (rule["x"] - rule["radius"])* image.shape[1]
            y = (rule["y"] - rule["radius"])* image.shape[0]
            w = (rule["x"] + rule["radius"])* image.shape[1]
            h = (rule["y"] + rule["radius"])* image.shape[0]
            return 0,x,y,w,h
        x = 0
        y = 0
        h, w, c = image.shape
        return -1, x,y,w,h

    def convertPointsResize(self, w_org, h_org, w_rs, h_rs, points):
        if w_rs == w_org and h_rs == h_org:
            return points
        points_ret = []
        for point in points:
            x = float(point[0] * float(w_rs)/float(w_org))
            y = float(point[1] * float(h_rs)/float(h_org))
            points_ret.append([x,y])
        return points_ret

    def convert_Rule_percentage2coordinate(self, type, frame, rule_per):
        points_ret = []
        if type == "polygon":
            for point in rule_per["points"]:
                x = int(point[0] * frame.shape[1])
                y = int(point[1] * frame.shape[0])
                points_ret.append([x,y])
        elif type == "rectangle":
            if rule_per["name"] == "__default__":
                x = 0
                y = 0
                h, w, c = frame.shape
                wid = w
                hei = h
                points_ret = [[x,y],[x+wid, y],[x+wid, y+hei],[x, y+ hei]]
            else:
                h, w, c = frame.shape
                x = int(rule_per["x"] * w)
                y = int(rule_per["y"] * h)
                wid = int(rule_per["width"] * w)
                hei = int(rule_per["height"] * h)
                points_ret = [[x,y],[x+wid, y],[x+wid, y+hei],[x, y+ hei]]
        else:
            ret, x, y, wid,hei = self._convertRule2Rect(frame, rules[0])
            points_ret = [[x,y],[x+wid, y],[x+wid, y+hei],[x, y+ hei]]
        return points_ret


    def ParserInput(self, frame, resolution_img_org, rules, coordinate_rois,labels_allow_helmet,labels_allow_uniform ):
        dataImages = []
        if frame.image is None:
            return dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform
        dataImages.append(frame)

        if self.type_ai == "GS_HangRao" or self.type_ai == "GS_KhuVuc_ThiCong" or self.type_ai == "GS_KhuVuc_HanChe" or self.type_ai == "GS_ThietBi_DongHo" or self.type_ai == "GS_NguoiMangVatThe":
            if len(rules) > 0:
                coordinate_rois[frame.cid] = self.convert_Rule_percentage2coordinate(rules[0]["type"], frame.image, rules[0])          
            else: 
                x = 0
                y = 0
                h, w, c = frame.image.shape
                wid = w
                hei = h
                # if self.type_ai == "GS_NguoiMangVatThe":
                coordinate_rois[frame.cid] =  [[x,y],[x+wid, y],[x+wid, y+hei],[x, y+ hei]]          

        elif self.type_ai == "GS_DoBaoHo_DaiAnToan":
            if len(rules) > 0:
                coordinate_rois[frame.cid] = self.convert_Rule_percentage2coordinate(rules[0]["type"], frame.image, rules[0])          
            else: 
                x = 0
                y = 0
                h, w, c = frame.image.shape
                wid = w
                hei = h
                # if self.type_ai == "GS_NguoiMangVatThe":
                coordinate_rois[frame.cid] =  [[x,y],[x+wid, y],[x+wid, y+hei],[x, y+ hei]]          

        elif self.type_ai == "ND_BienSo" or self.type_ai == "ND_KhuonMat":
            x = 0
            y = 0
            h, w, c = frame.image.shape
            wid = w
            hei = h
            coordinate_rois[frame.cid] = [x,y,x+wid, y+ hei]

        elif self.type_ai == "GS_ThietBi_DaoCachLy":
            if len(rules) > 0:
                for i, rule in enumerate(rules):
                    coordinate_rois[frame.cid][i] = self.convert_Rule_percentage2coordinate(rule["type"], frame.image, rule)
                    
        if self.type_ai == "GS_KhuVuc_ThiCong" or self.type_ai == "GS_KhuVuc_HanChe" or self.type_ai == "GS_DoBaoHo_DaiAnToan" :
            labels_allow_helmet[frame.cid] = [1,2,3,4,5,6]# [[0]]
            labels_allow_uniform[frame.cid] = [[4],[7]]# [quan=[4,5,6], ao=[7,8]]
        return dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform

    def ThreadRead(self, reader_shm, img_queue):
        while True:
            count = 0
            ret, arr = reader_shm.Read()
            if ret > 0:
                for list_frame in arr:
                    if len(list_frame) <= 0:
                        continue
                    for frame in list_frame:
                        img_queue.put(frame)
                        count+=1
    
    def ProcessWorker(self, pipe, pipe_demo, method_ptr):
        if method_ptr is None:
            return  

        time_begin= 0
        if self.type_ai == "ND_BienSo":
            self.shm.size_shm = 1
        else:
            self.shm.size_shm = method_ptr.size_shm
        self.shm.width_img = method_ptr.width_img
        self.shm.height_img = method_ptr.height_img
        self.shm.depth_img = method_ptr.depth_img
        reader_shm = SHMReader(self.shm)
        if self.method_pointer_init is False :
            method_ptr.Init()
            self.method_pointer_init = True

        coordinate_rois = {}
        labels_allow_helmet = {}
        labels_allow_uniform = {}

        debug_show_timeout = []

        cam_demo = None
        count_frame_failed = 0
        count_frame_demo = 0

        shm_writer = SHMWriter(pipe_demo)
        shm_writer.Init()

        list_timeout_ptz = []

        redis_c= None
        FILE_CFG = "./config.txt"
        with open(FILE_CFG,'r+') as f:
            cfg_str = f.read()
            cfg_json = json.loads(cfg_str)
            redis_c = RedisClient()
            redis_c.Init(cfg_json["redis_client"]["host"], cfg_json["redis_client"]["port"], cfg_json["redis_client"]["db"], cfg_json["redis_client"]["pwd"])
            redis_c_demo = RedisClient()
            redis_c_demo.Init(cfg_json["redis_client"]["host"], cfg_json["redis_client"]["port"], 8, cfg_json["redis_client"]["pwd"])

        while True:
            camera_demo = redis_c_demo.Get3("demo")
            if len(camera_demo) > 0 and camera_demo["ai_type_demo"] == self.type_ai:
                cam_ctx = CameraContext()
                cam_ctx.Load(camera_demo)       
                if cam_demo is None:
                    print("------------------------------->Capture")
                    cam_demo = cv2.VideoCapture(cam_ctx.GetRTSPBest())                    
                    # cam_demo = cv2.VideoCapture("/home/evnadmin/Documents/hangraocut.mp4")
                    continue
                else:
                    ret, frame = cam_demo.read()
                    if ret < 0:
                        count_frame_failed+=1
                        if count_frame_failed > 30:
                            cam_demo = None
                    else:
                        count_frame_demo+=1
                        rules = []
                        resolution_img_org = []
                        idc = cam_ctx.id_cam
                        dataImages = []
                        frame_data = DataImage(cam_ctx.id_cam, 1, count_frame_demo, frame) 
                        dataImages.append(frame_data)

                        key_cam = "vms:cameras:" + str(cam_ctx.id_cam)
                        print("---------------------------->", key_cam)
                        cam_ctx_json = redis_c.Get2(key_cam)
                        if cam_ctx_json == {}:
                            continue 
                        _camctx = CameraContext()
                        _camctx.Load(cam_ctx_json)
                        rules, resolution_img_org = _camctx.getRuleCurrent(self.type_ai)
                        list_timeout_ptz = _camctx.getRuleTimeoutPTZ(list_timeout_ptz, self.type_ai)
                        if len(resolution_img_org) == 2:
                            _, coordinate_rois,labels_allow_helmet,labels_allow_uniform = self.ParserInput(frame_data, resolution_img_org, rules, coordinate_rois,labels_allow_helmet,labels_allow_uniform )
                        if len(dataImages) > 0 and len(coordinate_rois) > 0:
                            result, imgs_debug = method_ptr.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
                            if imgs_debug is not None:
                                print(len(imgs_debug))
                                if type(imgs_debug) is dict:
                                    for cid_img in imgs_debug:
                                        scale = cv2.resize(imgs_debug[cid_img], (1280, 720))
                                        frame_data2 = DataImage(cam_ctx.id_cam, 1, count_frame_demo, scale) 
                                        shm_writer.Write(frame_data2)
                            if self.type_ai == "ND_KhuonMat":
                                result = method_ptr.GetResult()
                                for ret_face in result:
                                    if ret_face.face_image is not None:
                                        now_d = datetime.datetime.fromtimestamp(time.time())  
                                        now_d = datetime.datetime.utcnow() 
                                        hasLM = False
                                        data_face = {}
                                        for ip, point in enumerate(ret_face.landmarks):
                                            hasLM = True
                                            if ip == 0:
                                                data_face["left_eye"] = [point[0], point[1]]
                                            elif ip == 1:
                                                data_face["right_eye"] = [point[0], point[1]]
                                            elif ip == 2:
                                                data_face["nose"] = [point[0], point[1]]
                                            elif ip == 3:
                                                data_face["mouth_left"] = [point[0], point[1]]
                                            elif ip == 4:
                                                data_face["mouth_right"] = [point[0], point[1]]
                                        if hasLM is True:
                                            list_cam = []
                                            folder_path = "/mnt/hdd/vms/middlewares/" + self.type_ai   
                                            p = Path(folder_path)
                                            if p.exists() is False:
                                                os.mkdir(folder_path)
                                            name_img = folder_path+ "/" + str(ret_face.cid) + "_" + now_d.strftime("%Y%m%d_%H%M%S") + ".jpg"
                                            cv2.imwrite(name_img, ret_face.face_image)
                                            list_cam.append(name_img)

                                            # Event                       
                                            if len(list_cam) > 0:
                                                data = {"deviceId" : ret_face.cid,
                                                    "type": self.type_ai,
                                                    "data" : data_face,
                                                    "imagePaths" : list_cam,
                                                    "time" : now_d.strftime("%Y-%m-%d %H:%M:%S")}

                                                data_pipe_json = json.dumps(data)
                                                toPipe_string(pipe, data_pipe_json)
                                continue    
                            if result is not None :
                                # print(result) 
                                #now_d = datetime.datetime.now()
                                now_d = datetime.datetime.fromtimestamp(time.time())  
                                now_d = datetime.datetime.utcnow() 
                                list_cam = []
                                if cam_ctx.id_cam in result:    
                                    type_ai_s = self.type_ai 
                                    if self.type_ai == "GS_NguoiMangVatThe":                                
                                        type_ai_s = "GS_KhuVuc_ThiCong"                            
                                    folder_path = "/mnt/hdd/vms/middlewares/" + type_ai_s
                                    p = Path(folder_path)
                                    if p.exists() is False:
                                        os.mkdir(folder_path)
                                    name_img = folder_path+ "/" + str(cam_ctx.id_cam) + "_" + now_d.strftime("%Y%m%d_%H%M%S") + ".jpg"
                                    cv2.imwrite(name_img, result[cam_ctx.id_cam]["img"])
                                    list_cam.append(name_img)
                                
                                    # Event                       
                                    if len(list_cam) > 0:
                                        data_info = ""
                                        if self.type_ai == "ND_BienSo":
                                            data_info = result[cam_ctx.id_cam]["data"][0]
                                        else:
                                            data_info = ""
                                        type_ai_s = self.type_ai 
                                        if self.type_ai == "GS_NguoiMangVatThe" or self.type_ai == "GS_DoBaoHo_DaiAnToan":                                
                                            type_ai_s = "GS_KhuVuc_ThiCong"
                                        data = {"deviceId" : cam_ctx.id_cam,
                                            "type": type_ai_s,
                                            "data" : data_info, 
                                            "imagePaths" : list_cam,
                                            "time" : now_d.strftime("%Y-%m-%d %H:%M:%S")}

                                        data_pipe_json = json.dumps(data)
                                        toPipe_string(pipe, data_pipe_json)  
                    continue
            else:                
                cam_demo = None
                count_frame_failed = 0
                count_frame_demo = 0

            t1 = time.time()
            ret, arr = reader_shm.Read()
            count_t = 0
            # if self.type_ai == "ND_BienSo":
            #     print(self.type_ai, " | ", ret, " | ", len(arr))
            if ret > 0:
                dataImages = []
                for list_frame in arr:
                    if len(list_frame) <= 0:
                        continue

                    i_pos =  -1 
                    for i in range(len(self.list_camera)):
                        if list_frame[0].cid == self.list_camera[i].id_cam:
                            i_pos = i
                            break
                    if i_pos < 0:
                        continue

                    rules = []
                    resolution_img_org = []
                    idc = self.list_camera[i_pos].id_cam
                    for frame in list_frame:
                        dataImages.append(frame)
                        count_t+=1  
                        if DEBUG_SHOW_IMG is True:
                            if self.type_ai == "GS_HangRao":
                                img_scale = cv2.resize(frame.image, (640,360))
                                name = "Show_" + str(frame.cid)
                                f_existed_name = False
                                for i in range(len(debug_show_timeout)):
                                    if debug_show_timeout[i]["name"] == name:
                                        debug_show_timeout[i]["timecurrent"] = time.time() 
                                        f_existed_name = True
                                if f_existed_name is False:
                                    debug_show_timeout.append({"name": name, "timecurrent": time.time()})
                                cv2.imshow(name, img_scale)
                                cv2.waitKey(1)
                            for i in range(len(debug_show_timeout)):
                                if time.time() - debug_show_timeout[i]["timecurrent"] > 30:
                                    cv2.destroyWindow(debug_show_timeout[i]["name"])

                    key_cam = "vms:cameras:" + str(self.list_camera[i].id_cam)
                    # print("---------------------------->", key_cam)
                    cam_ctx_json = redis_c.Get2(key_cam) 
                    if cam_ctx_json == {}:
                            continue 
                    _camctx = CameraContext()
                    _camctx.Load(cam_ctx_json)
                    rules, resolution_img_org = _camctx.getRuleCurrent(self.type_ai)
                    # list_timeout_ptz = _camctx.getRuleTimeoutPTZ(list_timeout_ptz, self.type_ai)
                    if len(resolution_img_org) == 2:
                        dataImages1, coordinate_rois,labels_allow_helmet,labels_allow_uniform = self.ParserInput(list_frame[0], resolution_img_org, rules, coordinate_rois,labels_allow_helmet,labels_allow_uniform )
                    # if self.type_ai == "ND_BienSo":
                    #     print(len(resolution_img_org), len(dataImages), len(coordinate_rois))
                if len(dataImages) > 0 and len(coordinate_rois) > 0:
                    # if self.type_ai == "ND_BienSo":
                    #     for img in dataImages:
                    #         print(img.cid, " | ", img.count )
                    result, imgs_debug = method_ptr.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
                    if self.type_ai == "ND_KhuonMat":
                        result = method_ptr.GetResult()
                        for ret_face in result:
                            if ret_face.face_image is not None:
                                now_d = datetime.datetime.fromtimestamp(time.time())  
                                now_d = datetime.datetime.utcnow() 
                                hasLM = False
                                data_face = {}
                                for ip, point in enumerate(ret_face.landmarks):
                                    hasLM = True
                                    if ip == 0:
                                        data_face["left_eye"] = [point[0], point[1]]
                                    elif ip == 1:
                                        data_face["right_eye"] = [point[0], point[1]]
                                    elif ip == 2:
                                        data_face["nose"] = [point[0], point[1]]
                                    elif ip == 3:
                                        data_face["mouth_left"] = [point[0], point[1]]
                                    elif ip == 4:
                                        data_face["mouth_right"] = [point[0], point[1]]
                                if hasLM is True:
                                    list_cam = []
                                    folder_path = "/mnt/hdd/vms/middlewares/" + self.type_ai   
                                    p = Path(folder_path)
                                    if p.exists() is False:
                                        os.mkdir(folder_path)
                                    name_img = folder_path+ "/" + str(ret_face.cid) + "_" + now_d.strftime("%Y%m%d_%H%M%S") + ".jpg"
                                    cv2.imwrite(name_img, ret_face.face_image)
                                    list_cam.append(name_img)

                                    # Event                       
                                    if len(list_cam) > 0:
                                        data = {"deviceId" : ret_face.cid,
                                            "type": self.type_ai,
                                            "data" : data_face,
                                            "imagePaths" : list_cam,
                                            "time" : now_d.strftime("%Y-%m-%d %H:%M:%S")}

                                        data_pipe_json = json.dumps(data)
                                        toPipe_string(pipe, data_pipe_json)
                        continue    

                    if result is not None :
                        # print(result) 
                        now_d = datetime.datetime.fromtimestamp(time.time())  
                        now_d = datetime.datetime.utcnow() 
                        list_cam = []
                        for cam_info in self.list_camera:
                            if len(list_cam) == len(result):
                                break
                            if cam_info.id_cam in result:    
                                type_ai_s = self.type_ai 
                                if self.type_ai == "GS_NguoiMangVatThe":                                
                                    type_ai_s = "GS_KhuVuc_ThiCong"                            
                                folder_path = "/mnt/hdd/vms/middlewares/" + type_ai_s
                                p = Path(folder_path)
                                if p.exists() is False:
                                    os.mkdir(folder_path)
                                name_img = folder_path+ "/" + str(cam_info.id_cam) + "_" + now_d.strftime("%Y%m%d_%H%M%S") + ".jpg"
                                cv2.imwrite(name_img, result[cam_info.id_cam]["img"])
                                list_cam.append(name_img)
                            
                                # Event                       
                                if len(list_cam) > 0:
                                    if self.type_ai == "ND_BienSo":
                                        print("data = ", result)
                                    data_info = result[cam_info.id_cam]["data"]
                                    type_ai_s = self.type_ai 
                                    if self.type_ai == "GS_NguoiMangVatThe" or self.type_ai == "GS_DoBaoHo_DaiAnToan":                                
                                        type_ai_s = "GS_KhuVuc_ThiCong"
                                    data = {"deviceId" : cam_info.id_cam,
                                        "type": type_ai_s,
                                        "data" : {"warning" : True},
                                        "imagePaths" : list_cam,
                                        "time" : now_d.strftime("%Y-%m-%d %H:%M:%S")}

                                    data_pipe_json = json.dumps(data)
                                    toPipe_string(pipe, data_pipe_json)
    
    

"""
version:  release v3.1.2
"""
import time
import datetime
import cv2
import numpy as np
import os
import glob
from collections import OrderedDict
from dataclasses import dataclass
import os
import shutil
import yaml
import copy

isLog = False  #    True   False
isShowImgRes = False    # is show image result

# if(isLog != True):  # sau này sẽ bỏ if này đi
#     with open('./config/config_camera.yaml', 'r') as file:
#         config_mode = yaml.safe_load(file)
#     save_mode = config_mode['save_mode']
#     show_mode = config_mode['show_mode']

#     if save_mode['holdthing'] or save_mode['all']:
#         out_holdthingdetect = cv2.VideoWriter('./SAVE_VIDEO/out_holdthingdetect.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (640,360))


# offset_frame_x, offset_frame_y, margin_frame_x, margin_frame_y , MOG2_history = 30, MOG2_varThreshold = 15, do_cao_canh_bao = 0.5 , videoName = 0, vid_width = 1920, vid_height = 1080
# (200,200,0,0, do_cao_canh_bao=1.0, videoName = 1)
dict_para = { \
    "0": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 0, 
        "contourAreaThreshold" : 350,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 2688, 
        "vid_height": 1520 \

    }, \
    "1": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 1, \
        "contourAreaThreshold" : 250,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "2": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 2, \
        "contourAreaThreshold" : 350,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "3": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 3, \
        "contourAreaThreshold" : 500,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "4": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 4, \
        "contourAreaThreshold" : 150,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "5": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 5, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "6": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 6, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "7": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 7, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "8": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 8, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "9": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 9, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "10": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 10, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "11": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 11, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "12": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 12, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "13": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 13, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "14": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 14, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "15": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 15, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "16": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 16, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "17": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 17, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "18": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 18, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "19": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 19, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "20": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 20, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "21": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 21, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "22": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 22, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "23": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 23, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "24": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 24, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "25": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 25, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    }, \
    "26": { \
        "offset_frame_x": 200, \
        "offset_frame_y": 200, \
        "margin_frame_x": 0, \
        "margin_frame_y": 0, \
        "MOG2_history": 30, \
        "MOG2_varThreshold": 15, \
        "do_cao_canh_bao": 1.0, \
        "videoName": 26, \
        "contourAreaThreshold" : 50,
        "so_frame_truoc_lien_tiep" : 2,
        "vid_width": 1920, \
        "vid_height": 1080 \

    } \


}


@dataclass
class bboxInfo:
    bbox: list
    name_class: str
    id_tracking: int
    class_conf: float
    class_ids: list


class DtectBox_phuong:
    def __init__(self):
        self.bbox = None
        self.name_class = None
        self.id_tracking = None
        self.class_conf = None 
        self.class_ids = None

def debug_tool(*args, **kwargs):
    # print(*args, **kwargs)   # nếu cần print thì uncomment
    if(isLog == True): 
        chuoi = str()
        for tung_str in args:
            # print("tung_str = ", tung_str)
        
            chuoi += str(tung_str)
        chuoi += "\n"
        currTime = datetime.datetime.now().strftime("%Y_%m_%d_%H")   # %Y_%m_%d_%H_%M
        pathLogFile = "./test_phuong/" + str(currTime) + ".txt"

        # cần thêm chức năng xóa log file cũ ở phần delete file theo chu kỳ
        
        with open(pathLogFile,"a") as AFile:
            AFile.write(chuoi)

def deleteFile(folderPath):
    if(isLog == True): 
        for filename in sorted(os.listdir(folderPath))[:-40]:
            filename_relPath = os.path.join(folderPath,filename)
            os.remove(filename_relPath)


def createFolder(path):
    if(isLog == True): 
        # Check whether the specified path exists or not
        isExist = os.path.exists(path)

        if not isExist:
            
            # Create a new directory because it does not exist 
            os.makedirs(path)
            debug_tool("The new directory is created!")

        # xoa file dang co trong thu muc
        files = glob.glob(path + '*')
        for f in files:
            os.remove(f)

class DataTracking_phuong:
    # def __init__(self, frame, dtectBoxs, cid, type_id, count):
    def __init__(self):
        self.frame = None
        self.dtectBoxs = None

        self.cid = 0 #int
        self.type_id = 0# int
        self.count = 0 # int

"""
do_cao_canh_bao : độ cao so với độ cao người, ví dụ: 0.5 tức là 0.5 * độ cao người
"""
class personHoldThing:
    def __init__(self, offset_frame_x, offset_frame_y, margin_frame_x, margin_frame_y , MOG2_history = 30, MOG2_varThreshold = 15, do_cao_canh_bao = 0.5 , videoName = 0, vid_width = 1920, vid_height = 1080 , contourAreaThreshold = 50 , so_frame_truoc_lien_tiep = 2 ):
        self.vid_width = vid_width
        self.vid_height = vid_height
        self.offset_frame_x = offset_frame_x
        self.offset_frame_y = offset_frame_y
        self.margin_frame_x = margin_frame_x
        self.margin_frame_y = margin_frame_y
        self.list_toa_do_frame_bo_phan_truoc = [self.vid_width - 1 , self.vid_height - 1, self.vid_width , self.vid_height]  #x0, y0, x1, y1
        self.frameCnt = 0
        self.frameCnt_2 = 0
        self.personCnt = 0
        self.do_cao_canh_bao = do_cao_canh_bao
        self.contourAreaThreshold = contourAreaThreshold
        self.so_frame_truoc_lien_tiep = so_frame_truoc_lien_tiep
        
        # You can set custom kernel size if you want.
        self.kernel = None


        # Initialize the background object.
        # self.backgroundObject = cv2.createBackgroundSubtractorMOG2(history=550,varThreshold=15,detectShadows = True)  # goc
        self.backgroundObject = cv2.createBackgroundSubtractorMOG2(history=MOG2_history,varThreshold=MOG2_varThreshold,detectShadows = True)


        self.dict_trang_thai_frame = {}
        
        self.t1_FPS = time.time()

        # tao thu muc luu data
        path_base = r'./test_phuong/'
        # path_base = r'/media/mq/New Volume/personHoldThing/'

        pathVideoResult = path_base + str(videoName) + "/" 

        self.path_frame_chua_ketluan = pathVideoResult + 'frame_chua_ketluan/'
        self.path_frame_ketluan = pathVideoResult + 'frame_ketluan/'
        self.path_frame_mask = pathVideoResult + 'frame_mask/'
        self.path_frame_trung_nguoi = pathVideoResult + 'frame_trung_nguoi/'
        self.path_frame_bophan = pathVideoResult + 'frame_bophan/'


        createFolder(self.path_frame_chua_ketluan)
        createFolder(self.path_frame_ketluan)
        createFolder(self.path_frame_mask)
        createFolder(self.path_frame_trung_nguoi)
        createFolder(self.path_frame_bophan)


    def returnResultHandle(self):
        # tinh FPS
        if(self.frameCnt_2 > 10):
            t2_FPS = time.time()
            debug_tool("FPS phat hien vat cao = ", self.frameCnt_2/(t2_FPS-self.t1_FPS))   
            debug_tool("Person per second phat hien vat cao = ", self.personCnt/(t2_FPS-self.t1_FPS))   

            self.frameCnt_2 = 0  
            self.personCnt = 0  
            self.t1_FPS =    t2_FPS

            #Xoa file trong thu muc
            deleteFile(self.path_frame_chua_ketluan)
            deleteFile(self.path_frame_ketluan)
            deleteFile(self.path_frame_mask)
            deleteFile(self.path_frame_trung_nguoi)
            deleteFile(self.path_frame_bophan)

        # xóa trạng thái các frame cũ quá
        if(len(self.dict_trang_thai_frame) >= 50):
            self.dict_trang_thai_frame.pop(next(iter(self.dict_trang_thai_frame)))
        # debug_tool("self.dict_trang_thai_frame = ", self.dict_trang_thai_frame)


        

    """
    list_DtectBox: list bbox
    """
    
    # def detect(self,dataTracking:DataTracking):
    def detect(self,list_dataTracking):
        list_dataTracking_result = []

        for  dataTracking in list_dataTracking: 
            ket_qua_phat_hien_vat_cao = DataTracking_phuong()

            frame = dataTracking.frame.copy()
            list_DtectBox = dataTracking.dtectBoxs

            #region lấy rộng, cao của ảnh (video)
            # print("shape = ", dataTracking.frame.copy().shape)
            self.vid_height , self.vid_width , _ = dataTracking.frame.copy().shape
            #endregion
            
            ket_qua_phat_hien_vat_cao.frame = dataTracking.frame.copy()
            ket_qua_phat_hien_vat_cao.dtectBoxs = []
            ket_qua_phat_hien_vat_cao.cid = dataTracking.cid
            ket_qua_phat_hien_vat_cao.type_id = dataTracking.type_id
            ket_qua_phat_hien_vat_cao.count = dataTracking.count

            if(len(list_DtectBox) == 0):
                self.returnResultHandle()
                list_dataTracking_result.append(ket_qua_phat_hien_vat_cao)
                return list_dataTracking_result

            currTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
            # debug_tool("currTime = ", currTime)


            # Create a copy of the frame to draw bounding boxes around the detected cars.
            frameCopy = frame.copy()
            flag_trung_nguoi = False   # flag trùng người


            self.frameCnt += 1
            self.frameCnt_2 += 1

            dict_ds_nguoi = {}
            cnt_nguoi = 0

            offset_docao_2_nguoi = 0 # -10

            list_toa_do_x0 = []
            list_toa_do_x1 = []
            list_toa_do_y0 = []
            list_toa_do_y1 = []

            for tung_DtectBox in list_DtectBox:
                list_toa_do_tung_nguoi = tung_DtectBox.bbox
            


                x0 = list_toa_do_tung_nguoi[0]
                y0 = list_toa_do_tung_nguoi[1]
                x1 = list_toa_do_tung_nguoi[2]
                y1 = list_toa_do_tung_nguoi[3]
                # debug_tool(x0, y0 ,x1, y1)
                list_toa_do_x0.append(x0)
                list_toa_do_x1.append(x1)
                list_toa_do_y0.append(y0)
                list_toa_do_y1.append(y1)

                nguong_do_cao = y0 - int( (y1-y0)/4 )
                nguong_do_cao_up = y0 - int( (y1-y0) * self.do_cao_canh_bao )
                if(nguong_do_cao < 0):
                    nguong_do_cao = 0  # có thể bỏ câu lệnh này

                    # khi ngưỡng dưới của khu vực detect < 0. thì ko detect frame này, tiến hành return
                    self.returnResultHandle()
                    list_dataTracking_result.append(ket_qua_phat_hien_vat_cao)
                    return list_dataTracking_result
                
                if(nguong_do_cao_up < 0):
                    nguong_do_cao_up = 0

                khuvuc_detect_toado = [(x0, nguong_do_cao),(x0, nguong_do_cao_up),(x1, nguong_do_cao_up),(x1, nguong_do_cao)]
                khuvuc_detect_polygon = cv2.polylines(frameCopy, [np.array(khuvuc_detect_toado, np.int32)], True, (15,220,10), thickness=1)

                # ve nguoi
                # cv2.polylines(frameCopy, [np.array([(x0,y0 + offset_docao_2_nguoi), (x1, y0 + offset_docao_2_nguoi), (x1, y1 + offset_docao_2_nguoi), (x0, y1 + offset_docao_2_nguoi)], np.int32)], True, (255,0,0), thickness=1)
                cv2.rectangle(frameCopy, (x0,y0 + offset_docao_2_nguoi), (x1, y1),(255, 0, 0), 1)


                dict_ds_nguoi[cnt_nguoi] = [[x0, y0 ,x1, y1], khuvuc_detect_toado, tung_DtectBox]
                

                # debug_tool("ds nguoi = ", dict_ds_nguoi[cnt_nguoi] ) 

                cnt_nguoi += 1
                self.personCnt += 1


                y0_min = min(list_toa_do_y0)
                y1_max = max(list_toa_do_y1)
                x0_min = min(list_toa_do_x0)
                x1_max = max(list_toa_do_x1)
            
                isChangeFrameFlag = False

                if( \
                    (x0_min < (self.list_toa_do_frame_bo_phan_truoc[0] - self.margin_frame_x)) or \
                    ( x1_max > (self.list_toa_do_frame_bo_phan_truoc[2] - self.margin_frame_x)) or \
                    ( y0_min < (self.list_toa_do_frame_bo_phan_truoc[1] - self.margin_frame_y)) or \
                    (y1_max > (self.list_toa_do_frame_bo_phan_truoc[3] - self.margin_frame_y)) \
                    ):
                    isChangeFrameFlag = True

                # if(self.frameCnt  == 10 or (self.frameCnt % 100 == 0) ):
                # if(self.frameCnt  == 10  ):
                if( isChangeFrameFlag ):
                    if(x0_min - self.offset_frame_x >= 0):
                        self.list_toa_do_frame_bo_phan_truoc[0] = x0_min - self.offset_frame_x
                    else:
                        self.list_toa_do_frame_bo_phan_truoc[0] = 0

                    if(y0_min - self.offset_frame_y >= 0):
                        self.list_toa_do_frame_bo_phan_truoc[1] = y0_min - self.offset_frame_y
                    else:
                        self.list_toa_do_frame_bo_phan_truoc[1] = 0

                    if(x1_max + self.offset_frame_x >= self.vid_width):
                        self.list_toa_do_frame_bo_phan_truoc[2] = self.vid_width
                    else:
                        self.list_toa_do_frame_bo_phan_truoc[2] = x1_max + self.offset_frame_x

                    if( y1_max + self.offset_frame_y >= self.vid_height):
                        self.list_toa_do_frame_bo_phan_truoc[3] = self.vid_height
                    else:
                        self.list_toa_do_frame_bo_phan_truoc[3] = y1_max + self.offset_frame_y

            frame_bophan_x0 = 0
            frame_bophan_x1 = 0
            frame_bophan_y0 = 0
            frame_bophan_y1 = 0
            if(True):
                # frame_bophan_x0 = self.list_toa_do_frame_bo_phan_truoc[0]
                # frame_bophan_x1 = self.list_toa_do_frame_bo_phan_truoc[2]
                # frame_bophan_y0 = self.list_toa_do_frame_bo_phan_truoc[1]
                # frame_bophan_y1 = self.list_toa_do_frame_bo_phan_truoc[3]

                # frame_bophan_x0 = 200
                # frame_bophan_x1 = 400
                # frame_bophan_y0 = 200
                # frame_bophan_y1 = 400

                frame_bophan_x0 = 0
                frame_bophan_x1 = self.vid_width
                frame_bophan_y0 = 0
                frame_bophan_y1 = self.vid_height

            # khi chọn frame bộ phận, thì đang chưa xét kích thước frame bộ phận tương quan với thông số độ cao cảnh báo ???  => cần làm thêm
            frame_bophan = frame[ frame_bophan_y0:frame_bophan_y1, frame_bophan_x0:frame_bophan_x1]

            # cv2.imshow("frame_bophan", frame_bophan)
            if(isLog == True): 
                cv2.imwrite(self.path_frame_bophan + currTime + "_" + str(self.frameCnt) + ".jpg", frame_bophan)


            # Apply the background object on the frame to get the segmented mask.
            t1 = time.time()
            fgmask = self.backgroundObject.apply(frame_bophan)

            #initialMask = fgmask.copy()


            # Perform thresholding to get rid of the shadows.
            _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)  # goc
            # debug_tool("type(fgmask) = ", type(fgmask))   # type(fgmask) = <class 'numpy.ndarray'>
            # debug_tool("fgmask = ", fgmask)
            t2 = time.time()
            debug_tool("thoi gian tru nen = " , t2 -t1)
            #noisymask = fgmask.copy()

            # cv2.imshow("fgmask", cv2.resize(fgmask, None, fx=0.7, fy=0.7))
            if(isLog == True): 
                cv2.imwrite(self.path_frame_mask + currTime + "_" + str(self.frameCnt) + ".jpg", fgmask) 
                    
            # Apply some morphological operations to make sure you have a good mask
            fgmask = cv2.erode(fgmask, self.kernel, iterations = 1)
            fgmask = cv2.dilate(fgmask, self.kernel, iterations = 2)


            data = OrderedDict(dict_ds_nguoi)  
            # print("type(data) = ", type(data))  #type(data) =  <class 'collections.OrderedDict'>

            # debug_tool(data.keys())
            # data.move_to_end(2)
            # debug_tool(data.keys())

            data_2 = data.copy()

            
            for index, (key, value) in enumerate(data.items()):
                # pass
                # data goc
                # debug_tool("*************$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$***************")
                # debug_tool("data = ", data)
                # debug_tool(index, key, value)



                # data 2
                # debug_tool("data_2 luc truoc = ", data_2)
                data_2.move_to_end(key)
                # debug_tool("data_2 = ", data_2)
                # for index_2, (key_2, value_2) in enumerate(data_2.items()):
                #     debug_tool(index_2, key_2, value_2)
                # debug_tool("********************************************************")

                # *************$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$***************
                # data =  OrderedDict([(

                # 0, [[554, 183, 584, 225], [(554, 173), (554, 99), (584, 99), (584, 173)]]), (
                # 1, [[519, 189, 562, 303], [(519, 161), (519, 0), (562, 0), (562, 161)]]), (
                # 2, [[626, 181, 661, 227], [(626, 170), (626, 89), (661, 89), (661, 170)]]), (
                # 3, [[480, 210, 584, 472], [(480, 145), (480, 0), (584, 0), (584, 145)]])])

                # 1 1 [[519, 189, 562, 303], [(519, 161), (519, 0), (562, 0), (562, 161)]]

                # data_2 luc truoc =  OrderedDict([(
                # 1, [[519, 189, 562, 303], [(519, 161), (519, 0), (562, 0), (562, 161)]]), (
                # 2, [[626, 181, 661, 227], [(626, 170), (626, 89), (661, 89), (661, 170)]]), (
                # 3, [[480, 210, 584, 472], [(480, 145), (480, 0), (584, 0), (584, 145)]]), (
                # 0, [[554, 183, 584, 225], [(554, 173), (554, 99), (584, 99), (584, 173)]])])

                # data_2 =  OrderedDict([(2, [[626, 181, 661, 227], [(626, 170), (626, 89), (661, 89), (661, 170)]]), (3, [[480, 210, 584, 472], [(480, 145), (480, 0), (584, 0), (584, 145)]]), (0, [[554, 183, 584, 225], [(554, 173), (554, 99), (584, 99), (584, 173)]]), (1, [[519, 189, 562, 303], [(519, 161), (519, 0), (562, 0), (562, 161)]])])

                # 0 2 [[626, 181, 661, 227], [(626, 170), (626, 89), (661, 89), (661, 170)]]
                # 1 3 [[480, 210, 584, 472], [(480, 145), (480, 0), (584, 0), (584, 145)]]
                # 2 0 [[554, 183, 584, 225], [(554, 173), (554, 99), (584, 99), (584, 173)]]
                # 3 1 [[519, 189, 562, 303], [(519, 161), (519, 0), (562, 0), (562, 161)]]
                # ********************************************************

                

                khuvuc_detect_toado = value[1]
                # debug_tool("type(khuvuc_detect_toado) = ", type(khuvuc_detect_toado))
                # debug_tool("(khuvuc_detect_toado) = ", khuvuc_detect_toado)
                x0_chinh = value[0][0]
                y0_chinh = value[0][1]
                x1_chinh = value[0][2]
                y1_chinh = value[0][3]

                
                # xét người đó có bị trùng với các người còn lại hay không
                # ở đây có so sánh lại, vì các điểm mỗi lần là khác nhau, ví dụ: id 0 so sánh với 1 rồi, lần for tiếp theo thì 1 lại so sánh với 0, 2 lần so sánh này là khác nhau, vì được thêm các điểm phụ của id làm mốc so sánh
                for index, (key, value) in enumerate(data_2.items()):
                    if(index == (len(data_2) -1 ) ):
                        break

                    x0 = value[0][0]
                    y0 = value[0][1]
                    x1 = value[0][2]
                    y1 = value[0][3]

                    # debug_tool(x0,x1,y0,y1)

                    chieu_rong_crop = x1 - x0
                    do_cao_crop = y1 - y0

                    
                    result_trung_nguoi = \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0,y0 + offset_docao_2_nguoi) ,False) >= 0) or \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x1,y1 + offset_docao_2_nguoi) ,False) >= 0) or \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0,y1 + offset_docao_2_nguoi) ,False) >= 0) or \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x1,y0 + offset_docao_2_nguoi) ,False) >= 0) or \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0 + 2 * int(chieu_rong_crop/4),y0 + offset_docao_2_nguoi) ,False) >= 0) or \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0 + 1 * int(chieu_rong_crop/4),y0 + offset_docao_2_nguoi) ,False) >= 0) or \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0 + 3 * int(chieu_rong_crop/4),y0 + offset_docao_2_nguoi) ,False) >= 0) 
                    
                    # ở đây còn thiếu tính result_trung_nguoi dựa trên các điểm ở cạnh dưới bounding box, tức các điểm: y1 + offset_docao_2_nguoi   ; bên dưới mới là đủ,  nhưng chưa ban hành, vì sợ làm chậm đi nhiều                
                    # result_trung_nguoi = \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0,y0 + offset_docao_2_nguoi) ,False) >= 0) or \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x1,y1 + offset_docao_2_nguoi) ,False) >= 0) or \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0,y1 + offset_docao_2_nguoi) ,False) >= 0) or \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x1,y0 + offset_docao_2_nguoi) ,False) >= 0) or \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0 + 2 * int(chieu_rong_crop/4),y0 + offset_docao_2_nguoi) ,False) >= 0) or \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0 + 1 * int(chieu_rong_crop/4),y0 + offset_docao_2_nguoi) ,False) >= 0) or \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0 + 3 * int(chieu_rong_crop/4),y0 + offset_docao_2_nguoi) ,False) >= 0) or \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0 + 2 * int(chieu_rong_crop/4),y1 + offset_docao_2_nguoi) ,False) >= 0) or \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0 + 1 * int(chieu_rong_crop/4),y1 + offset_docao_2_nguoi) ,False) >= 0) or \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0 + 3 * int(chieu_rong_crop/4),y1 + offset_docao_2_nguoi) ,False) >= 0) or \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0 , y0 + 1 * int(do_cao_crop/4) + offset_docao_2_nguoi) ,False) >= 0) or \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0 , y0 + 2 * int(do_cao_crop/4) + offset_docao_2_nguoi) ,False) >= 0) or \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x0 , y0 + 3 * int(do_cao_crop/4) + offset_docao_2_nguoi) ,False) >= 0) or \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x1 , y0 + 1 * int(do_cao_crop/4) + offset_docao_2_nguoi) ,False) >= 0) or \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x1 , y0 + 2 * int(do_cao_crop/4) + offset_docao_2_nguoi) ,False) >= 0) or \
                    #     (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x1 , y0 + 3 * int(do_cao_crop/4) + offset_docao_2_nguoi) ,False) >= 0)                     

              
                        


                    # debug_tool("result_trung_nguoi = ", result_trung_nguoi)
                    if(result_trung_nguoi == True):
                        font                   = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (800,500)
                        fontScale              = 1
                        fontColor              = (0,0,255)
                        thickness              = 2
                        lineType               = 2
                        cv2.putText(frameCopy,'trung_nguoi', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                        
                        if(isLog == True): 
                            cv2.imwrite(self.path_frame_trung_nguoi + currTime + "_" + str(self.frameCnt) + ".jpg", frameCopy) 

                        flag_trung_nguoi = True

                        break

                if(flag_trung_nguoi == True):
                    break

            if(flag_trung_nguoi == True):
                # cv2.imshow('frameCopy', cv2.resize(frameCopy, None, fx=0.7, fy=0.7))
                # self.dict_trang_thai_frame[self.frameCnt] = False # chưa để False, vì trùng người thì vẫn có khả năng là có vật cao
                self.returnResultHandle()
                list_dataTracking_result.append(ket_qua_phat_hien_vat_cao)
                return list_dataTracking_result
                                
            # Detect contours in the frame.
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            
            # trang_thai_frame_truoc = False  # khai bao

            # if(len(self.dict_trang_thai_frame) <= 2):
            #     trang_thai_frame_truoc = False
            # else:
            #     trang_thai_frame_truoc = self.dict_trang_thai_frame[list(self.dict_trang_thai_frame)[-1]] and self.dict_trang_thai_frame[list(self.dict_trang_thai_frame)[-2]]

            # # debug_tool("trang_thai_frame_truoc", trang_thai_frame_truoc)

            
            # self.dict_trang_thai_frame[self.frameCnt] = False



            # loop over each contour found in the frame.
            currFrameInfo = {}
            
            for index, (key, value) in enumerate(data.items()):
                khuvuc_detect_toado = value[1]
                # debug_tool("type(khuvuc_detect_toado) = ", type(khuvuc_detect_toado))
                # debug_tool("(khuvuc_detect_toado) = ", khuvuc_detect_toado)
                x0_chinh = value[0][0]
                y0_chinh = value[0][1]
                x1_chinh = value[0][2]
                y1_chinh = value[0][3]

                currBoxId = value[2].id_tracking

                for cnt in contours:       
                    # Make sure the contour area is somewhat higher than some threshold to make sure its a car and not some noise.
                    if cv2.contourArea(cnt) > self.contourAreaThreshold:  
                        
                        # Retrieve the bounding box coordinates from the contour.
                        x, y, width, height = cv2.boundingRect(cnt)
                        # debug_tool("khuvuc_detect_toado = ", khuvuc_detect_toado)
                        # debug_tool("value[0] = ", value[0])
                        # debug_tool("tung_DtectBox.bbox = ", tung_DtectBox.bbox)
                        # debug_tool("value[2] = ", value[2].bbox)
                        result = \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x,y) ,False) >= 0) or \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x, y+height) ,False) >= 0) or \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x + width, y) ,False) >= 0) or \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x + 2 * int(width/4), y + height) ,False) >= 0) or \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x + 1 * int(width/4), y + height) ,False) >= 0) or \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x + 3 * int(width/4), y + height) ,False) >= 0) or \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x + 2 * int(width/4), y ) ,False) >= 0) or \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x + 1 * int(width/4), y ) ,False) >= 0) or \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x + 3 * int(width/4), y ) ,False) >= 0) or \
                        (cv2.pointPolygonTest(np.array(khuvuc_detect_toado, np.int32), (x + width, y + height) ,False) >= 0)


                        # if(True):
                        if(result):
                            cv2.rectangle(frameCopy, (frame_bophan_x0 + x , frame_bophan_y0 + y), (frame_bophan_x0 + x + width, frame_bophan_y0 + y + height),(0, 0, 255), 3)
                            cv2.line(frameCopy, (int((frame_bophan_x0 + x0_chinh + x1_chinh)/2), frame_bophan_y0 + y0_chinh), (int((frame_bophan_x0 + x0_chinh + x1_chinh)/2), frame_bophan_y0 + y), (0, 255, 0), thickness=2, lineType=8) 
                            
                            trang_thai_frame_truoc = False  # khai bao
                            currBoxResult = True
                            # 8: {1: [True], 2: [True]}, 9: {1: [True]}, 10: {1: [True], 2: [True]}, 11: {1: [True], 2: [True]}, 12: {2: [True]}, 13: {1: [True]}, 14: {}, 15: {}, 16: {},
                            
                            #region linh động số frame liên tiếp
                            list_prevFrameResult = []

                            if(len(self.dict_trang_thai_frame) <= self.so_frame_truoc_lien_tiep):
                                trang_thai_frame_truoc = False
                            else:
                                for i in range(1, self.so_frame_truoc_lien_tiep + 1):
                                    list_element = self.dict_trang_thai_frame[list(self.dict_trang_thai_frame)[-1 * i]]
                                    list_prevFrameResult.append(list_element.copy())
                                
                                list_currId_prevFrameResult = []
                                for prevFrameResult in list_prevFrameResult:
                                    list_currId_prevFrameResult.append(False) # mặc định kết quả trong frame trước của ID hiện tại = false, bên dưới sẽ set giá trị tương ứng nếu frame trước có ID hiện tại
                                    for key_FrameRes, value_FrameRes in prevFrameResult.items(): # 10: {1: [True], 2: [True]}
                                        
                                        if(key_FrameRes == currBoxId ):   # Có ID đang xét ở những frame trước thì set giá trị là kết quả của frame trước
                                            list_currId_prevFrameResult[-1] = value_FrameRes[0] #copy.deepcopy(value_FrameRes[0])

                                # Check if all items in a list are True:
                                trang_thai_frame_truoc = all(list_currId_prevFrameResult)  


                            #endregion linh động số frame liên tiếp

                            # ben duowis la goc của số frame liên tiếp là fix
                            # if(len(self.dict_trang_thai_frame) <= 2):
                            #     trang_thai_frame_truoc = False
                                
                            # else:
                            #     prevFrameResult_1 = self.dict_trang_thai_frame[list(self.dict_trang_thai_frame)[-1]]
                            #     prevFrameResult_2 = self.dict_trang_thai_frame[list(self.dict_trang_thai_frame)[-2]]
                            #     debug_tool("prevFrameResult_1 = ", prevFrameResult_1)  # prevFrameResult_1 = {1: [True], 2: [True]}
                            #     debug_tool("prevFrameResult_2 = ", prevFrameResult_2)

                            #     for key_3, value_3 in prevFrameResult_1.items():
                            #         # debug_tool("key_3 = ", key_3)
                            #         # debug_tool("value_3 = ", value_3)
                                    
                            #         if(key_3 == currBoxId ):
                            #             # nếu True thì mới xét tiếp frame trước đó nữa
                            #             if(value_3[0] == True):
                            #                 for key_4, value_4 in prevFrameResult_2.items():
                            #                     if(key_4 == currBoxId ):
                            #                         if(value_4[0] == True):
                            #                             trang_thai_frame_truoc = True
                                                        



                            # if(True):
                            if(trang_thai_frame_truoc == True):
                                cv2.putText(frameCopy, 'co vat the', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                                if(isLog == True): 
                                    cv2.imwrite(self.path_frame_ketluan + currTime + "_" + str(self.frameCnt) + ".jpg", frameCopy)
                                # gan cho bien kết quả
                                dtectBox_co_vat_cao = DtectBox_phuong()
                                dtectBox_co_vat_cao.bbox = value[2].bbox
                                dtectBox_co_vat_cao.name_class = value[2].name_class
                                dtectBox_co_vat_cao.id_tracking = currBoxId
                                dtectBox_co_vat_cao.class_conf = value[2].class_conf
                                dtectBox_co_vat_cao.class_ids = value[2].class_ids

                
                                
                                # ko chay voi  element la class
                                if(dtectBox_co_vat_cao not in ket_qua_phat_hien_vat_cao.dtectBoxs):
                                    ket_qua_phat_hien_vat_cao.dtectBoxs.append(dtectBox_co_vat_cao)


                            # self.dict_trang_thai_frame[self.frameCnt] =  True
                            if(isLog == True): 
                                cv2.imwrite(self.path_frame_chua_ketluan + currTime + "_" + str(self.frameCnt) + ".jpg", frameCopy)

                            currFrameInfo[currBoxId] = [currBoxResult]
                                                
            self.dict_trang_thai_frame[self.frameCnt]  = currFrameInfo
            debug_tool("self.dict_trang_thai_frame = ", self.dict_trang_thai_frame)  # 11: {1: [True], 2: [True]}, 12: {2: [True]}, 13: {1: [True]}, 14: {},

                    
                    # if( (x > x0 and x < x1 ) or ((x + width) > x0 and (x + width) < x1 )):
                    #     if( (y < nguong_do_cao) and (y > nguong_do_cao_up) ):
                    #     # if( (y < nguong_do_cao)  ):
                    #         # Draw a bounding box around the car.
                    #         cv2.rectangle(frameCopy, (x , y), (x + width, y + height),(0, 0, 255), 1)
                    #         cv2.line(frameCopy, (int((x0 + x1)/2), y0), (int((x0 + x1)/2), y), (0, 255, 0), thickness=2, lineType=8)
                    #         # Write Car Detected near the bounding box drawn.
                    #         # cv2.putText(frameCopy, 'Car Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
                            
            # Extract the foreground from the frame using the segmented mask.
            # foregroundPart = cv2.bitwise_and(frame, frame, mask=fgmask)
                
            # Stack the original frame, extracted foreground, and annotated frame. 
            # stacked = np.hstack((frame, foregroundPart, frameCopy))

            # Display the stacked image with an appropriate title.
            # cv2.imshow('Original Frame, Extracted Foreground and Detected Cars', cv2.resize(stacked, None, fx=0.25, fy=0.25))
            #cv2.imshow('initial Mask', initialMask)
            #cv2.imshow('Noisy Mask', noisymask)
            #cv2.imshow('Clean Mask', fgmask)

            # cv2.imshow('frameCopy', cv2.resize(frameCopy, None, fx=0.7, fy=0.7))

            # if(isLog != True):   # sau này sẽ bỏ if này đi
            #     if show_mode['all'] or show_mode['holdthing']:
            #         resized = cv2.resize(frameCopy,(640,360))                
            #         cv2.imshow('Detect personHoldThing', resized)
                    
            #     if save_mode['holdthing'] or save_mode['all']:
            #         out_holdthingdetect.write(resized)

            if(isShowImgRes == True):
                resized = cv2.resize(frameCopy,(640,360))                
                cv2.imshow('Detect personHoldThing', resized)
            

            self.returnResultHandle()
            list_dataTracking_result.append(ket_qua_phat_hien_vat_cao)
            if(True):
                for tung_dataTracking_result in list_dataTracking_result:
                    for tung_dtectBox in tung_dataTracking_result.dtectBoxs:
                        debug_tool("box = ", tung_dtectBox.bbox)
            
        
        return list_dataTracking_result


    """
    vidPath: video path
    """
    def getVideoInfo(self, vidPath):
        video = cv2.VideoCapture(r'Z:\2_dataset_vms\2_thuy_dien_Hoa_Binh\2022-10-20 16-00-09 196PTZ OPY_Trim_4.mp4')
        vid_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # debug_tool("vid_height = ", vid_height)
        # debug_tool("vid_width = ", vid_width)

        self.vid_width = vid_width
        self.vid_height = vid_height



class personHoldThing_All:
    def __init__(self):

        # xoa folder luu anh ket qua de tao moi
        dirpath = os.path.join('./', 'test_phuong')
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        
        #khoi tao cam
        camId = 0
        self.personHoldThing_cam_0 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
        
        camId += 1
        self.personHoldThing_cam_1 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
        
        camId += 1
        self.personHoldThing_cam_2 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
        
        camId += 1
        self.personHoldThing_cam_3 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                       
        camId += 1
        self.personHoldThing_cam_4 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_5 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_6 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_7 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_8 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_9 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_10 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_11 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_12 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_13 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_14 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_15 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_16 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_17 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_18 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_19 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_20 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_21 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_22 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_23 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
                
        camId += 1
        self.personHoldThing_cam_24 = personHoldThing(offset_frame_x = dict_para[str(camId)]["offset_frame_x"], offset_frame_y = dict_para[str(camId)]["offset_frame_y"], margin_frame_x =  dict_para[str(camId)]["margin_frame_x"], margin_frame_y = dict_para[str(camId)]["margin_frame_y"] , MOG2_history = dict_para[str(camId)]["MOG2_history"] , MOG2_varThreshold = dict_para[str(camId)]["MOG2_varThreshold"] , do_cao_canh_bao = dict_para[str(camId)]["do_cao_canh_bao"] , videoName = dict_para[str(camId)]["videoName"] , vid_width = dict_para[str(camId)]["vid_width"] , vid_height = dict_para[str(camId)]["vid_height"] , contourAreaThreshold = dict_para[str(camId)]["contourAreaThreshold"] , so_frame_truoc_lien_tiep = dict_para[str(camId)]["so_frame_truoc_lien_tiep"]  )
        
                
        # camId += 1
        # self.personHoldThing_cam_25 = personHoldThing()   
        # self.personHoldThing_cam_26 = personHoldThing()
        # self.personHoldThing_cam_27 = personHoldThing()
        # self.personHoldThing_cam_28 = personHoldThing()
        # self.personHoldThing_cam_29 = personHoldThing()
        # self.personHoldThing_cam_30 = personHoldThing()
        # self.personHoldThing_cam_31 = personHoldThing()
        # self.personHoldThing_cam_32 = personHoldThing()   
        # self.personHoldThing_cam_33 = personHoldThing()
        # self.personHoldThing_cam_34 = personHoldThing()
        # self.personHoldThing_cam_35 = personHoldThing()
        # self.personHoldThing_cam_36 = personHoldThing()
        # self.personHoldThing_cam_37 = personHoldThing()
        # self.personHoldThing_cam_38 = personHoldThing()  
        # self.personHoldThing_cam_39 = personHoldThing()  
        # 

    def detect(self,list_dataTracking_All):
        list_dataTracking_All_result = []

        for dataTracking_All in list_dataTracking_All:
            list_dataTracking = []
            list_dataTracking.append(dataTracking_All)
            dataTracking_perFrame = None

            assert dataTracking_All.cid <= 24,  "Chương trình phát hiện vật thể cao chưa khai báo từ cam thứ 25 trở đi"
            
            
            if(dataTracking_All.cid == 0):
                dataTracking_perFrame = self.personHoldThing_cam_0.detect(list_dataTracking)

            if(dataTracking_All.cid == 1):
                dataTracking_perFrame = self.personHoldThing_cam_1.detect(list_dataTracking)

            if(dataTracking_All.cid == 2):
                dataTracking_perFrame = self.personHoldThing_cam_2.detect(list_dataTracking)

            if(dataTracking_All.cid == 3):
                dataTracking_perFrame = self.personHoldThing_cam_3.detect(list_dataTracking)

            if(dataTracking_All.cid == 4):
                dataTracking_perFrame = self.personHoldThing_cam_4.detect(list_dataTracking)

            if(dataTracking_All.cid == 5):
                dataTracking_perFrame = self.personHoldThing_cam_5.detect(list_dataTracking)

            if(dataTracking_All.cid == 6):
                dataTracking_perFrame = self.personHoldThing_cam_6.detect(list_dataTracking)

            if(dataTracking_All.cid == 7):
                dataTracking_perFrame = self.personHoldThing_cam_7.detect(list_dataTracking)

            if(dataTracking_All.cid == 8):
                dataTracking_perFrame = self.personHoldThing_cam_8.detect(list_dataTracking)

            if(dataTracking_All.cid == 9):
                dataTracking_perFrame = self.personHoldThing_cam_9.detect(list_dataTracking)

            if(dataTracking_All.cid == 10):
                dataTracking_perFrame = self.personHoldThing_cam_10.detect(list_dataTracking)

            if(dataTracking_All.cid == 11):
                dataTracking_perFrame = self.personHoldThing_cam_11.detect(list_dataTracking)

            if(dataTracking_All.cid == 12):
                dataTracking_perFrame = self.personHoldThing_cam_12.detect(list_dataTracking)

            if(dataTracking_All.cid == 13):
                dataTracking_perFrame = self.personHoldThing_cam_13.detect(list_dataTracking)

            if(dataTracking_All.cid == 14):
                dataTracking_perFrame = self.personHoldThing_cam_14.detect(list_dataTracking)

            if(dataTracking_All.cid == 15):
                dataTracking_perFrame = self.personHoldThing_cam_15.detect(list_dataTracking)

            if(dataTracking_All.cid == 16):
                dataTracking_perFrame = self.personHoldThing_cam_16.detect(list_dataTracking)

            if(dataTracking_All.cid == 17): 
                dataTracking_perFrame = self.personHoldThing_cam_17.detect(list_dataTracking)

            if(dataTracking_All.cid == 18):
                dataTracking_perFrame = self.personHoldThing_cam_18.detect(list_dataTracking)

            if(dataTracking_All.cid == 19):
                dataTracking_perFrame = self.personHoldThing_cam_19.detect(list_dataTracking)

            if(dataTracking_All.cid == 20):
                dataTracking_perFrame = self.personHoldThing_cam_20.detect(list_dataTracking)

            if(dataTracking_All.cid == 21):
                dataTracking_perFrame = self.personHoldThing_cam_21.detect(list_dataTracking)

            if(dataTracking_All.cid == 22):
                dataTracking_perFrame = self.personHoldThing_cam_22.detect(list_dataTracking)

            if(dataTracking_All.cid == 23):
                dataTracking_perFrame = self.personHoldThing_cam_23.detect(list_dataTracking)

            if(dataTracking_All.cid == 24):
                dataTracking_perFrame = self.personHoldThing_cam_24.detect(list_dataTracking)

            if(dataTracking_All.cid > 24):
                debug_tool("chưa khai báo từ cam thứ 25 trở đi")

            if(dataTracking_perFrame[0].dtectBoxs): # có vật thể cao mới trả lại dữ liệu dataTracking của frame đó, nếu không thì không trả
                list_dataTracking_All_result.append(dataTracking_perFrame[0])  # mới chỉ dùng index 0 của list, do dữ liệu vào detect là 1 list nhưng chỉ có  1 phần tử




        return list_dataTracking_All_result

    
            
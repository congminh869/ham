"""
lam theo link 
https://bleedai.com/video-contour-detection-101-the-basics-pt1-2-2-2/
"""

import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

#yolov5 ben duoi
from yolov5.detect import run as yolov5_nguoi
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device, smart_inference_mode
import argparse
import os
import glob
import platform
import sys
from pathlib import Path
import torch
from yolov5.utils.general import (CONFIG_DIR, FONT, LOGGER, check_font, check_requirements, clip_boxes, increment_path,
                           is_ascii, xywh2xyxy, xyxy2xywh)
import time

from collections import OrderedDict

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

device = select_device('')
# print("type(device) = ", type(device))
# print("device = ", device)
model = DetectMultiBackend(ROOT / 'yolov5/yolov5s.pt', device=device , dnn=False, data=ROOT / 'yolov5/data/coco128.yaml', fp16=False) # lay theo path root
# model = DetectMultiBackend('E:/phuong/MQ/2_VMS/11_motion_detect/yolov5/yolov5s.pt', device=device , dnn=False, data=  'E:/phuong/MQ/2_VMS/11_motion_detect/yolov5/data/coco128.yaml', fp16=False)






def deleteFile(folderPath):
    for filename in sorted(os.listdir(folderPath))[:-40]:
        filename_relPath = os.path.join(folderPath,filename)
        os.remove(filename_relPath)

def createFolder(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)

    if not isExist:
        
        # Create a new directory because it does not exist 
        os.makedirs(path)
        print("The new directory is created!")

    # xoa file dang co trong thu muc
    files = glob.glob(path + '*')
    for f in files:
        os.remove(f)



"""
do_cao_canh_bao : độ cao so với độ cao người, ví dụ: 0.5 tức là 0.5 * độ cao người
"""
class personHoldThing:
    def __init__(self, offset_frame_x, offset_frame_y, margin_frame_x, margin_frame_y , MOG2_history = 30, MOG2_varThreshold = 15, do_cao_canh_bao = 0.5 , videoName = 0, vid_width = 0, vid_height = 0):
        self.vid_width = vid_width
        self.vid_height = vid_height
        self.offset_frame_x = offset_frame_x
        self.offset_frame_y = offset_frame_y
        self.margin_frame_x = margin_frame_x
        self.margin_frame_y = margin_frame_y
        self.list_toa_do_frame_bo_phan_truoc = [self.vid_width - 1 , self.vid_height - 1, self.vid_width , self.vid_height]  #x0, y0, x1, y1
        self.frameCnt = 0
        self.personCnt = 0
        self.do_cao_canh_bao = do_cao_canh_bao
        
        # You can set custom kernel size if you want.
        self.kernel = None


        # Initialize the background object.
        # self.backgroundObject = cv2.createBackgroundSubtractorMOG2(history=550,varThreshold=15,detectShadows = True)  # goc
        self.backgroundObject = cv2.createBackgroundSubtractorMOG2(history=MOG2_history,varThreshold=MOG2_varThreshold,detectShadows = True)


        self.dict_trang_thai_frame = {}
        self.t1_FPS = time.time()

        # tao thu muc luu data
        # path_base = r'./test/'
        path_base = r'/media/mq/New Volume/personHoldThing/'

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


        

    """
    bboxes: list bbox
    """
    def detect(self,frame,bboxes):



        if(bboxes is None):
            return


        # Create a copy of the frame to draw bounding boxes around the detected cars.
        frameCopy = frame.copy()
        flag_trung_nguoi = False   # flag trùng người


        self.frameCnt += 1

        dict_ds_nguoi = {}
        cnt_nguoi = 0

        offset_docao_2_nguoi = -10

        list_toa_do_x0 = []
        list_toa_do_x1 = []
        list_toa_do_y0 = []
        list_toa_do_y1 = []

        for tung_nguoi in bboxes:
            tung_nguoi = torch.tensor(tung_nguoi).view(-1, 4)
            b = xyxy2xywh(tung_nguoi)  # boxes

            tung_nguoi = xywh2xyxy(b).long()
            # print("tung_nguoi = ", tung_nguoi)

            x0 = int(tung_nguoi[0, 0])
            y0 = int(tung_nguoi[0, 1])
            x1 = int(tung_nguoi[0, 2])
            y1 = int(tung_nguoi[0, 3])
            # print(x0, y0 ,x1, y1)
            list_toa_do_x0.append(x0)
            list_toa_do_x1.append(x1)
            list_toa_do_y0.append(y0)
            list_toa_do_y1.append(y1)

            nguong_do_cao = y0 - int( (y1-y0)/4 )
            nguong_do_cao_up = y0 - int( (y1-y0) * self.do_cao_canh_bao )
            if(nguong_do_cao < 0):
                nguong_do_cao = 0
            if(nguong_do_cao_up < 0):
                nguong_do_cao_up = 0

            khuvuc_detect_toado = [(x0, nguong_do_cao),(x0, nguong_do_cao_up),(x1, nguong_do_cao_up),(x1, nguong_do_cao)]
            khuvuc_detect_polygon = cv2.polylines(frameCopy, [np.array(khuvuc_detect_toado, np.int32)], True, (15,220,10), thickness=1)

            # ve nguoi
            # cv2.polylines(frameCopy, [np.array([(x0,y0 + offset_docao_2_nguoi), (x1, y0 + offset_docao_2_nguoi), (x1, y1 + offset_docao_2_nguoi), (x0, y1 + offset_docao_2_nguoi)], np.int32)], True, (255,0,0), thickness=1)
            cv2.rectangle(frameCopy, (x0,y0 + offset_docao_2_nguoi), (x1, y1),(255, 0, 0), 1)


            dict_ds_nguoi[cnt_nguoi] = [[x0, y0 ,x1, y1], khuvuc_detect_toado]
            

            # print("ds nguoi = ", dict_ds_nguoi[cnt_nguoi] ) 

            cnt_nguoi += 1
            self.personCnt += 1

        # print("dict nguoi = ", dict_ds_nguoi)
        # print("list_toa_do_x0 = ", list_toa_do_x0)
        # print("list_toa_do_x1 = ", list_toa_do_x1)
        # print("list_toa_do_y0 = ", list_toa_do_y0)
        # print("list_toa_do_y1 = ", list_toa_do_y1)
        # print("min x0 = ", min(list_toa_do_x0))
        # print("max x1 = ", max(list_toa_do_x1))

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



        # if(frame_bodem  == 10 or (frame_bodem % 100 == 0) ):
        # if(frame_bodem  == 10  ):
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
            frame_bophan_x0 = self.list_toa_do_frame_bo_phan_truoc[0]
            frame_bophan_x1 = self.list_toa_do_frame_bo_phan_truoc[2]
            frame_bophan_y0 = self.list_toa_do_frame_bo_phan_truoc[1]
            frame_bophan_y1 = self.list_toa_do_frame_bo_phan_truoc[3]

            # frame_bophan_x0 = 200
            # frame_bophan_x1 = 400
            # frame_bophan_y0 = 200
            # frame_bophan_y1 = 400

            # frame_bophan_x0 = 0
            # frame_bophan_x1 = vid_width
            # frame_bophan_y0 = 0
            # frame_bophan_y1 = vid_height


        frame_bophan = frame[ frame_bophan_y0:frame_bophan_y1, frame_bophan_x0:frame_bophan_x1]
        cv2.imshow("frame_bophan", frame_bophan)
        cv2.imwrite(self.path_frame_bophan + str(frame_bodem) + ".jpg", frame_bophan)


        # Apply the background object on the frame to get the segmented mask.
        t1 = time.time()
        fgmask = self.backgroundObject.apply(frame_bophan)

        #initialMask = fgmask.copy()


        # Perform thresholding to get rid of the shadows.
        _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)  # goc
        t2 = time.time()
        print("thoi gian tru nen" , t2 -t1)
        #noisymask = fgmask.copy()

        cv2.imshow("fgmask", cv2.resize(fgmask, None, fx=0.7, fy=0.7))
        cv2.imwrite(self.path_frame_mask + str(frame_bodem) + ".jpg", fgmask) 
                
        # Apply some morphological operations to make sure you have a good mask
        fgmask = cv2.erode(fgmask, self.kernel, iterations = 1)
        fgmask = cv2.dilate(fgmask, self.kernel, iterations = 2)


        data = OrderedDict(dict_ds_nguoi)
        # print(data.keys())
        # data.move_to_end(2)
        # print(data.keys())

        data_2 = data.copy()

        for index, (key, value) in enumerate(data.items()):
            # pass
            # data goc
            # print(index, key, value)
            # 0 0 [[1538, 878, 1608, 1077], [(1538, 829), (1538, 779), (1608, 779), (1608, 829)]]
            # 1 1 [[1478, 916, 1553, 1077], [(1478, 876), (1478, 836), (1553, 836), (1553, 876)]]
            # 2 2 [[1634, 812, 1728, 1072], [(1634, 747), (1634, 682), (1728, 682), (1728, 747)]] 


            # data 2
            data_2.move_to_end(key)
            # print("data_2 = ", data_2)




            khuvuc_detect_toado = value[1]
            # print("type(khuvuc_detect_toado) = ", type(khuvuc_detect_toado))
            # print("(khuvuc_detect_toado) = ", khuvuc_detect_toado)
            x0_chinh = value[0][0]
            y0_chinh = value[0][1]
            x1_chinh = value[0][2]
            y1_chinh = value[0][3]

            for index, (key, value) in enumerate(data_2.items()):
                if(index == (len(data_2) -1 ) ):
                    break

                x0 = value[0][0]
                y0 = value[0][1]
                x1 = value[0][2]
                y1 = value[0][3]

                # print(x0,x1,y0,y1)

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

                # print("result_trung_nguoi = ", result_trung_nguoi)
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
                    
                    cv2.imwrite(self.path_frame_trung_nguoi + str(frame_bodem) + ".jpg", frameCopy) 

                    flag_trung_nguoi = True

                    break

            if(flag_trung_nguoi == True):
                break

        if(flag_trung_nguoi == True):
            cv2.imshow('frameCopy', cv2.resize(frameCopy, None, fx=0.7, fy=0.7))
            # self.dict_trang_thai_frame[frame_bodem] = False # chưa để False, vì trùng người thì vẫn có khả năng là có vật cao
            return
                            
        # Detect contours in the frame.
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if(len(self.dict_trang_thai_frame) <= 2):
            trang_thai_frame_truoc = False
        else:
            trang_thai_frame_truoc = self.dict_trang_thai_frame[list(self.dict_trang_thai_frame)[-1]] and self.dict_trang_thai_frame[list(self.dict_trang_thai_frame)[-2]]

        # print("trang_thai_frame_truoc", trang_thai_frame_truoc)
        self.dict_trang_thai_frame[frame_bodem] = False

        # loop over each contour found in the frame.
        for cnt in contours:       
            # Make sure the contour area is somewhat higher than some threshold to make sure its a car and not some noise.
            if cv2.contourArea(cnt) > 50:
                
                # Retrieve the bounding box coordinates from the contour.
                x, y, width, height = cv2.boundingRect(cnt)

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
                    

                    # if(True):
                    if(trang_thai_frame_truoc == True):
                        cv2.putText(frameCopy, 'co vat the', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                        cv2.imwrite(self.path_frame_ketluan + str(frame_bodem) + ".jpg", frameCopy)

                    self.dict_trang_thai_frame[frame_bodem] =  True
                    cv2.imwrite(self.path_frame_chua_ketluan + str(frame_bodem) + ".jpg", frameCopy)
                                        


                
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
        cv2.imshow('frameCopy', cv2.resize(frameCopy, None, fx=0.7, fy=0.7))

        # tinh FPS
        if(self.frameCnt > 10):
            t2_FPS = time.time()
            print("FPS = ", self.frameCnt/(t2_FPS-self.t1_FPS))   
            print("Person per second = ", self.personCnt/(t2_FPS-self.t1_FPS))   

            self.frameCnt = 0  
            self.personCnt = 0  
            self.t1_FPS =    t2_FPS

            #Xoa file trong thu muc
            deleteFile(self.path_frame_chua_ketluan)
            deleteFile(self.path_frame_ketluan)
            deleteFile(self.path_frame_mask)
            deleteFile(self.path_frame_trung_nguoi)
            deleteFile(self.path_frame_bophan)


    """
    vidPath: video path
    """
    def getVideoInfo(self, vidPath):
        video = cv2.VideoCapture(r'Z:\2_dataset_vms\2_thuy_dien_Hoa_Binh\2022-10-20 16-00-09 196PTZ OPY_Trim_4.mp4')
        vid_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("vid_height = ", vid_height)
        print("vid_width = ", vid_width)

        self.vid_width = vid_width
        self.vid_height = vid_height


if __name__ == "__main__":
    # load a video
    # videoPath = r'Z:\2_dataset_vms\2_thuy_dien_Hoa_Binh\2022-10-20 16-00-09 196PTZ OPY_Trim_4.mp4'
    # videoPath = r'\\192.168.6.124\thuannd\VMS_HOABINH\VMS_BASE1\backupHB\NS1066-3\2022-10-19_17-00-36_135Gian_may_H1_.mp4'
    # videoPath = r'Z:/2_dataset_vms/2_thuy_dien_Hoa_Binh/2022-10-19_09-00-04.mp4'
    # videoPath = "E:/phuong/MQ/2_VMS/11_motion_detect/data_test/camco_1.mp4"
    # videoPath = r"E:\phuong\MQ\2_VMS\11_motion_detect_2\background_subtration\data_test/dongtaccamco.mp4"
    # videoPath = "E:/phuong/MQ/2_VMS/11_motion_detect/data_test/2022-10-19_09-00-04_Trim_2.mp4"
    videoPath = r'Z:\2_dataset_vms\2_thuy_dien_Hoa_Binh\2022-10-20 16-00-09 196PTZ OPY_Trim_4.mp4'
    

        
    video = cv2.VideoCapture(videoPath)
    frame_bodem = 0
    
    


    # khoi tao
    phat_hien_vat_the_video_1 = personHoldThing(200,200,0,0)

    # lay thong tin video
    phat_hien_vat_the_video_1.getVideoInfo(videoPath)


    while True:
        
        # Read a new frame.
        ret, frame = video.read()
        frame_bodem += 1

        # Check if frame is not read correctly.
        if not ret:
            
            # Break the loop.

            break

        if(frame_bodem % 10 == 0):
            
            
            # luu anh
            list_box = []
            cv2.imwrite(ROOT / "1.jpg", frame)
            list_box = yolov5_nguoi(model=model)

            phat_hien_vat_the_video_1.detect(frame,list_box)

    
            # Wait until a key is pressed.
            # Retreive the ASCII code of the key pressed
            k = cv2.waitKey(1) & 0xff
            
            # Check if 'q' key is pressed.
            if k == ord('q'):
                
                # Break the loop.
                break

    # Release the VideoCapture Object.
    video.release()

    # Close the windows.q
    cv2.destroyAllWindows()
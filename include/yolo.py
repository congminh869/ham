import cv2
import torch
from PIL import Image
import time

import argparse
import os
import platform
import sys
import glob
from pathlib import Path
import copy

#debug log
from inspect import currentframe, getframeinfo
import datetime

# from helmet_processes import to_batches, helmet_scaling # import processes helmets
# from CheckViolate import CheckViolate

debug_log = False

def Debug_log(cf, filename, name = None):
    if debug_log:
        ct = datetime.datetime.now()
        print(f'================ [{ct}] file {filename} , line : {cf.f_lineno} {name}')

#https://github.com/ultralytics/yolov5/issues/36
class DtectBox:
    def __init__(self):
        self.bbox = None # [x1, y1, x2, y2]
        self.name_class = None # string
        self.id_tracking = None #int
        self.class_conf = None  # float
        self.ComeIn = None
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



class YOLOv562:
    def __init__(self, weights, classes=None, device=0, iou_thres = 0.4, conf_thres = 0.5, img_size = 640, max_det=1000, agnostic_nms=False, engine = False):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        #load model
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, os.path.exists(weights))

        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, verbose=True)
        
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        #config model
        self.model.conf = conf_thres  # NMS confidence threshold
        self.model.iou = iou_thres  # NMS IoU threshold
        self.model.agnostic = agnostic_nms  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.classes = classes  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        self.model.max_det = max_det  # maximum number of detections per image
        self.model.amp = False  # Automatic Mixed Precision (AMP) inference
        print(self.model.names)

        #check tensorrt
        self.engine = engine

        #config gpu
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            self.device=torch.device(device)
            self.model.to(self.device)
        else:
            self.model.cpu()
        
        self.dataTrackings = []
        self.img_size = img_size

        self.palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        # print('*****************************')
        # print(self.model.__getattribute__)

    def detect(self, dataTrackings):
        if self.engine:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            arr_im = []
            dict_scale = []
            for i, dataTracking in enumerate(dataTrackings):
                image = copy.deepcopy(dataTracking.frame)
                resized_image = cv2.resize(image, (640, 640), interpolation = cv2.INTER_AREA) 
                h, w, c = image.shape
                w_scale = self.img_size/w
                h_scale = self.img_size/h
                dict_scale.append([w_scale, h_scale])

                arr_im.append(resized_image)


            # print('len(arr_im) : ', len(arr_im))
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            results = self.model(arr_im, size=self.img_size)
            # print(results)

            self.results = results

            for idx_frame, result in enumerate(results.pandas().xyxy):
                dtectBoxs = []
                for idx_object, pre in enumerate(result.values.tolist()):
                    # print('pre : ', pre)
                    x1, y1, x2, y2 = int(pre[0]/dict_scale[idx_frame][0]), int(pre[1]/dict_scale[idx_frame][1]), int(pre[2]/dict_scale[idx_frame][0]), int(pre[3]/dict_scale[idx_frame][1])
                    # print('x1, y1, x2, y2 : ', x1, y1, x2, y2)
                    conf = pre[4]
                    cls = pre[5]
                    label = pre[6]
                    dtectBox = DtectBox()
                    dtectBox.bbox = [x1, y1, x2, y2]
                    dtectBox.name_class = label
                    dtectBox.class_conf = conf 
                    dtectBoxs.append(dtectBox)
                    # cv2.rectangle(arr_im[i], (x1, y1), (x2, y2), (255,255,0), 2)
                    # cv2.imshow(str(i), arr_im[i])
                    # cv2.waitKey(0)
                dataTrackings[idx_frame].dtectBoxs = dtectBoxs
                # print('----')
        else:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            arr_im = []
            for i, dataTracking in enumerate(dataTrackings):
                # frame = cv2.cvtColor(dataTracking.frame, cv2.COLOR_BGR2RGB)
                # arr_im.append(cv2.cvtColor(dataTracking.frame, cv2.COLOR_BGR2RGB))
                arr_im.append(dataTracking.frame)
                # cv2.imshow(str(i), image)
                # cv2.imwrite(f'./output/{str(i)}_yolo.jpg', dataTracking.frame)

            # for i in arr_im:
            #     cv2.imshow('image_arr_im', i)
            # print('len(arr_im) : ', len(arr_im))
            Debug_log(currentframe(), getframeinfo(currentframe()).filename,'len(arr_im) : '+ str(len(arr_im)))
            results = self.model(arr_im, size=self.img_size)
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            # print(results)

            self.results = results

            for idx_frame, result in enumerate(results.pandas().xyxy):
                dtectBoxs = []
                for idx_object, pre in enumerate(result.values.tolist()):
                    # print('pre : ', pre)
                    x1, y1, x2, y2 = int(pre[0]), int(pre[1]), int(pre[2]), int(pre[3])
                    # print('x1, y1, x2, y2 : ', x1, y1, x2, y2)
                    conf = pre[4]
                    cls = pre[5]
                    label = pre[6]
                    dtectBox = DtectBox()
                    dtectBox.bbox = [x1, y1, x2, y2]
                    dtectBox.name_class = label
                    dtectBox.class_conf = conf 
                    dtectBoxs.append(dtectBox)
                    # cv2.rectangle(arr_im[idx_frame], (x1, y1), (x2, y2), (255,255,0), 2)
                    # cv2.putText(arr_im[idx_frame], label + '_' + str(conf), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
                    # cv2.imwrite(f'./event_fence/{str(i)}.jpg', arr_im[i])
                    # cv2.imshow(str(i), arr_im[i])
                    # cv2.waitKey(1)
                dataTrackings[idx_frame].dtectBoxs = dtectBoxs
                # print('*******************************************************************************************')
                # cv2.imshow('check_yolo',dataTrackings[idx_frame].frame)
                # cv2.waitKey(0)
                # print('----')
        return dataTrackings

    def detect_RGB(self, dataTrackings):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        arr_im = []
        for i, dataTracking in enumerate(dataTrackings):
            # frame = cv2.cvtColor(dataTracking.frame, cv2.COLOR_BGR2RGB)
            # arr_im.append(cv2.cvtColor(dataTracking.frame, cv2.COLOR_BGR2RGB))
            arr_im.append(cv2.cvtColor(dataTracking.frame, cv2.COLOR_BGR2RGB))
            # cv2.imshow(str(i), image)
            # cv2.imwrite(f'./output/{str(i)}_yolo.jpg', dataTracking.frame)

        for i in arr_im:
            cv2.imshow('image_arr_im_rgb', i)
        print('len(arr_im) : ', len(arr_im))
        Debug_log(currentframe(), getframeinfo(currentframe()).filename,'len(arr_im) : '+ str(len(arr_im)))
        results = self.model(arr_im, size=self.img_size)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        # print(results)

        self.results = results

        for idx_frame, result in enumerate(results.pandas().xyxy):
            dtectBoxs = []
            for idx_object, pre in enumerate(result.values.tolist()):
                # print('pre : ', pre)
                x1, y1, x2, y2 = int(pre[0]), int(pre[1]), int(pre[2]), int(pre[3])
                # print('x1, y1, x2, y2 : ', x1, y1, x2, y2)
                conf = pre[4]
                cls = pre[5]
                label = pre[6]
                dtectBox = DtectBox()
                dtectBox.bbox = [x1, y1, x2, y2]
                dtectBox.name_class = label
                dtectBox.class_conf = conf 
                dtectBoxs.append(dtectBox)
                cv2.rectangle(arr_im[idx_frame], (x1, y1), (x2, y2), (255,255,0), 2)
                cv2.putText(arr_im[idx_frame], label + '_' + str(conf), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
                # cv2.imwrite(f'./event_fence/{str(i)}.jpg', arr_im[i])
                # cv2.imshow(str(i), arr_im[i])
                # cv2.waitKey(1)
            dataTrackings[idx_frame].dtectBoxs = dtectBoxs
            # print('*******************************************************************************************')
            cv2.imshow('check_yolo_RGB',dataTrackings[idx_frame].frame)
            if cv2.waitKey(0) == 27:
                sys.exit()
            # print('----')
        return dataTrackings

class YOLOv5Src:
    def __init__():
        pass 
    def detect():
        pass


if __name__ == '__main__':
    # Model
    weights_engine = '/home/mq/Documents/AI_hoabinh/yolov5/yolov5s.engine'
    weights_pt = '/home/mq/Documents/AI_hoabinh/minh/weight/person/crowdhuman_yolov5m.pt'
    

    # model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
    
    # model.conf = 0.25  # NMS confidence threshold
    # model.iou = 0.45  # NMS IoU threshold
    # model.agnostic = False  # NMS class-agnostic
    # model.multi_label = False  # NMS multiple labels per box
    # model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    # model.max_det = 1000  # maximum number of detections per image
    # model.amp = False  # Automatic Mixed Precision (AMP) inference

    # use_gpu = torch.cuda.is_available()
    # if use_gpu:
    #     device=torch.device(0)
    #     model.to(device)
    # else:
    #     model.cpu()
    weights_person = '/home/evnadmin/Documents/AI_hoabinh/weight/person/crowdhuman_yolov5m.pt'
    classes=0
    device=0
    iou_thres=0.4
    conf_thres=0.5
    img_size=640
    max_det=1000
    agnostic_nms=False
    yolo = YOLOv562(weights_person, 
                    classes=classes, 
                    device=device, 
                    iou_thres = iou_thres, 
                    conf_thres = conf_thres, 
                    img_size = img_size, 
                    max_det=max_det, 
                    agnostic_nms=agnostic_nms)
    # dataTrackings = []
    # # # Images
    # # # for f in 'zidane.jpg',/"d_url_to_file('https://ultralytics.com/images/' + f, f):  # download 2 images
    # im1 = Image.open('zidane.jpg')  # PIL image
    # im2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
    # ims = []
    # page_img = cv2.resize(im2, (640,640), interpolation = cv2.INTER_AREA)
    # for i in range(0, 20):
    #     ims.append(page_img)

    # # # Inference
    # # a = [im1]
    # # print(len(a))
    # # # for i in range(0, 10):
    # # i = 1
    # dtectBoxs = []
    # # cid = i
    # # type_id = i
    # # count = i

    # # im3 = cv2.imread('00040001.jpg')

    # dataTrackings = [DataTracking(im, dtectBoxs, idx, idx, idx) for idx, im in enumerate(ims)]
    # while True:
        

    #     # dataTrackings.append(DataTracking(im2, dtectBoxs, 1, 2, 2))
    #     # dataTrackings.append(DataTracking(im1, dtectBoxs, 2, 3, 3))

    #     t1 = time.time()

    #     dataTrackings = yolo.detect(dataTrackings)
    #     t2 = time.time()
    #     print('time inferen : ', t2 - t1)



    #     dtects, in_rects_preds = helmet_scaling(yolo, round_rects, num_grids_each_rect) # mapping

    #     # print(in_rects_preds)
    #     dataTrackings = [DataTracking(im3[...,::-1], dtects, cid, type_id, count)]
    #     # print('***********************************')

    #     a = check.runcheck(in_rects_preds, [[0], [0]])
    #     # print(a)



    # # Results
    # # results.print()  
    # # print(results.xyxy )
    # # results.save()  # or .show()

    # # results.xyxy[0]  # im1 predictions (tensor)
    # # results.pandas().xyxy[0]  # im1 predictions (pandas)
    # #      xmin    ymin    xmax   ymax  confidence  class    name
    # # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
    # # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

    # filename_video = '/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/fire1.mp4'

    # cap = cv2.VideoCapture(filename_video)

    # while True:
    #     ret, frame = cap.read()
    #     if ret!=True:
    #         continue
    #     dataTrackings_person = []
    #     dtectBoxs_person = []
    #     cid = 1
    #     type_id = 1
    #     count = 1
    #     dataTrackings_person.append(DataTracking(frame, dtectBoxs_person, cid, type_id, count))

    #     a = yolo.detect(dataTrackings_person)
    #     b = yolo.detect_RGB(dataTrackings_person)



    list_file = []
    folder_src = "/home/evnadmin/Documents/AI_hoabinh/video/image"
    print('folder_src = ',folder_src)
    for root, dirs, files in os.walk(folder_src):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                list_file.append(folder_src + "/" + file)
                # print('\n==========> file = ',file)
    
    
    for file in list_file:
                    
        frame = cv2.imread(file)

        dataTrackings_person = []
        dtectBoxs_person = []
        cid = 1
        type_id = 1
        count = 1
        dataTrackings_person.append(DataTracking(frame, dtectBoxs_person, cid, type_id, count))

        a = yolo.detect(dataTrackings_person)
        b = yolo.detect_RGB(dataTrackings_person)


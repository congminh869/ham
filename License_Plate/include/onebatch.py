#debug log
from inspect import currentframe, getframeinfo
import datetime
import time
import copy

debug_log = False 
save_img = True
def Debug_log(cf, filename, name = None):
    if debug_log:
        ct = datetime.datetime.now()
        print(f'[{ct}] file {filename} , line : {cf.f_lineno} {name}')
        
from License_Plate.include.models_retina import model_retina, ModelRetinaCpp

Debug_log(currentframe(), getframeinfo(currentframe()).filename)
from License_Plate.include.model_PaddleOCR import PaddleOcR_Reg
Debug_log(currentframe(), getframeinfo(currentframe()).filename)
import torch
import os
import cv2
import numpy as np
import time
import glob
import sys 

import datetime

from include.include_main import crop_img_polygon
Debug_log(currentframe(), getframeinfo(currentframe()).filename)
# import threading
# lock = threading.Lock()

IS_VISUALIZE = False
SAVE_VIDEO = False
SAVE_IMG = False
Debug_log(currentframe(), getframeinfo(currentframe()).filename)

class License_Plate():
    def __init__(self):
        self.count_plate_per_id = {}
        self.buffer_plate_per_id = {}
        self.lp_rets = []
        self.count_id = 0
        self.buffer_reid = {}

        Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'time sleep model paddle')
        self.reg_plate = PaddleOcR_Reg()
        self.model_retina = ModelRetinaCpp()#model_retina()
        # time.sleep(100)

        name_folder = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")
        self.video = cv2.VideoWriter(f'./event_plate/{name_folder}.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         5, (1920, 1080))

    def detect(self, dataImages, coordinate_rois, dict_data):
        '''
        detect_reID
        dataImages: len(dataImages) = 1
        dict_data = {"cid": {"img": image, "data": data},
            data = [{id: id, “lpNumber”: “84A89563” }, {id: id, “lpNumber”: “84A89563” }, {id: id, “lpNumber”: “84A89563” }...]
        
            lp_rest : {'id': id, 'txt':lp_ret, 'box_kps': box_kps, 'frame': frame, 'time': now.strftime("%Y-%m-%d_%H-%M-%S")}
        '''
        #dict_data = {}
        # print('==============================check detect===================')
        
        # self.video.write(cv2.resize(dataImages[0].image, (1920,1080), interpolation = cv2.INTER_AREA)) 
        self.buffer_reid, dict_data, self.count_id, lp_rets_dict_data = main_reid(dataImages, 
                            coordinate_rois, 
                            self.reg_plate,
                            self.model_retina,
                            self.buffer_reid,
                            self.count_plate_per_id, 
                            self.buffer_plate_per_id, 
                            self.lp_rets,
                            self.count_id)
        # print('self.buffer_reid : ', self.buffer_reid)

        if len(lp_rets_dict_data) > 0:
            dict = {}
            datas = []      
            dict_data = {}
            str_show = ''
            for lp in lp_rets_dict_data:
                data = {}
                str_show =str_show + str(lp['id'])+"_"+lp['txt']+"_"
                # print("lp['box_kps'] : ", lp['box_kps'])
                # print(lp['frame'].shape)
                cv2.rectangle(lp['frame'], (lp['box_kps'][0], lp['box_kps'][1]), (lp['box_kps'][2], lp['box_kps'][3]), (0, 0, 255), 2)
                cv2.putText(lp['frame'],str(lp['id'])+"_"+lp['txt'],(lp['box_kps'][0], lp['box_kps'][1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4,cv2.LINE_AA)
                # resized = cv2.resize(lp['frame'], (int(lp['frame'].shape[1]/3),int(lp['frame'].shape[0]/3)), interpolation = cv2.INTER_AREA)             
                # cv2.imshow(str_show, resized)
                # cv2.imwrite(str_show+".jpg",resized)
                # time.sleep(5)
                dict['img'] = lp['frame']
                
                data['id'] = lp['id']
                data['lpNumber'] = lp['txt']
                
                datas.append(data)
                name_folder = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")
                cv2.imwrite(f'./event_plate/data_event/biso_{str_show}_{name_folder}.jpg', lp['frame'])
            #     cv2.imshow(str_show, resized)  
            # if cv2.waitKey(0) == 27:
            #     sys.exit()

            dict['data'] = datas 
            dict_data[dataImages[0].cid] = dict

        return dict_data

    def detectLiveStr(self, dataImages, coordinate_rois, dict_data):
        self.buffer_reid, dict_data, self.count_id, lp_rets_dict_data, image_detect = LiveStream(dataImages, 
                            coordinate_rois, 
                            self.reg_plate,
                            self.model_retina,
                            self.buffer_reid,
                            self.count_plate_per_id, 
                            self.buffer_plate_per_id, 
                            self.lp_rets,
                            self.count_id)

def draw_img_detect(results, frame):
    for result in results:
        x1, y1, x2, y2 = result[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return frame

def draw_img(results, frame):
    global out
    # try:
    if True:
        for result in results:
            # try:
            if True:
                id = result[0]
                txt_result = str(id) +'-'+ result[1]
                x1, y1, x2, y2 = result[2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                cx = x1
                cy = y1 - 12
                cv2.putText(frame, txt_result, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        if IS_VISUALIZE:
            # Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            window_name = 'frame'
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
        if SAVE_VIDEO:
            out.write(frame)
        if SAVE_IMG:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            name = './out_img/'+str(time.time()).replace('.','_')+'.jpg'
            cv2.imwrite(name, frame)
    return frame
    
def mergerLP(list_lp_old):
    '''
        list_lp_old: a list results = [id, txt_result, [x1, y1, x2, y2], cropped_img, kps, img_raw] of the same number plate
    '''
    # print('======================mergerLP=========================')
    list_lp = []
    # print('list_lp_old : ', list_lp_old)
    index_frame = int(len(list_lp_old)/2)
    frame = list_lp_old[index_frame][-1]
    box_kps = list_lp_old[index_frame][2]
    # print()
    # Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    for i in range(0, len(list_lp_old)):
        list_lp.append(list(list_lp_old[i][1]))
    #     print(f'list_lp_old[i][i] id : {list_lp_old[i][0]}, txt_result: {list_lp_old[i][1]}, box: {list_lp_old[i][2]}')
    # Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    # print('list_lp : ', list_lp)
    # print('list_lp[0] : ', list_lp[0])
    lp_ret = ''
    for i in range(9):
        list_char = []

        #Fill char to list_char
        for j in range(len(list_lp)):
            ch = list_lp[j][i]
            f_existed = False
            for k in list_char:
                if ch == k["char"]:
                    f_existed = True
                    k["count"] += 1
            if f_existed is False:
                new = {
                    "char" : ch,
                    "count" : 1
                }
                list_char.append(new)

        #Find max count 
        max = 0
        ch_max = 0
        for k in list_char:
            if max < k["count"]:
                ch_max = k["char"]
                max = k["count"]                
        lp_ret += ch_max

    lp_ret = lp_ret.replace('#','')
    # print('lp_ret : ', lp_ret)
    id = list_lp_old[0][0]

    now = datetime.datetime.now()
            # print(now.strftime("%Y-%m-%d %H:%M:%S"))
    return {'id': id, 'txt':lp_ret, 'box_kps': box_kps, 'frame': frame, 'time': now.strftime("%Y-%m-%d_%H-%M-%S")} #now.strftime("%Y-%m-%d %H:%M:%S") 
    # return {'id': id, 'txt':lp_ret, 'time': now.strftime("%Y-%m-%d %H:%M:%S")}


def main(dataImages, coordinate_rois, reg_plate, model_retina, count_plate_per_id = {}, buffer_plate_per_id = {}, lp_rets = []):
    #processing number plate results
    # count_plate_per_id = {} #{'id' : count}
    # buffer_plate_per_id = {} #{'id', []}
    max_buffer_per_id = 5 #buffer_plate_per_id
    len_buffer_time = 10 #count_plate_per_id: after 10 frame delete id
    lp_rets_dict_data = []
    # lp_rets = []
    x1, y1, x2, y2 = coordinate_rois[dataImages[0].cid]
    print('x1, y1, x2, y2 :', x1, y1, x2, y2)
    print('type(x1) : ', type(x1))
    name_folder = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")
    # cv2.imwrite(f'./event_plate/image_in.jpg', dataImages[0].image)
    cv2.imshow('image_in', cv2.resize(dataImages[0].image, (int(dataImages[0].image.shape[1]/4), int(dataImages[0].image.shape[0]/4))) )
    frame = dataImages[0].image.copy()[y1:y2, x1:x2]
    # cv2.imwrite('./event_plate/image_out_crop.jpg', dataImages[0].image.copy()[y1:y2, x1:x2])
    # cv2.imshow('src_', cv2.resize(frame, (int(frame.shape[1]/3),int(frame.shape[0]/3)), interpolation = cv2.INTER_AREA))
    time_total1 = time.time()
    t_detect_plate1 = time.time()
    # print('time sleep 1')
    # time.sleep(10)
    # print('time sleep 2')
    dets_plate = model_retina.detect_plate(frame, file_img = False) #[[x1, y1, x2, y2], box_kps, id, conf, img_raw]
    if len(dets_plate)>0:
        for det_plate in dets_plate:
            x1_, y1_, x2_, y2_ = det_plate[0]
            box_kps = det_plate[1]
            cv2.rectangle(frame, (x1_, y1_), (x2_, y2_), (255, 0, 255), 2)
            cv2.circle(frame, (box_kps[0], box_kps[1]), 1, (0, 0, 255), 4)
            cv2.circle(frame, (box_kps[2], box_kps[3]), 1, (0, 255, 255), 4)
            cv2.circle(frame, (box_kps[4], box_kps[5]), 1, (255, 0, 255), 4)
            cv2.circle(frame, (box_kps[6], box_kps[7]), 1, (0, 255, 0), 4)
            cv2.circle(frame, (box_kps[8], box_kps[9]), 1, (255, 0, 0), 4)
        # resized = cv2.resize(frame, (int(frame.shape[1]/2),int(frame.shape[0]/2)), interpolation = cv2.INTER_AREA)
    
        # cv2.imwrite(f'./event_plate/dets_plate{name_folder}.jpg',frame)
        # cv2.imshow('dets_plate', resized)

    # time.sleep(100)
    # Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    t_detect_plate2 = time.time()
    # print('time detect : ', t_detect_plate2 - t_detect_plate1)

    t_reg1=time.time()
    #[id, txt_result, [x1, y1, x2, y2], cropped_img, kps, img_raw]
    results = reg_plate.reg_plate(frame, dets_plate) #results.append([id, txt_result, [x1, y1, x2, y2], cropped_img, kps, check_plate, conf])
    #draw_img(results, frame.copy())  
    t_reg2=time.time()
    # print('time reg : ', t_reg2 - t_reg1)

    #processing number plate results
    #check new plate or old plate
    for result in results:
        del result[3]
        txt_result = result[1]
        id = result[0]
        if id in buffer_plate_per_id:
            buffer_plate_per_id[id].append(result)
        else:
            buffer_plate_per_id[id] = [result]
            count_plate_per_id[id] = len_buffer_time

        #delete id and mergerLP if count_plate_per_id <=0
        #count_plate_per_id : count number  of per id plate count plate id follow frame, count decrease, plate may or may not appear
        key_deletes = []

        for key in count_plate_per_id:
            if count_plate_per_id[key]<=0: # nếu quá n frame ko xuất hiện lại 
                check_Duplicate_id = False
                lp_ret = mergerLP(buffer_plate_per_id[key])
                for i in range(len(lp_rets)):
                    if lp_ret['id'] == lp_rets[i]['id']:
                        check_Duplicate_id = True
                if check_Duplicate_id == False:
                    #send plate
                    lp_rets.append(lp_ret)
                    lp_rets_dict_data.append(lp_ret)
                key_deletes.append(key)
        
        #delete plate
        for key in key_deletes:
            del count_plate_per_id[key]
            del buffer_plate_per_id[key]

        key_deletes = []
        #delete and mergerLP if buffer_plate_per_id[key]>=max_buffer_per_id
        for key in buffer_plate_per_id:
            if len(buffer_plate_per_id[key])>=max_buffer_per_id:
                check_Duplicate_id = False
                lp_ret = mergerLP(buffer_plate_per_id[key])
                for i in range(len(lp_rets)): # nếu giống id thì ko add thêm 
                    if lp_ret['id'] == lp_rets[i]['id']:
                        check_Duplicate_id = True
                if check_Duplicate_id == False:
                    #send plate
                    lp_rets.append(lp_ret)
                    lp_rets_dict_data.append(lp_ret)
                key_deletes.append(key) # quá số frame cho phép reset 
        #delete plate
        for key in key_deletes:
            del count_plate_per_id[key]
            del buffer_plate_per_id[key]

        for key in count_plate_per_id:
            count_plate_per_id[key]-=1

    #buffer results < = 20 elements
    if len(lp_rets)>=20:
        for i in range(int(len(lp_rets)/2)):
            lp_rets.pop(0)

    # Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    # print(buffer_plate_per_id)
    # print(type(buffer_plate_per_id))
    # for key in buffer_plate_per_id:
    #     print('key : ', list(buffer_plate_per_id.keys()))
    #     for i in range(len(buffer_plate_per_id[key])):
    #         print('buffer_plate_per_id[key][i][:3] : ', buffer_plate_per_id[key][i][:3])
    #     print('--')
    # Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    #########################

    # show
    if debug_log:
        frame = draw_img(results, frame)  
        cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, frame.shape[0]-10), (0, 0, 255), 2)
        dataImages[0].image[y1:y2, x1:x2] = frame
        frame_mapping = dataImages[0].image
        name_folder = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")
        cv2.imwrite(f'./event/biso_{name_folder}.jpg', frame_mapping)
        # resized = cv2.resize(frame_mapping, (int(frame_mapping.shape[1]/2),int(frame_mapping.shape[0]/2)), interpolation = cv2.INTER_AREA)
        # cv2.imshow('frame_mapping', resized)
        # if cv2.waitKey(0)==27:
        #     sys.exit()
        # print('results : ', results)
    else:
        frame = draw_img(results, frame)  
        cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, frame.shape[0]-10), (0, 0, 255), 2)
        dataImages[0].image[y1:y2, x1:x2] = frame
        frame_mapping = dataImages[0].image
    
    cv2.imshow('frame_mapping', cv2.resize(frame_mapping, (int(frame_mapping.shape[1]/4), int(frame_mapping.shape[0]/4))) )
    cv2.waitKey(1)
    if len(dets_plate)>0:
        cv2.imwrite(f'./event_plate/image_frame_mapping_{name_folder}.jpg', frame_mapping)
    time_total2 = time.time()
    # print('time total : ', time_total2 - time_total1)
    
    # if len(lp_rets)>0:
    #     print(lp_rets[-1]['id'])
    #     print(lp_rets[-1]['box_kps'])
    #     print(lp_rets[-1]['txt'])

    return count_plate_per_id , buffer_plate_per_id , lp_rets, results, frame_mapping, lp_rets_dict_data


def main_LP(dataImages, 
            coordinate_rois, 
            reg_plate, 
            model_retina, 
            count_plate_per_id = {}, 
            buffer_plate_per_id = {}, 
            lp_rets = []):
    # print("MAIN main_LP".center(100,"*"))
    datas = []
    dict_data = {}
    

    max_buffer_per_id = 5 #buffer_plate_per_id
    len_buffer_time = 10 #count_plate_per_id: after 10 frame delete id
    lp_rets_dict_data = []

    x1, y1, x2, y2 = coordinate_rois[dataImages[0].cid]
    frame = dataImages[0].image.copy()[y1:y2, x1:x2]

    dets_plate = model_retina.detect_plate(frame, file_img = False, check_sort = False)
    results = reg_plate.reg_plate(frame, dets_plate)


    frame_raw = draw_img(results, copy.deepcopy(frame))  
    cv2.rectangle(frame_raw, (10, 10), (frame_raw.shape[1]-10, frame_raw.shape[0]-10), (0, 0, 255), 2)
    dataImages[0].image[y1:y2, x1:x2] = frame
    frame_mapping = dataImages[0].image
    
    name_folder = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")
    # cv2.imwrite(f'./event/biso_{name_folder}.jpg', frame_mapping)
    resized = cv2.resize(frame_raw, (int(frame_raw.shape[1]/2),int(frame_raw.shape[0]/2)), interpolation = cv2.INTER_AREA)
    cv2.imshow('frame_mapping', resized)
    if cv2.waitKey(0)==27:
        sys.exit()


    # buffer_plate_per_id = checkSimilarLP(results, buffer_plate_per_id)

    for result in results:
        txt = result[1]
        txt_result = result[1]
        x1, y1, x2, y2 = result[2]
        w = x2 - x1
        h = y2 - y1
        if (w+h)>=160:
            if txt not in buffer_plate_per_id:
                # print('w, h, w+h, txt : ', w, h, w+h, txt_result)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cx = x1  
                cy = y1 - 12
                cv2.putText(frame, txt_result, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

                datas.append(txt_result)

                buffer_plate_per_id[txt] = txt
            # else:
            #     print('w, h, w+h, txt : ', w, h, w+h, txt_result)

    if len(datas)>0:
        dict_data[dataImages[0].cid] = {"img": frame, "data": datas}


    return buffer_plate_per_id, dict_data


def main_test(dataImages, 
            coordinate_rois, 
            reg_plate, 
            model_retina, 
            buffer_reid,
            count_plate_per_id = {}, 
            buffer_plate_per_id = {}, 
            lp_rets = [],
            count_id=0):
    # print("MAIN main_LP".center(100,"*"))
    datas = []
    dict_data = {}
    

    max_buffer_per_id = 5 #buffer_plate_per_id
    len_buffer_time = 10 #count_plate_per_id: after 10 frame delete id
    lp_rets_dict_data = []

    x1, y1, x2, y2 = coordinate_rois[dataImages[0].cid]
    frame = dataImages[0].image.copy()[y1:y2, x1:x2]

    dets_plate = model_retina.detect_plate(frame, file_img = False, check_sort = False)
    results_reg = reg_plate.reg_plate(frame, dets_plate)

    

    dataImages[0].image[y1:y2, x1:x2] = frame
    frame_mapping = dataImages[0].image
    
    #detect
    event_detect = draw_img_detect(dets_plate, frame)
    cv2.imshow('event_detect ', event_detect)
    cv2.waitKey(0)
    
    results = []
    for idx, result in enumerate(results_reg):
        x1, y1, x2, y2 = result[2]
        w = x2 - x1
        h = y2 - y1
        if (w+h)>=160:
            results.append(result)


    buffer_reid, results, count_id = reID(results, buffer_reid, count_id)  
    

    if len(results)>0:
    # if debug_log:
        frame_raw = draw_img(results, copy.deepcopy(frame))  
        cv2.rectangle(frame_raw, (10, 10), (frame_raw.shape[1]-10, frame_raw.shape[0]-10), (0, 0, 255), 2)
        name_folder = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")
        cv2.imwrite(f'./event_plate/main/biso_{name_folder}.jpg', frame_raw)

        # resized = cv2.resize(frame_raw, (int(frame_raw.shape[1]/2),int(frame_raw.shape[0]/2)), interpolation = cv2.INTER_AREA)
        # cv2.imshow('frame_mapping', resized)
        # if cv2.waitKey(0)==27:
        #     sys.exit()
    ########################################
    #processing number plate results
    #check new plate or old plate
    if debug_log:
        print('result')
        for result in results:
            print(result[0], result[1])
        for key in buffer_plate_per_id:
            print(f'buffer_plate_per_id[{str(key)}] : , {buffer_plate_per_id[key][0][1]}')
        print('count_plate_per_id : ', count_plate_per_id)
        print('lp_rets : ', lp_rets)
    for result in results:
        del result[3]
        txt_result = result[1]
        id = result[0]
        if id in buffer_plate_per_id:
            buffer_plate_per_id[id].append(result)
        else:
            buffer_plate_per_id[id] = [result]
            count_plate_per_id[id] = len_buffer_time

    #delete id and mergerLP if count_plate_per_id <=0
    #count_plate_per_id : count number  of per id plate count plate id follow frame, count decrease, plate may or may not appear
    key_deletes = []
    
    for key in count_plate_per_id:
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, key)
        if count_plate_per_id[key]<=0: # nếu quá n frame ko xuất hiện lại 
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            # if len(buffer_plate_per_id[key]) == 1:
            #     Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            #     #send plate
            #     lp_rets.append(lp_ret)
            #     lp_rets_dict_data.append(lp_ret)
            # else:
                # Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            check_Duplicate_id = False
            lp_ret = mergerLP(buffer_plate_per_id[key])
            # for i in range(len(lp_rets)):
            #     if lp_ret['id'] == lp_rets[i]['id']:
            #         if lp_ret['txt'] != lp_rets[i]['txt']:
            #             lp_rets[i]['txt']=lp_ret['txt']
            #             check_Duplicate_id = False
            #         else:
            #             check_Duplicate_id = True
            if check_Duplicate_id == False:
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                #send plate
                lp_rets.append(lp_ret)
                lp_rets_dict_data.append(lp_ret)
            key_deletes.append(key)
    
    #delete plate
    for key in key_deletes:
        del count_plate_per_id[key]
        del buffer_plate_per_id[key]

    key_deletes = []
    #delete and mergerLP if buffer_plate_per_id[key]>=max_buffer_per_id
    for key in buffer_plate_per_id:
        if len(buffer_plate_per_id[key])>=max_buffer_per_id:
            check_Duplicate_id = False
            lp_ret = mergerLP(buffer_plate_per_id[key])
            # for i in range(len(lp_rets)): # nếu giống id thì ko add thêm 
            #     if lp_ret['id'] == lp_rets[i]['id']:
            #         if lp_ret['txt'] != lp_rets[i]['txt']:
            #             lp_rets[i]['txt']=lp_ret['txt']
            #             check_Duplicate_id = False
            #         else:
            #             check_Duplicate_id = True
            if check_Duplicate_id == False:
                #send plate
                lp_rets.append(lp_ret)
                lp_rets_dict_data.append(lp_ret)
            key_deletes.append(key) # quá số frame cho phép reset 
    #delete plate
    for key in key_deletes:
        del count_plate_per_id[key]
        del buffer_plate_per_id[key]
        # del buffer_reid[key]

    for key in count_plate_per_id:
        count_plate_per_id[key]-=1
    # if len(results)<0:
    #buffer results < = 20 elements
    if len(lp_rets)>=1:
        for i in range(int(len(lp_rets)/2)):
            lp_rets.pop(0)
    


    #####################


    #check buffer_reid , count_id 
    key_reids = []
    for key_reid in buffer_reid:
        if key_reid<(count_id - 10):
            key_reids.append(key_reid)

    for key_reid in key_reids:
        del buffer_reid[key_reid]


    if len(datas)>0:
        dict_data[dataImages[0].cid] = {"img": frame, "data": datas}

    if debug_log:
        print('after')
        print('result')
        for result in results:
            print(result[0], result[1])
        for key in buffer_plate_per_id:
            print(f'buffer_plate_per_id[{str(key)}] : , {buffer_plate_per_id[key][0][1]}')
        print('count_plate_per_id : ', count_plate_per_id)
        print('lp_rets : ', lp_rets)
        print('lp_rets_dict_data : ', lp_rets_dict_data)

    if debug_log:
        for result in results:
            id = result[0]
            txt = result[1]
            txt_result = result[1] + '_'+ str(id)
            x1, y1, x2, y2 = result[2]
            w = x2 - x1
            h = y2 - y1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cx = x1  
            cy = y1 - 12
            cv2.putText(frame, txt_result, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('frame', cv2.resize(frame, (int(frame.shape[1]/2),int(frame.shape[0]/2)), interpolation = cv2.INTER_AREA))
        if cv2.waitKey(1)==27:
            sys.exit()


    return buffer_reid, dict_data, count_id, lp_rets_dict_data



def checkTextSimilar(txt, list_txts, thres=1.3):
    thres_list = len(list_txts)/thres
    count_list = 0
    for idx, list_txt in enumerate(list_txts):
        len_txt = len(txt) if len(txt)<len(list_txt) else len(list_txt)
        thres_txt = len_txt/thres
        count = 0
        for i in range(0,len_txt):
            if txt[i] == list_txt[i]:
                count+=1 
        if count>thres_txt:
            count_list+=1
    if count_list>thres_list:
        return True 
    return False



def reID(results, buffer_reid, count_id):
    for idx, result in enumerate(results):
        txt_result = result[1]
        print('txt_result : ', txt_result)
        check_ID = False
        if len(buffer_reid)==0:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            buffer_reid[count_id] = [txt_result]
            results[idx][0] = count_id
            count_id+=1
        else:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            for key_idx in buffer_reid:
                # if txt in buffer_reid
                if txt_result in buffer_reid[key_idx]:
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                    check_ID = True
                    results[idx][0] = key_idx
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename, buffer_reid)
                else:
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                    # if plate does not in buffer check similar
                    check_txt = checkTextSimilar(txt_result,buffer_reid[key_idx])
                    if check_txt:
                        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                        check_ID = True
                        buffer_reid[key_idx].append(txt_result)
                        results[idx][0] = key_idx
            if check_ID==False:
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                # if plate does not in buffer and not similar any plates in buffer => create new id
                buffer_reid[count_id] = [txt_result]
                results[idx][0] = count_id
                count_id+=1
    # print('buffer_reid : ', buffer_reid)
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    return buffer_reid, results, count_id

def main_reid(dataImages, 
            coordinate_rois, 
            reg_plate, 
            model_retina, 
            buffer_reid,
            count_plate_per_id = {}, 
            buffer_plate_per_id = {}, 
            lp_rets = [],
            count_id=0):
    # print("MAIN main_LP".center(100,"*"))
    t1 =time.time()
    datas = []
    dict_data = {}
    

    max_buffer_per_id = 5 #buffer_plate_per_id
    len_buffer_time = 10 #count_plate_per_id: after 10 frame delete id
    lp_rets_dict_data = []
    h, w, _ = dataImages[0].image.shape
    # print('h, w : ', h, w)
    w_scale = w/640
    h_scale = h/w_scale
    # print('w_scale, h_scale: ', w_scale, h_scale)
    # dataImages[0].image = cv2.resize(dataImages[0].image , (640, int(h_scale)), interpolation = cv2.INTER_AREA) 
    x1, y1, x2, y2 = coordinate_rois[dataImages[0].cid]
    y1 = 100
    frame = dataImages[0].image.copy()[y1:y2, x1:x2]
    # if save_img:
    #     name_frame = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")
    #     cv2.imwrite(f'./event_plate/raw/{str(dataImages[0].cid)}.jpg', frame)
    t1 = time.time()

    dets_plate = model_retina.detect_plate(frame, file_img = False, check_sort = False)#[[x1, y1, x2, y2], box_kps, id, conf, img_raw]
    t2 = time.time()
    results_reg = reg_plate.reg_plate(frame, dets_plate) #results.append([id, txt_result, [x1, y1, x2, y2], cropped_img, kps, check_plate, conf])

    t3 = time.time()
    print('time detect ', t2 - t1)
    print('time reg : ', t3-t2)
    

    dataImages[0].image[y1:y2, x1:x2] = frame
    frame_mapping = dataImages[0].image    
    
    
    results = []
    for idx, result in enumerate(results_reg):
        x1, y1, x2, y2 = result[2]
        w = x2 - x1
        h = y2 - y1
        if (w+h)>=160:
            results.append(result)

    t33 = time.time()
    buffer_reid, results, count_id = reID(results, buffer_reid, count_id)  
    t34 = time.time()
    # print('time reid ', t34-t33)
    name_folder = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")
    name_folder_day = datetime.datetime.now().strftime("%d_%m_%y")


    # if os.path.isdir(f'./event_plate/detect/{name_folder_day}')==False:
    #     os.mkdir(f'./event_plate/detect/{name_folder_day}')

    # if os.path.isdir(f'./event_plate/reg/{name_folder_day}')==False:
    #     os.mkdir(f'./event_plate/reg/{name_folder_day}')
    # if save_img:
    #     if len(results)>0:
    #         #detect
    #         event_detect = draw_img_detect(dets_plate, copy.deepcopy(frame))
    #         #reg
    #         frame_raw = draw_img(results, copy.deepcopy(frame))  
    #         cv2.rectangle(frame_raw, (10, 10), (frame_raw.shape[1]-10, frame_raw.shape[0]-10), (0, 0, 255), 2)
            

    #         if os.path.isdir(f'./event_plate/main/{name_folder}')==False:
    #             os.mkdir(f'./event_plate/main/{name_folder}')

    #         cv2.imwrite(f'./event_plate/main/{name_folder}/reg.jpg', frame_raw)
    #         cv2.imwrite(f'./event_plate/main/{name_folder}/detect.jpg', event_detect)


    #         cv2.imwrite(f'./event_plate/detect/{name_folder_day}/{name_folder}.jpg', event_detect)
    #         cv2.imwrite(f'./event_plate/reg/{name_folder_day}/{name_folder}.jpg', frame_raw)

        # cv2.imshow()



        # resized = cv2.resize(frame_raw, (int(frame_raw.shape[1]/2),int(frame_raw.shape[0]/2)), interpolation = cv2.INTER_AREA)
        # cv2.imshow('frame_mapping', resized)
        # if cv2.waitKey(0)==27:
        #     sys.exit()
    ########################################
    t35 = time.time()
    # print('time save 1 : ', t35-t34)
    #processing number plate results
    #check new plate or old plate
    # if debug_log:
    #     print('result')
    #     for result in results:
    #         print(result[0], result[1])
    #     for key in buffer_plate_per_id:
    #         print(f'buffer_plate_per_id[{str(key)}] : , {buffer_plate_per_id[key][0][1]}')
    #     print('count_plate_per_id : ', count_plate_per_id)
    #     print('lp_rets : ', lp_rets)
    for result in results:
        del result[3]
        txt_result = result[1]
        id = result[0]
        if id in buffer_plate_per_id:
            buffer_plate_per_id[id].append(result)
            #count_plate_per_id[id] = len_buffer_time#10
        else:
            buffer_plate_per_id[id] = [result]
            count_plate_per_id[id] = len_buffer_time#10

    #delete id and mergerLP if count_plate_per_id <=0
    #count_plate_per_id : count number  of per id plate count plate id follow frame, count decrease, plate may or may not appear
    key_deletes = []
    t36 = time.time()
    # print('time get result : ', t36-t35)
    for key in count_plate_per_id:
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, key)
        if count_plate_per_id[key]<=0: # nếu quá n frame ko xuất hiện lại 
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            lp_ret = mergerLP(buffer_plate_per_id[key])
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            #send plate
            lp_rets.append(lp_ret)
            lp_rets_dict_data.append(lp_ret)
            key_deletes.append(key)
        elif len(buffer_plate_per_id[key])>=max_buffer_per_id:
            lp_ret = mergerLP(buffer_plate_per_id[key])
            #send plate
            lp_rets.append(lp_ret)
            lp_rets_dict_data.append(lp_ret)
            key_deletes.append(key) # quá số frame cho phép reset 
    
    #delete plate
    for key in key_deletes:
        del count_plate_per_id[key]
        del buffer_plate_per_id[key]
        # del buffer_reid[key]

    for key in count_plate_per_id:
        count_plate_per_id[key]-=1
    # if len(results)<0:
    #buffer results < = 20 elements
    if len(lp_rets)>=1:
        for i in range(int(len(lp_rets)/2)):
            lp_rets.pop(0)
    
    t37 = time.time()
    # print('time get result 2 : ', t37-t36)

    #####################


    #check buffer_reid , count_id 
    key_reids = []
    for key_reid in buffer_reid:
        if key_reid<(count_id - 10):
            key_reids.append(key_reid)

    for key_reid in key_reids:
        del buffer_reid[key_reid]


    if len(datas)>0:
        dict_data[dataImages[0].cid] = {"img": frame, "data": datas}

    # if save_img:
    #     #check event
    #     if os.path.isdir(f'./event_plate/event/{name_folder_day}')==False:
    #         os.mkdir(f'./event_plate/event/{name_folder_day}')

    #     if len(lp_rets_dict_data) > 0 :#and len(results)>0:
    #         for lp in lp_rets_dict_data:
    #             cv2.rectangle(lp['frame'], (lp['box_kps'][0], lp['box_kps'][1]), (lp['box_kps'][2], lp['box_kps'][3]), (0, 0, 255), 2)
    #             cv2.putText(lp['frame'],str(lp['id'])+"_"+lp['txt'],(lp['box_kps'][0], lp['box_kps'][1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4,cv2.LINE_AA)
    #             cv2.imwrite(f'./event_plate/event/{name_folder_day}/{name_folder}.jpg', lp['frame'])
    #             cv2.imwrite(f'./event_plate/main/{name_folder}/event.jpg', lp['frame'])

    # if debug_log:
    #     print('after')
    #     print('result')
    #     for result in results:
    #         print(result[0], result[1])
    #     for key in buffer_plate_per_id:
    #         print(f'buffer_plate_per_id[{str(key)}] : , {buffer_plate_per_id[key][0][1]}')
    #     print('count_plate_per_id : ', count_plate_per_id)
    #     print('lp_rets : ', lp_rets)
    #     print('lp_rets_dict_data : ', lp_rets_dict_data)

    if True:#debug_log:
        for result in results:
            id = result[0]
            txt = result[1]
            txt_result = result[1] + '_'+ str(id)
            x1, y1, x2, y2 = result[2]
            w = x2 - x1
            h = y2 - y1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cx = x1  
            cy = y1 - 12
            cv2.putText(frame, txt_result, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # cv2.imshow('frameLP', cv2.resize(frame, (int(frame.shape[1]/2),int(frame.shape[0]/2)), interpolation = cv2.INTER_AREA))
    

    t4 =time.time()
    # print('time process 1 : ', t4 - t37)
    print('time process : ', t4 - t3)
    print('time all licence plate : ', str(dataImages[0].cid), (time.time()-t1))
    # if cv2.waitKey(1)==27:
    #     sys.exit()
    return buffer_reid, dict_data, count_id, lp_rets_dict_data

def LiveStream(dataImages, 
            coordinate_rois, 
            reg_plate, 
            model_retina, 
            buffer_reid,
            count_plate_per_id = {}, 
            buffer_plate_per_id = {}, 
            lp_rets = [],
            count_id=0):
    # print("MAIN main_LP".center(100,"*"))
    datas = []
    dict_data = {}
    

    max_buffer_per_id = 5 #buffer_plate_per_id
    len_buffer_time = 10 #
    # count_plate_per_id: after 10 frame delete id
    lp_rets_dict_data = []

    x1, y1, x2, y2 = coordinate_rois[dataImages[0].cid]
    frame = dataImages[0].image.copy()[y1:y2, x1:x2]

    # name_frame = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")
    # cv2.imwrite(f'./event_plate/raw/raw.jpg', frame)

    dets_plate = model_retina.detect_plate(frame, file_img = False, check_sort = False)#[[x1, y1, x2, y2], box_kps, id, conf, img_raw]
    results_reg = reg_plate.reg_plate(frame, dets_plate) #results.append([id, txt_result, [x1, y1, x2, y2], cropped_img, kps, check_plate, conf])

    #detect
    image_detect = draw_img_detect(dets_plate, copy.deepcopy(frame))
    cv2.imshow('event_detect : ', cv2.resize(image_detect, (int(image_detect.shape[1]/2), int(image_detect.shape[0]/2))))
    if cv2.waitKey(1) == 27:
        sys.exit()

    dataImages[0].image[y1:y2, x1:x2] = frame
    frame_mapping = dataImages[0].image
    
    
    
    results = []
    for idx, result in enumerate(results_reg):
        x1, y1, x2, y2 = result[2]
        w = x2 - x1
        h = y2 - y1
        if (w+h)>=160:
            results.append(result)


    buffer_reid, results, count_id = reID(results, buffer_reid, count_id)  
    
    for result in results:
        del result[3]
        txt_result = result[1]
        id = result[0]
        if id in buffer_plate_per_id:
            buffer_plate_per_id[id].append(result)
            #count_plate_per_id[id] = len_buffer_time#10
        else:
            buffer_plate_per_id[id] = [result]
            count_plate_per_id[id] = len_buffer_time#10

    #delete id and mergerLP if count_plate_per_id <=0
    #count_plate_per_id : count number  of per id plate count plate id follow frame, count decrease, plate may or may not appear
    key_deletes = []
    
    for key in count_plate_per_id:
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, key)
        if count_plate_per_id[key]<=0: # nếu quá n frame ko xuất hiện lại 
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            lp_ret = mergerLP(buffer_plate_per_id[key])
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            #send plate
            lp_rets.append(lp_ret)
            lp_rets_dict_data.append(lp_ret)
            key_deletes.append(key)
        elif len(buffer_plate_per_id[key])>=max_buffer_per_id:
            lp_ret = mergerLP(buffer_plate_per_id[key])
            #send plate
            lp_rets.append(lp_ret)
            lp_rets_dict_data.append(lp_ret)
            key_deletes.append(key) # quá số frame cho phép reset 
    
    #delete plate
    for key in key_deletes:
        del count_plate_per_id[key]
        del buffer_plate_per_id[key]
        # del buffer_reid[key]

    for key in count_plate_per_id:
        count_plate_per_id[key]-=1
    # if len(results)<0:
    #buffer results < = 20 elements
    if len(lp_rets)>=1:
        for i in range(int(len(lp_rets)/2)):
            lp_rets.pop(0)
    
    #####################


    #check buffer_reid , count_id 
    key_reids = []
    for key_reid in buffer_reid:
        if key_reid<(count_id - 10):
            key_reids.append(key_reid)

    for key_reid in key_reids:
        del buffer_reid[key_reid]


    if len(datas)>0:
        dict_data[dataImages[0].cid] = {"img": frame, "data": datas}

    return buffer_reid, dict_data, count_id, lp_rets_dict_data, image_detect

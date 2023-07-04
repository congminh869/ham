import sys
sys.path.insert(0, './yolov5')
sys.path.insert(0, './sort')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
import copy


import torch
import torch.backends.cudnn as cudnn

from include.yolo import DtectBox, DataTracking

#yolov5
# from yolov5.utils.datasets import LoadImages, LoadStreams
# from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
# from yolov5.utils.torch_utils import select_device, time_synchronized

#SORT
import skimage
from sort import *

#debug log
from inspect import currentframe, getframeinfo
import datetime

debug_log = True
# Debug_log(currentframe(), getframeinfo(currentframe()).filename)
def Debug_log(cf, filename, name = None):
    if debug_log:
        ct = datetime.datetime.now()
        print(f'================ [{ct}] file {filename} , line : {cf.f_lineno} {name}')

torch.set_printoptions(precision=3)

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def concat_img(dataTrackings, mul=576):
    def get_sub_img(dataTrackings):
        datas = []
        for dataTracking in dataTrackings:
            frame = dataTracking.frame
            for dtectBox in dataTracking.dtectBoxs:
                xmin, ymin, xmax, ymax = dtectBox.bbox
                sub_i = frame[int(ymin): int(ymax), int(xmin): int(xmax), ::-1]
                # cv2.imshow('image_crop', sub_i)
                # cv2.waitKey(0)
                sub_data ={
                    'img': sub_i,
                    'bbox': dtectBox.bbox,
                    'cid': dataTracking.cid,
                    'id_tracking': dtectBox.id_tracking,
                    'count': dataTracking.count
                                }
                datas.append(copy.deepcopy(sub_data))
        return datas

    datas = get_sub_img(dataTrackings)
    # sorted img and its boxes by the length (H then W) of dimension in descending order
    datas_sorted = list(sorted(
            datas,   
            key=lambda x: [x['bbox'][3]-x['bbox'][1], x['bbox'][2]-x['bbox'][0]],
            reverse=True))
    

    sub_img_info_list = []
    bgs = [] # list to save concat-images

    bg = np.zeros((mul, mul, 3)) # concat image -> return
    w_left = h_left = mul
    _begin = True # check if this is the begining of the assignment 
    current_pos_w = current_pos_h = 0 # current position of cursor
    max_lastline_h = 0 # the maximum height of all image of the last line assignment 

    min_w_left_of_patch = 0 # save the min column left of each line assignment, 
                            # use when connecting with new patch
    patch_num = 0 # mark the patch that sub images is assigned

    for data in  datas_sorted:
        img = data['img']
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        h_img, w_img = img.shape[:2]

        if h_img <= 0 or w_img <=0:
            continue 
        
        if h_img > bg.shape[0]:
            # TODO
            continue
        
        if w_img > bg.shape[1]:
            # TODO
            continue

        if _begin:
            _begin = False
            max_lastline_h = h_img
            min_w_left_of_patch = w_left
    

        if w_left < w_img:
            h_left -= max_lastline_h

            if w_left < min_w_left_of_patch:
                min_w_left_of_patch = w_left

            if h_left < h_img:
                current_pos_h = 0
                current_pos_w = 0
                _bg = copy.deepcopy(bg)
                bgs.append(DataTracking(_bg, [], -1, -1, -1))

                bg = np.zeros((mul, mul, 3))

                w_left = mul
                h_left = mul
                max_lastline_h = h_img

                patch_num += 1
    
            else:
                current_pos_h += max_lastline_h
                current_pos_w = 0
                # h_left -= max_lastline_h
                max_lastline_h = h_img
                w_left = mul

        # print(current_pos_h, current_pos_w, img.shape, h_left, w_left)
        bg[current_pos_h: current_pos_h + h_img,
           current_pos_w: current_pos_w + w_img, :] = img

        # print(bg[current_pos_h: current_pos_h + h_img,
        #    current_pos_w: current_pos_w + w_img, :])
        # cv2.imshow('img', img)
        # cv2.imshow('bg', np.uint8(bg))
        # cv2.waitKey(0)
        sub_img_info_list.append({
            # 'sub_img': img,
            'current_pos': (current_pos_h, current_pos_w, h_img, w_img), # y, x w, h
            'origin_pos': data['bbox'], # x, y, x, y
            'cid': data['cid'],
            'count': data['count'],
            'id_tracking': data['id_tracking'],
            'patch_id': patch_num
        })
        
        current_pos_w += (w_img)
        w_left -= (w_img)
    bgs.append(DataTracking(bg, [], -1, -1, -1))
    # for sii in sub_img_info_list:
    #   print(sii)
    # for idx, i in enumerate(bgs):
    #     a = list(filter(lambda p: p['patch_id']==idx, sub_img_info_list))
    #     for info in a:
    #         y, x, h, w = info['current_pos']
    #         _cid = info['cid']
    #         idt = info['id_tracking']
    #         pid = info['patch_id']
    #         cv2.putText(i.frame, f'{_cid} - {idt} - {pid}', 
    #                (int(x+w/6), int(y+h/2)), 
    #                cv2.FONT_HERSHEY_SIMPLEX,
    #                0.3, (0,255,0), 2, cv2.LINE_AA)
    #     cv2.imshow(f"itest/{idx}.jpg", np.uint8(i.frame))
        # cv2.waitKey(0)
        # cv2.imwrite(f"itest/{idx}.jpg", i.frame)
    return bgs, sub_img_info_list

######### MAPPING BACK
def check_inside_rectangle(rect, point):
    '''
    rect: y, x, h, w
    '''
    x,y = point
    yr, xr, h, w = rect

    if x < xr+w and x >= xr and y < yr+h and y >= yr:
        return True
    return False
 

def mapping_back(sub_img_info_list, detection_of_concats, dataTrackings_concat_img=None):
    '''Args
        
        detection_of_concat: detection bboxes of concat images, [[detections #1], ... [detections #n]] 

        dataTrackings_concat_img: pass when debug only
    '''
    final_boxes = []
    for pidx, detects in enumerate(detection_of_concats):
        # Debug_log(currentframe(), getframeinfo(currentframe()).filename)

        _detects = copy.deepcopy(detects)
        _detects = list(sorted(_detects, key = lambda p: p[-2], reverse=True))
        sub_info = list(filter(lambda p: p['patch_id'] == pidx, sub_img_info_list))
        # Debug_log(currentframe(), getframeinfo(currentframe()).filename, sub_info)
        # # print(len(sub_info))
        # x, y, x, y, conf, label

        for idx, info in enumerate(sub_info):
            # Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            flag_assigned = False
            current_idx = None
            xmin_last, ymin_last, xmax_last, ymax_last, conf_last, label_last = [None]*6
            for detect in _detects:
                # Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                xmin, ymin, xmax, ymax, conf, label = detect
                flag = check_inside_rectangle(info['current_pos'], ((xmin+xmax)/2,(ymin+ymax)/2))
                # Debug_log(currentframe(), getframeinfo(currentframe()).filename, flag)
                
                if flag and not flag_assigned:
                    current_idx = idx
                    xmin_last, ymin_last, xmax_last, ymax_last, conf_last, label_last = xmin, ymin, xmax, ymax, conf, label
                    flag_assigned = True
                elif flag:
                    if (ymin_last+ymax_last) > (ymin+ymax):
                        # print((ymin_last+ymax_last), (ymin+ymax))
                        xmin_last, ymin_last, xmax_last, ymax_last, conf_last, label_last = xmin, ymin, xmax, ymax, conf, label
                        # break
            if current_idx==None:
                continue

            # y,x,h,w = info['current_pos']
            # image_show = dataTrackings_concat_img[pidx].frame
            # cv2.rectangle(image_show, (int(x), int(y)), (int(x+w), int(y+h)), (255,255,0), 2)
            # cv2.circle(image_show, (int((xmin_last+xmax_last)/2),int((ymin_last+ymax_last)/2)), 5, (255,255,0), -1)
            # try: 
            
            offset_y, offset_x = sub_info[current_idx]['current_pos'][:2]
            x_origin, y_origin = sub_info[current_idx]['origin_pos'][:2]
            xmin_final = xmin_last - offset_x + x_origin
            ymin_final = ymin_last - offset_y + y_origin
            xmax_final = xmax_last - offset_x + x_origin
            ymax_final = ymax_last - offset_y + y_origin

            final_boxes.append({'bbox':[int(xmin_final),
                                        int(ymin_final),
                                        int(xmax_final),
                                        int(ymax_final),
                                        float(conf_last), float(label_last)],
                                'id_tracking': sub_info[current_idx]['id_tracking'],
                                'cid': sub_info[current_idx]['cid'],
                                'count': sub_info[current_idx]['count']})
            # except:
            #   continue 
        # cv2.imshow('image_show_check_flag', np.uint8(image_show))
    return final_boxes

#############
class NCheckViolateID:
    def __init__ (self, minimum_frame_to_count_violate=3, max_time_keep_id=150, num_track_keeped = 11):
        self.minimum_frame_to_count_violate = minimum_frame_to_count_violate # The minimum frames tracked an obj to determine violation
        self.max_time_keep_id = max_time_keep_id # seconds
        self.num_track_keeped = num_track_keeped # Length of history frames tracking an obj
        self.track_violate_history = defaultdict(lambda: defaultdict(list)) # store history of objs
        self.last_time_appearence = defaultdict(lambda: defaultdict(datetime.datetime)) # store the time of obj last appear in frame
        self.last_time_obj_appearence_in_cid = defaultdict(datetime.datetime) # store the last time that objs appear in frames
        self.begin = True # first frame

        
    def check(self, detections, permitted_labels):
        '''
        detections[{
            bbox,
            cid,
            id_tracking
        }, ...]

        permitted_label: {cid: [list of label], ...}, 0: None, 1: White, 2: Orange
        Mapping {
                0: 'None',
                1: 'Red',
                2: 'Yellow',
                3: 'White',
                4: 'Blue',
                5: 'Orange',
                6: 'Others'} 0,4,6->0, 3->1, 1,2,5->2
        '''
        detections = list(sorted(detections, key=lambda x: [x['cid'], x['count']]))
        warning_list = []
        for detect in detections:
            label = detect['bbox'][-1]
            if label in [0,4,6]:
              new_label = 0
            elif label in [3]:
              new_label = 1
            elif label in [1,2,5]:
              new_label = 2
            # print(label) 
            cid = detect['cid']
            id_tracking = detect['id_tracking']

            if new_label not in permitted_labels[cid]:
                self.track_violate_history[cid][id_tracking].append(0) # 0 for false
            else:
                self.track_violate_history[cid][id_tracking].append(1) # 1 for true

            self.last_time_obj_appearence_in_cid[cid] = datetime.datetime.now()  
            self.last_time_appearence[cid][id_tracking] = datetime.datetime.now()  
            self.track_violate_history[cid][id_tracking] = self.track_violate_history[cid][id_tracking][-self.num_track_keeped:]

            # in the nearest (num_track_keeped) frames, if an obj has violated in more than (1/3) the total frames keeped  -> false
            if sum(self.track_violate_history[cid][id_tracking]) < len(self.track_violate_history[cid][id_tracking])*(2/3) and \
               len(self.track_violate_history[cid][id_tracking]) >= self.minimum_frame_to_count_violate:
                warning_list.append(detect)
                #print('---history: ', id_tracking, new_label,label)
        return warning_list
           
    def remove_disappear_id(self, detections):
        current_appear = defaultdict(list)
        detections = list(sorted(detections, key=lambda x: [x['cid'], x['count']]))
        for detect in detections:
            current_appear[detect['cid']].append(detect['id_tracking'])
            
        if self.begin:
            self.begin = False
            return
        else:
            for cid in list(self.last_time_appearence.keys()):
                # if there is none object appearing in a frame in certain amount of time, 
                # delete track history of that frame
                if cid not in current_appear:
                    time_cid_none_obj = datetime.datetime.now() - self.last_time_obj_appearence_in_cid[cid]
                    if time_cid_none_obj.seconds > self.max_time_keep_id:
                        del self.track_violate_history[cid]
                        del self.last_time_appearence[cid]
                        del self.last_time_obj_appearence_in_cid[cid]
                
                # if objects do not appear in current frame and have been disappear for more than 
                # maximum time keep tracking -> delete track history of objects
                else:
                    for id_tracking in list(self.last_time_appearence[cid].keys()):
                        if id_tracking not in current_appear[cid]:
                            if id_tracking in self.track_violate_history[cid]:
                                current_time = datetime.datetime.now()
                                disappear_time = current_time - self.last_time_appearence[cid][id_tracking]
                                if (disappear_time.seconds) > self.max_time_keep_id:
                                    # print('----disappear time: ', cid, id_tracking, disappear_time)
                                    del self.track_violate_history[cid][id_tracking]
                                    del self.last_time_appearence[cid][id_tracking]



def final(dataTrackings_human, detections, warning_list):
    '''
    detections[{
            bbox,
            cid,
            id_tracking
        }, ...]
    '''
    def convention(_detections):
        tracks = defaultdict(list)
        for detect in _detections:
            dB = DtectBox() # check this line
            dB.bbox = detect['bbox'][:4]
            label = detect['bbox'][-1]
            if label in [0,4,6]:
                label = 0
                namec = 'none'
            elif label in [3]:
                label = 1
                namec = 'white'
            elif label in [1,2,5]:
                label = 2
                namec = 'orange'
            dB.name_class = namec
            dB.id_tracking = detect['id_tracking']
            dB.class_conf = detect['bbox'][-2]
            dB.class_id = None
            tracks[detect['cid'], detect['count']].append(dB)
        return tracks

    d = convention(detections) # all detects
    w = convention(warning_list) # false cases
    # print(len(detections), len(warning_list))

    dataTrackings = []
    dataTrackings_false = []
    dataTrackings_human_false_hat = []
  
    for dataTracking in dataTrackings_human:
        cid = dataTracking.cid
        count = dataTracking.count
        # print(d)
        if (cid, count) not in d:
            dtTracking = DataTracking(dataTracking.frame,
                                      [],
                                      cid, 
                                      dataTracking.type_id,
                                      count)
        else:
            dtTracking = DataTracking(dataTracking.frame,
                                      d[cid, count],
                                      cid, 
                                      dataTracking.type_id,
                                      count)
        dataTrackings.append(dtTracking)

        if (cid, count) not in w:
            dtTracking_false = DataTracking(dataTracking.frame,
                                           [],
                                           cid, 
                                           dataTracking.type_id,
                                           count)
            dtTracking_human_false_hat = DataTracking(dataTracking.frame,
                                           [],
                                           cid, 
                                           dataTracking.type_id,
                                           count)
        else:
            dtbox_human_false_hat = []
            for dtbox_hat_false in w[cid, count]:
                id_tracking = dtbox_hat_false.id_tracking
                for dtbox_human in dataTracking.dtectBoxs:
                    if dtbox_human.id_tracking == id_tracking:
                        dtbox_human_false_hat.append(dtbox_human)
                        break

            dtTracking_false = DataTracking(dataTracking.frame,
                                           w[cid, count],
                                           cid, 
                                           dataTracking.type_id,
                                           count)
            dtTracking_human_false_hat =  DataTracking(dataTracking.frame,
                                           dtbox_human_false_hat,
                                           cid, 
                                           dataTracking.type_id,
                                           count)
        dataTrackings_false.append(dtTracking_false)
        dataTrackings_human_false_hat.append(dtTracking_human_false_hat)
    return dataTrackings, dataTrackings_false, dataTrackings_human_false_hat


def run(dataTrackings, helmet_model, permitted_labels, NCheckViolateID, mul=576):
    names = helmet_model.model.names
    list_of_concat_imgs, sub_img_info_list = concat_img(dataTrackings, mul)
    dataTrackings_concat_imgs = helmet_model.detect(list_of_concat_imgs)

    detection_of_concat_imgs = []
    for idx, dtTracking in enumerate(dataTrackings_concat_imgs):
        list_bb = []
        for dtBox in dtTracking.dtectBoxs:
            xmin, ymin, xmax, ymax = dtBox.bbox
            conf = dtBox.class_conf
            label = dtBox.name_class
            for k in names:
                if names[k] == label:
                    cls_id = k
                    break
            bb = [xmin, ymin, xmax, ymax, conf, cls_id]
            list_bb.append(bb)
        detection_of_concat_imgs.append(list_bb)

    detections = mapping_back(sub_img_info_list, detection_of_concat_imgs)
    warning_list = NCheckViolateID.check(detections, permitted_labels)
    # print('---wraning list: ', warning_list)

    NCheckViolateID.remove_disappear_id(detections)
    dataTrackings_hat, dataTrackings_hat_false, dataTrackings_human_false_hat = final(dataTrackings, detections, warning_list)
    return dataTrackings_hat, dataTrackings_hat_false, dataTrackings_human_false_hat
    

    
    




        
    




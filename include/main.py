import sys
import argparse
import cv2
import os 
import glob
import time
import numpy as np
import copy
from multiprocessing import Process
from multiprocessing import Queue

from include.yolo import YOLOv562, DataTracking
from include.roi import ROI, ROI_polygon, centerInROI
from include.get_output import draw_box, draw_boxs, draw_boxs_helmet, \
                                draw_boxs_polygon, draw_boxs_HSV, show_debug_log, save_event,\
                                draw_boxs_only, draw_boxs_only_fence, draw_boxs_only_personHoldThong,\
                                draw_boxs_only_Count, get_draw_boxs
from include.model_paddle_class import paddleClas, infer_paddleClass
from include.include_main import crop_img, mapping_coordinate, \
                                mapping_dataTrackings, get_rects, pretreatment_tracking, \
                                load_sort, crop_img_polygon, mapping_dataTrackings_polygon, \
                                convert_event, check_is_high_HSV_and_convert_event, convert_envet_fence,\
                                show, check_is_clock_and_convert_event, pretreatment_tracking_loop_cid, \
                                meger_vehicle_person, split_vehicle_person, crop_img_polygon_fence
# from include.backgroud_subtration_hoabinh import personHoldThing
from include.personHoldThing import preprecess_output
from include.helmet_processes import to_batches, N_scaling # import processes helmets
from include.CheckSort import run, NCheckViolateID

#region Phat hien vat the cao
from include.personHoldThingDetect import personHoldThing_All

#debug log
from inspect import currentframe, getframeinfo
import datetime

import yaml
import json

# with open('./config/config_main_include.yaml', 'r') as file:
#     configuration = yaml.safe_load(file)

debug_log = False
# Debug_log(currentframe(), getframeinfo(currentframe()).filename)
def Debug_log(cf, filename, name = None):
    if debug_log:
        ct = datetime.datetime.now()
        print(f'================ [{ct}] file {filename} , line : {cf.f_lineno} {name}')

debog_main_log = True


def main_TBA(dataImages, 
            coordinate_rois, 
            yolo_person, 
            yolo_vehicle, 
            yolo_hat, 
            personHoldThingDetect_video_1, 
            labels_allow_helmet, 
            list_sort,
            nCheckViolateID,
            convertEvent):
    '''
    input: 
        dataImages: contain n frame from n camera (n<30)
                    coordinate_rois : region of interest
                    cid : camera id
                    type_id 
                    count 
                    image : frame
        coordinate_rois: {1:[500, 300, 948, 638]}
        labels_allow_helmet:                            v
        multi detect: hat                               v
        multi detect: person                            v
        multi detect: person atribute                   v
        multi detect: person carries object             v
    output:
        the object is not allowed to appear in the specified area
        warn people in and out of the area
        warn people carrying objects higher than 2m
        warn vehicles in and out of the area
        warn people not to wear hats
        return dataImages

    dataTrackings_person_output: list dataTracking of people
    dataTrackings_helmet, violates: list dataTracking of vihecles and violates is True or False
    dataTrackings_vihecle_output
    dataTrackings_holdThingDet
    
    dataTracking_vihecle_warning: list dataTracking of vihecles touch the boder and image drawed
    dataTracking_person_warning: list dataTracking of people touch the boder and image drawed

    '''
    t0 = time.time()
    if debog_main_log:
        print("MAIN TBA".center(150,"*"))
    #detect person
    dataTrackings_person = []
    coordinate_roi_border = 200
    new_coordinate_rois = {}
    coordinate_mapping_xyxys = {}
    t1= time.time()
    # print('time 1 : ', time.time()-t0)
    for idx, dataImage in enumerate(dataImages): 
        # Debug_log(currentframe(), getframeinfo(currentframe()).filename, idx)
        cid = dataImage.cid
        # if dataImage.cid in coordinate_rois:
        type_id = dataImage.type_id
        count = dataImage.count
        t11 = time.time()
        # draw polygon
        # pts = np.array(coordinate_roi,np.int32)
        # # pts = pts.reshape((-1, 1, 2))                 
        # image = cv2.polylines(image, [pts],True, (0, 0, 255), 3)
        image, new_coordinate_rois[cid], coordinate_mapping_xyxys[cid] = crop_img_polygon(dataImage.image, coordinate_rois[dataImage.cid], coordinate_roi_border)#dataImage.image.copy()
        # print('time crop polygon: ', time.time()-t11)
        # cv2.imwrite(f'./event/image_crop{str(cid)}.jpg', image)
        # cv2.imwrite(f'./event/image_src{str(cid)}.jpg', dataImage.image)
        
        dtectBoxs_person = []
        dataTrackings_person.append(DataTracking(image, dtectBoxs_person, cid, type_id, count))


    # for i, data in enumerate(dataTrackings_person):
    #     print(dataTrackings_person[i].cid, ' | ', dataTrackings_person[i].type_id, ' | ', dataTrackings_person[i].count)
    #     print(dataImages[i].cid, ' | ', dataImages[i].type_id, ' | ', dataImages[i].count)

    dataTrackings_vehicle = copy.deepcopy(dataTrackings_person)
    t2 = time.time()
    # print('time 2 precess datatracking: ', time.time()-t1)

    #model detect person: input dataTrackings_person
    dataTrackings_person_output = yolo_person.detect(dataTrackings_person)
    t3 = time.time()
    # print('time 2 time detect person: ', time.time()-t2)
    # if debug_log:
    #draw_boxs_only(dataTrackings_person_output, 'person')
    
    # print('time inferen person: ', t2- t1)

    #model detect vihecle
    dataTrackings_vihecle_output = yolo_vehicle.detect(dataTrackings_vehicle)
    # if debug_log:
    #     draw_boxs_only(dataTrackings_vihecle_output,'vihecle')
    t4 = time.time()
    # print('time 3 detect vehicle: ', time.time()-t3)


    # dataTrackings_person_vihecle = meger_vehicle_person(dataTrackings_person_output, dataTrackings_vihecle_output)
    #meger_vehicle_person
    for idx in range(len(dataTrackings_person_output)):
        if len(dataTrackings_vihecle_output[idx].dtectBoxs)>0:
            dataTrackings_person_output[idx].dtectBoxs +=dataTrackings_vihecle_output[idx].dtectBoxs
    t5 = time.time()
    # print('time 4 : ', time.time()-t4)
    dataTrackings_person_vihecle_sort = pretreatment_tracking_loop_cid(dataTrackings_person_output, list_sort)
    
    dataTrackings_person_vihecle_sort_dbg = mapping_dataTrackings_polygon(dataTrackings_person_vihecle_sort, dataImages, coordinate_rois, coordinate_mapping_xyxys)

    t6 = time.time()

    #checkPersonFalse
    # dataTrackings_person_vihecle_sort = personFalse.checkPersonFalse(person_vihecle_sort,dataTrackings_person_output)

    # print('time 5 sort: ', time.time()-t5)
    # if debug_log:
    # draw_boxs_only(dataTrackings_person_vihecle_sort, 'person_vihicle')

    #-----------------model detect Hat-------------------------------
    permited_label = labels_allow_helmet#{di.cid:[1] for di in dataImages}
    # ncheck = NCheckViolateID() # Init out of while loop
    dataTrackings_hat, dataTrackings_hat_false, dataTrackings_human_false_hat = run(dataTrackings_person_vihecle_sort, yolo_hat, permited_label, nCheckViolateID)
    # if debug_log:
    #     draw_boxs_only(dataTrackings_hat)
    t7=time.time()
    # print('time 6 detect hat: ', time.time()-t6)

    #-------------------------------------------------------------------------------------#


    #person carry object tall: input dataTrackings_person_output
    # dataTrackings_holdThingDet = personHoldThingDetect_video_1.detect(dataTrackings_person_sort)
    t5 = time.time()
    # print('time inferen holdThingDet: ', t5- t4)
    t6 = time.time()
    # print('time inferen vihecle warning: ', t6- t5)

    #warning person in or out area
    t66 = time.time()
    if debug_log:
        dataTrackings_person_vihecle_sort_warning= ROI_polygon(copy.deepcopy(dataTrackings_person_vihecle_sort), coordinate_roi_border, coordinate_rois=new_coordinate_rois)
        
    else:
        dataTrackings_person_vihecle_sort_warning= ROI_polygon(dataTrackings_person_vihecle_sort, coordinate_roi_border, coordinate_rois=new_coordinate_rois)
    # print('time 7 ROI : ', time.time()-t7)
    t8 = time.time()
    # mapping and output
    dataTrackings_person_vihecle_sort_mapping = mapping_dataTrackings_polygon(dataTrackings_person_vihecle_sort, dataImages, coordinate_rois, coordinate_mapping_xyxys)
    dataTrackings_person_vihecle_sort_warning_mapping = mapping_dataTrackings_polygon(dataTrackings_person_vihecle_sort_warning, dataImages, coordinate_rois, coordinate_mapping_xyxys)
    
    dataTrackings_hat_mapping = mapping_dataTrackings_polygon(dataTrackings_hat, dataImages, coordinate_rois, coordinate_mapping_xyxys)
    dataTrackings_hat_false_mapping = mapping_dataTrackings_polygon(dataTrackings_hat_false, dataImages, coordinate_rois, coordinate_mapping_xyxys)
    # print('time 8 mapping: ', time.time()-t8)
    t10 = time.time()
    
    # if debug_log:
    #     draw_boxs_only(dataTrackings_hat_mapping)
    #dataTrackings_helmet

    #violates
    ##########################################
    #draw 

    dataTracking_person_warnings, dataTracking_vihecle_warnings = split_vehicle_person(dataTrackings_person_vihecle_sort_warning_mapping)
    # print('time 9 : ', time.time()-t10)
    # if debug_log:
    #     draw_boxs_only(dataTracking_person_warnings,'warning_person a')
    #     draw_boxs_only(dataTracking_vihecle_warnings,'warning_vihecle b')

    t77 = time.time()
    dict_tracking = {"person": dataTracking_person_warnings,
                     "vehicle": dataTracking_vihecle_warnings,
                      "hat": dataTrackings_hat_false_mapping}

    # dict_tracking = {"person": dataTrackings_holdThingDet_mapping,
    #                  "vihecle": dataTrackings_person_sort_mapping,
    #                  "holdThingDet": dataTrackings_person_sort_mapping,
    #                   "hat": dataTrackings_person_sort_mapping,
    #                   "uniform": dataTrackings_person_sort_mapping}
    t11 = time.time()
    if debug_log:
        dict_data = convertEvent.convert(copy.deepcopy(dict_tracking), coordinate_rois)#convert_event(copy.deepcopy(dict_tracking), coordinate_rois)
    else:
        dict_data = convertEvent.convert(dict_tracking, coordinate_rois)#convert_event(copy.deepcopy(dict_tracking), coordinate_rois)

    # print('time 10 convert event: ', time.time()-t11)
    t12 = time.time()
    

    
    if debug_log:
        save_event(copy.deepcopy(dict_data))
        draw_boxs_polygon(dataTrackings_person_vihecle_sort_mapping, coordinate_rois, coordinate_mapping_xyxys, 'detect sort person')
        draw_boxs_polygon(dataTrackings_person_vihecle_sort_warning_mapping, coordinate_rois, coordinate_mapping_xyxys, 'detect person warningqq')
        
        # draw_boxs_polygon(dataTrackings_vihecle_output_mapping, coordinate_rois, coordinate_mapping_xyxys, 'detect vihecle')
        # draw_boxs_polygon(dataTracking_vihecle_warning_mapping, coordinate_rois, coordinate_mapping_xyxys, 'warning vihecle')
        
        # # draw_boxs_polygon(dataTrackings_holdThingDet_mapping, coordinate_rois, coordinate_mapping_xyxys, 'holdThingDet')
        # print('+++++++++++++++++++++++++done++++++++++++++++++++++++++++++++++++++++++++')
        # draw_boxs_polygon(dataTrackings_hat_mapping, coordinate_rois, coordinate_mapping_xyxys, 'detect hat')
        draw_boxs_polygon(dataTrackings_hat_false_mapping, coordinate_rois, coordinate_mapping_xyxys, 'warning hat')
        draw_boxs_only(dataTrackings_hat_false_mapping, 'hat warning')
        draw_boxs_only(dataTrackings_person_vihecle_sort_warning_mapping, 'sort warning')
        # draw_boxs_only(dataTrackings_person_vihecle_sort_mapping, 'sortmapping')

        # show_debug_log([dataTrackings_person_sort_mapping,
        #             dataTracking_person_warning_mapping,
        #             dataTrackings_vihecle_output_mapping,
        #             dataTracking_vihecle_warning_mapping,
        #             dataTrackings_hat_mapping,
        #             dataTrackings_hat_false_mapping
        #             ])

        # draw_boxs_helmet(dataTrackings_helmet, coordinate_rois,'detect helmet', violates)
        # show
        # draw_boxs(dataTrackings_person_mapping)
        # for i in range(len(dataTrackings_person_sort_mapping)):
        #     # Debug_log(currentframe(), getframeinfo(currentframe()).filename, i)
        #     resized = cv2.resize(dataTrackings_person_sort_mapping[i].frame, (640,360), interpolation = cv2.INTER_AREA)
        #     str_show = "person" + str(dataTrackings_person_sort_mapping[i].cid);
        #     cv2.imshow(str_show, resized)
        #     cv2.waitKey(1)
    t_final = time.time()
    # print('time inferen all: ', t_final- t0)
    imgs_out = get_draw_boxs(dataImages, [dataTrackings_person_vihecle_sort_dbg, dataTrackings_person_vihecle_sort_mapping, dataTrackings_hat_false_mapping], coordinate_rois)
    if cv2.waitKey(1) == 27:
        sys.exit()
    return dict_data, imgs_out

def main_tunnel(dataImages, 
                coordinate_rois,
                labels_allow_helmet, 
                labels_allow_uniform,
                list_sort,
                nCheckViolateID,
                yolo_person, 
                yolo_hat, 
                engine,
                personFalse):
    '''
    input: 
        dataImages: contain n frame from n camera (n<30)
                    cid : camera id
                    type_id 
                    count 
                    image : frame
        coordinate_rois : region of interest
        multi detect: hat             v
        multi detect: person          v
        multi detect: person atribute v
    output:
        the object is not allowed to appear in the specified area
        the people wear uniform: Hat and clothes
        return dataImages
    
    dataTrackings_person_output: list dataTracking of people
    dataTrackings_helmet, violates: list dataTracking of vihecles and violates is True or False
    dataTrackings_vihecle_output
    dataTrackings_holdThingDet
    
    dataTracking_vihecle_warning: list dataTracking of vihecles touch the boder and image drawed
    dataTracking_person_warning: list dataTracking of people touch the boder and image drawed
    
    '''
    # if debog_main_log:
    #     print("MAIN TUNNEL".center(100,"*"))
    t0 = time.time()

    dataTrackings_person = []
    coordinate_roi_border = 10
    new_coordinate_rois = {}
    coordinate_mapping_xyxys = {}
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    for idx, dataImage in enumerate(dataImages): 
        # Debug_log(currentframe(), getframeinfo(currentframe()).filename, idx)
        tcheck = time.time()
        cid = dataImage.cid
        type_id = dataImage.type_id
        count = dataImage.count
        image, new_coordinate_rois[cid], coordinate_mapping_xyxys[cid] = crop_img_polygon(dataImage.image.copy(), coordinate_rois[dataImage.cid], coordinate_roi_border)#dataImage.image.copy()
        dtectBoxs_person = []
        dataTrackings_person.append(DataTracking(image, dtectBoxs_person, cid, type_id, count))
        # Debug_log(currentframe(), getframeinfo(currentframde()).filename, 'tcheck = ' +str(time.time()-tcheck))

    t1 = time.time()
    Debug_log(currentframe(), getframeinfo(currentframe()).filename, 't0 = ' +str(t1-t0))
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    #detect person
    dataTrackings_person_sort = yolo_person.detect(dataTrackings_person)
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    
    # if debug_log:
    #     draw_boxs_only(copy.deepcopy(dataTrackings_person_output), 'dataTrackings_person_output')
    t1_ = time.time()
    # dataTrackings_person_sort = pretreatment_tracking(dataTrackings_person_output, list_sort)
    Debug_log(currentframe(), getframeinfo(currentframe()).filename, str(time.time()-t1_))
    t2 = time.time()
    # print('time time_person : ', t2-t1)

    #warning person in or out area
    dataTrackings_person_sort_warning = ROI_polygon(copy.deepcopy(dataTrackings_person_sort), coordinate_roi_border, coordinate_rois=new_coordinate_rois)
    t2_ = time.time()
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    #model detect uniform

    dataTracking_uniform, dataTracking_uniform_false= infer_paddleClass(engine, dataTrackings_person_sort, labels_allow_uniform)
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    ################
    t3  = time.time()
    t4 =time.time()
    # mapping and output
    dataTrackings_person_sort_mapping = mapping_dataTrackings_polygon(dataTrackings_person_sort, dataImages, coordinate_rois, coordinate_mapping_xyxys)
    dataTrackings_person_sort_warning_mapping = mapping_dataTrackings_polygon(dataTrackings_person_sort_warning, dataImages, coordinate_rois, coordinate_mapping_xyxys)
    dataTracking_uniform_mapping = mapping_dataTrackings_polygon(dataTracking_uniform, dataImages, coordinate_rois, coordinate_mapping_xyxys)
    dataTracking_uniform_false_mapping = mapping_dataTrackings_polygon(dataTracking_uniform_false, dataImages, coordinate_rois, coordinate_mapping_xyxys)

    dict_tracking = {"person": dataTrackings_person_sort_warning_mapping,
                     "uniform": dataTracking_uniform_false_mapping}
    t5 =time.time()
    # print('time mapping : ', t5-t4)
    if debug_log:
        dict_data = convert_event(copy.deepcopy(dict_tracking), coordinate_rois)
    else:
        dict_data = convert_event(dict_tracking, coordinate_rois)

    t6 =time.time()
    # print('time convert_event : ', t6-t5)

    if debug_log:
        save_event(dict_data)
        draw_boxs_only(dataTracking_uniform_mapping,'dataTracking_uniform_mapping')
        draw_boxs_only(dataTracking_uniform_false_mapping,'dataTracking_uniform_false_mapping')
    
    # t6 =time.time()
    # print('time convert_event : ', t6-t5)
    t_final =time.time()
    # print('time draw_boxs_polygon : ', t_final-t0)
    imgs_out = get_draw_boxs(dataImages, [dataTrackings_person_sort_mapping, dataTracking_uniform_mapping], coordinate_rois)

    if cv2.waitKey(1) == 27:
        sys.exit()

    return dict_data, imgs_out

def StreamTunnel(dataImages, 
                coordinate_rois,
                labels_allow_helmet, 
                labels_allow_uniform,
                list_sort,
                nCheckViolateID,
                yolo_person, 
                yolo_hat, 
                engine,
                personFalse):
    '''
    input: 
        dataImages: contain n frame from n camera (n<30)
                    cid : camera id
                    type_id 
                    count 
                    image : frame
        coordinate_rois : region of interest
        multi detect: hat             v
        multi detect: person          v
        multi detect: person atribute v
    output:
        the object is not allowed to appear in the specified area
        the people wear uniform: Hat and clothes
        return dataImages
    
    dataTrackings_person_output: list dataTracking of people
    dataTrackings_helmet, violates: list dataTracking of vihecles and violates is True or False
    dataTrackings_vihecle_output
    dataTrackings_holdThingDet
    
    dataTracking_vihecle_warning: list dataTracking of vihecles touch the boder and image drawed
    dataTracking_person_warning: list dataTracking of people touch the boder and image drawed
    
    '''
    if debog_main_log:
        print("MAIN TUNNEL".center(100,"*"))
    t0 = time.time()

    dataTrackings_person = []
    coordinate_roi_border = 100
    new_coordinate_rois = {}
    coordinate_mapping_xyxys = {}

    for idx, dataImage in enumerate(dataImages): 
        tcheck = time.time()
        cid = dataImage.cid
        type_id = dataImage.type_id
        count = dataImage.count
        image, new_coordinate_rois[cid], coordinate_mapping_xyxys[cid] = crop_img_polygon(dataImage.image.copy(), coordinate_rois[dataImage.cid], coordinate_roi_border)#dataImage.image.copy()
        dtectBoxs_person = []
        dataTrackings_person.append(DataTracking(image, dtectBoxs_person, cid, type_id, count))

    t1 = time.time()
    #detect person
    dataTrackings_person = yolo_person.detect(dataTrackings_person)
    t1_ = time.time()
    person_sort_ = pretreatment_tracking(dataTrackings_person, list_sort)
    
    #checkPersonFalse
    dataTrackings_person_sort= personFalse.checkPersonFalse(person_sort_,dataTrackings_person)

    t2 = time.time()
    # print('time time_person : ', t2-t1)

    #warning person in or out area
    dataTrackings_person_sort_warning = ROI_polygon(copy.deepcopy(dataTrackings_person_sort), coordinate_roi_border, coordinate_rois=new_coordinate_rois)
    t2_ = time.time()
    #model detect uniform
    dataTracking_uniform, dataTracking_uniform_false= infer_paddleClass(engine, dataTrackings_person_sort, labels_allow_uniform)
    ################
    t3  = time.time()
    t4 =time.time()
    # mapping and output
    dataTrackings_person_sort_mapping = mapping_dataTrackings_polygon(dataTrackings_person_sort, dataImages, coordinate_rois, coordinate_mapping_xyxys)
    dataTrackings_person_sort_warning_mapping = mapping_dataTrackings_polygon(dataTrackings_person_sort_warning, dataImages, coordinate_rois, coordinate_mapping_xyxys)
    dataTracking_uniform_mapping = mapping_dataTrackings_polygon(dataTracking_uniform, dataImages, coordinate_rois, coordinate_mapping_xyxys)
    dataTracking_uniform_false_mapping = mapping_dataTrackings_polygon(dataTracking_uniform_false, dataImages, coordinate_rois, coordinate_mapping_xyxys)

    dict_tracking = {"person": dataTrackings_person_sort_warning_mapping,
                     "uniform": dataTracking_uniform_false_mapping}
    t5 =time.time()
    # print('time mapping : ', t5-t4)
    if debug_log:
        dict_data = convert_event(copy.deepcopy(dict_tracking), coordinate_rois)
    else:
        dict_data = convert_event(dict_tracking, coordinate_rois)

    t6 =time.time()
    # print('time convert_event : ', t6-t5)

    if debug_log:
        save_event(dict_data)
        draw_boxs_only(dataTracking_uniform_mapping,'dataTracking_uniform_mapping')
        draw_boxs_only(dataTracking_uniform_false_mapping,'dataTracking_uniform_false_mapping')
    
    # t6 =time.time()
    # print('time convert_event : ', t6-t5)
    t_final =time.time()
    # print('time draw_boxs_polygon : ', t_final-t0)
    if cv2.waitKey(1) == 27:
        sys.exit()

    return dict_data

def main_high_volt_switch(dataImages, coordinate_rois, yolo_HVS):
    '''
        input many roi and id per HVS
        output true or false per HVS

        support many roi
        check if detect HVS in roi => return true
        esle without hvs in roi => return false
        
        recive full image
        coordinate_rois{cid: [{id: [x1, y1, x2, y2]}, {id: [x1, y1, x2, y2]}, ...],
                        cid: [{id: [x1, y1, x2, y2]}, {id: [x1, y1, x2, y2]}, ...],
                        ...}

        detect => compare roi and set ip => true or false with per HSV

    '''

    if debog_main_log:
        print("MAIN HIGH VOLT SWITCH".center(100,"*"))
    t0 = time.time()
    #detect person
    dataTrackings = []
    coordinate_roi_border = 20
    for idx, dataImage in enumerate(dataImages): 
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, idx)
        cid = dataImage.cid
        type_id = dataImage.type_id
        count = dataImage.count
        image = dataImage.image.copy()
        dtectBoxs_person = []
        dataTrackings.append(DataTracking(image, dtectBoxs_person, cid, type_id, count))

    t1 = time.time()
    
    dataTrackings_output = yolo_HVS.detect(dataTrackings)
    t2 = time.time()
    # print('time time_person : ', t2-t1)
    draw_boxs_HSV(dataTrackings_output, coordinate_rois, 'HSV')
    # for i in range(len(dataTrackings_output)):
    #     Debug_log(currentframe(), getframeinfo(currentframe()).filename, i)
    #     resized = cv2.resize(dataTrackings_output[i].frame, (640,360), interpolation = cv2.INTER_AREA)
    #     str_show = "person" + str(dataTrackings_output[i].cid);
    #     cv2.imshow(str_show, resized)
    #     if cv2.waitKey(0) == 27:
    #         sys.exit()

    dict_data = check_is_high_HSV_and_convert_event(dataTrackings_output, coordinate_rois)
    # print('time all : ', time.time()-t0)
    return dict_data

def main_clock(dataImages, yolo_clock):
    '''
        input image
        output i or o
    '''
    if debog_main_log:
        print("MAIN CLOCK".center(100,"*"))
    t0 = time.time()
    #detect person
    dataTrackings = []
    coordinate_roi_border = 20
    for idx, dataImage in enumerate(dataImages): 
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, idx)
        cid = dataImage.cid
        type_id = dataImage.type_id
        count = dataImage.count
        image = dataImage.image.copy()
        dtectBoxs_person = []
        dataTrackings.append(DataTracking(image, dtectBoxs_person, cid, type_id, count))

    t1 = time.time()
    
    dataTrackings_output = yolo_clock.detect(dataTrackings)
    t2 = time.time()
    # print('time time_person : ', t2-t1)
    # draw_boxs_HSV(dataTrackings_output, coordinate_rois, 'HSV')
    # for i in range(len(dataTrackings_output)):
    #     Debug_log(currentframe(), getframeinfo(currentframe()).filename, i)
    #     resized = cv2.resize(dataTrackings_output[i].frame, (640,360), interpolation = cv2.INTER_AREA)
    #     str_show = "person" + str(dataTrackings_output[i].cid);
    #     cv2.imshow(str_show, resized)
    #     if cv2.waitKey(0) == 27:
    #         sys.exit()

    dict_data = check_is_clock_and_convert_event(dataTrackings_output)
    # print('time all : ', time.time()-t0)
    return dict_data

def main_belt(dataImages, coordinate_rois, yolo_belt, yolo_person, list_sort, personBelt):
    '''
        input image
        output true or false
    '''
    t0 = time.time()
    if debog_main_log:
        print("MAIN BELT".center(100,"*"))
    #detect person
    dataTrackings = []
    coordinate_roi_border = 10
    new_coordinate_rois = {}
    coordinate_mapping_xyxys = {}

    # print('coordinate_rois : ', coordinate_rois)
    for idx, dataImage in enumerate(dataImages): 
        cid = dataImage.cid
        type_id = dataImage.type_id
        count = dataImage.count
        # cv2.imshow(f'imagesrc_{str(cid)}', cv2.resize(dataImage.image, (int(dataImage.image.shape[1]/4), int(dataImage.image.shape[0]/4))))

        image, new_coordinate_rois[cid], coordinate_mapping_xyxys[cid] = crop_img_polygon(dataImage.image, coordinate_rois[dataImage.cid], coordinate_roi_border)#dataImage.image.copy()
        # cv2.imshow(f'image_{str(cid)}', cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4))))
        # cv2.waitKey(1)
        dtectBoxs_person = []
        dataTrackings.append(DataTracking(image, dtectBoxs_person, cid, type_id, count))

    dataTrackings_belts = copy.deepcopy(dataTrackings)
    t1 = time.time()
    #detect person
    dataTrackings = yolo_person.detect(dataTrackings)
    dataTrackings_sort = pretreatment_tracking(dataTrackings, list_sort)

    # if debug_log:
    #     draw_boxs_only(dataTrackings_sort, 'sortmapping')
    #dataTracking_person_warning = ROI(copy.deepcopy(dataTrackings_sort), coordinate_roi_border, coordinate_rois=coordinate_rois)  

    dataTrackings_sort_mapping = mapping_dataTrackings_polygon(dataTrackings_sort, dataImages, coordinate_rois, coordinate_mapping_xyxys)

    t1 = time.time()
    #detect person
    dataTrackings_belt= yolo_belt.detect(dataTrackings_belts)
    dataTrackings_belt_mapping = mapping_dataTrackings_polygon(dataTrackings_belt, dataImages, coordinate_rois, coordinate_mapping_xyxys)
    if debug_log:
        draw_boxs_only(dataTrackings_belt_mapping, 'dataTrackings_belt_mapping')
    t2 = time.time()
    # print('time time_person : ', t2-t1)    
    personBelt.check(dataTrackings_sort_mapping, dataTrackings_belt_mapping)
    # print('time all : ', time.time()-t0)
    
    dict_data = personBelt.dict_data_tempt
    
    if debug_log:
        save_event(dict_data)
        draw_boxs_only(dataTrackings_belt_mapping, 'dataTrackings_belt_mapping')
    
    # print('dict_data : ', dict_data)
    return dict_data


def mapping_coordinate_polygon_fence(dtectBoxs, coordinate_roi, coordinate_mapping_xy):
    dtectBoxs_person = copy.deepcopy(dtectBoxs)
    
    MIN_X, MIN_Y, _, _ = coordinate_mapping_xy
     
    for idx, dtectBox in enumerate(dtectBoxs):
        if dtectBox.bbox!=None:
            x1, y1, x2, y2 = dtectBox.bbox[0], dtectBox.bbox[1], dtectBox.bbox[2], dtectBox.bbox[3]
            x1_mapping = MIN_X+x1
            y1_mapping = MIN_Y+y1
            x2_mapping = MIN_X+x2
            y2_mapping = MIN_Y+y2
            dtectBoxs_person[idx].bbox = [x1_mapping, y1_mapping, x2_mapping, y2_mapping]
    return dtectBoxs_person

def mapping_dataTrackings_polygon_fence(dataTrackings, dataImages, coordinate_rois, coordinate_mapping_xys):
    '''
        function mapping image_crop's coordinate with image_src' coordinate
    '''
    dataTrackings_mapping = []
    for idx, dataTracking in enumerate(dataTrackings): 
        cid = dataTracking.cid
        type_id = dataTracking.type_id
        count = dataTracking.count
        image = dataTracking.frame_src#copy.deepcopy(dataImages[idx].image)
        coordinate_roi = coordinate_rois[dataTracking.cid]
        coordinate_mapping_xy = coordinate_mapping_xys[dataTracking.cid]
        dtectBoxs = dataTracking.dtectBoxs

        # draw polygon
        # pts = np.array(coordinate_roi,np.int32)
        # # pts = pts.reshape((-1, 1, 2))                 
        # image = cv2.polylines(image, [pts],True, (0, 0, 255), 3)

        # cv2.imwrite(f'./event_fence/dataTrackings_fence_{str(cid)}_{str(count)}.jpg', dataTracking.frame_src)
        # cv2.imwrite(f'./event_fence/dataImages_fence_{str(cid)}_{str(count)}.jpg', dataImages[idx].image)
        dtectBoxs_mapping = mapping_coordinate_polygon_fence(dtectBoxs, coordinate_roi, coordinate_mapping_xy)
        dataTrackings_mapping.append(DataTracking(image, dtectBoxs_mapping, cid, type_id, count))
    return dataTrackings_mapping

def main_fence(dataImages, yolo_person, yolo_FS, convertEventFence, coordinate_rois, check_detect_person=True):
    '''
        input: dataImages (n frame get from buffer)
        output: person and fire and smoke
    '''
    if debog_main_log:
        print("MAIN FENCE".center(100,"*"))
    t0 = time.time()
    #detect person
    dataTrackings_person = []
    dataTrackings_FS = []
    coordinate_roi_border = 0
    new_coordinate_rois = {}
    coordinate_mapping_xyxys = {}
    for idx, dataImage in enumerate(dataImages): 
        cid = dataImage.cid
        type_id = dataImage.type_id
        count = dataImage.count
        # cv2.imwrite(f'./event_fence/input_fence_{str(cid)}_{str(count)}.jpg', dataImage.image)
        # cv2.imshow('dataImage.image', dataImage.image)
        # cv2.waitKey(0)
        # if dataImage.cid in coordinate_rois:
            # cv2.imwrite(f'./event_fence/fence_{str(cid)}.jpg', dataImage.image)

            # print('coordinate_rois : ', coordinate_rois)
            # print('dataImage.cid : ', dataImage.cid)
            # print('coordinate_rois[dataImage.cid] : ', coordinate_rois[dataImage.cid])
        image, new_coordinate_rois[cid], coordinate_mapping_xyxys[cid] = crop_img_polygon_fence(dataImage.image, coordinate_rois[dataImage.cid], coordinate_roi_border)#dataImage.image.copy()
        dtectBoxs_person = []
        dataTrackings_person.append(DataTracking(image, dtectBoxs_person, cid, type_id, count, frame_src = dataImage.image))
    dataTrackings_FS = copy.deepcopy(dataTrackings_person)

    t1 = time.time()
    #detect person
    dataTrackings_person_output = yolo_person.detect(dataTrackings_person)
    # draw_boxs_only_fence(dataTrackings_person_output, 'detect')
    t2 = time.time()
    # print('time time_person : ', t2-t1)

    #detect Fire and Smoke
    dataTrackings_FS_output = yolo_FS.detect(dataTrackings_FS)
    t3 = time.time()
    # print('time time_FS : ', t3-t2)

    dataTrackings_person_output_mapping = mapping_dataTrackings_polygon_fence(dataTrackings_person_output, dataImages, coordinate_rois, coordinate_mapping_xyxys)
    dataTrackings_FS_output_mapping = mapping_dataTrackings_polygon_fence(dataTrackings_FS_output, dataImages, coordinate_rois, coordinate_mapping_xyxys)
    

    # draw_boxs_only_fence(dataTrackings_person_output_mapping, 'mapping')
    dict_data = {}
    # dict_data = convert_envet_fence(dataTrackings_person_output, dataTrackings_FS_output)
    dict_data = convertEventFence.convert(dataTrackings_person_output_mapping, dataTrackings_FS_output, coordinate_rois)
    # print('time all: ', time.time()-t0)

    # if debug_log:
        # save_event(dict_data)
        # draw_boxs_polygon(dataTrackings_person_output_mapping, coordinate_rois, coordinate_mapping_xyxys, 'person sort')
        # draw_boxs_polygon(dataTrackings_FS_output, coordinate_rois, coordinate_mapping_xyxys, 'uniform_mapping')


        # draw_boxs_only(dataTrackings_person_output_mapping, 'person')

    imgs_ret = get_draw_boxs(dataImages, [dataTrackings_person_output_mapping, dataTrackings_FS_output_mapping], coordinate_rois)

    return dict_data, imgs_ret

def main_personHoldThingDetect(dataImages, coordinate_rois, yolo_person, list_sort, convertEvent, coordinate_roi_border, bgshb):
    if debog_main_log:
        print("MAIN main_personHoldThingDetect".center(100,"*"))
    t0 = time.time()
    dataTrackings_person = []
    new_coordinate_rois = {}
    coordinate_mapping_xyxys = {}
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    # print('len(dataImages) : ', len(dataImages))
    for idx, dataImage in enumerate(dataImages): 
        # Debug_log(currentframe(), getframeinfo(currentframe()).filename, idx)
        cid = dataImage.cid
        type_id = dataImage.type_id
        count = dataImage.count
        # print('cid : ', cid, 'count', count)
        # cv2.imwrite('./event_holdthingdetect/image.jpg', dataImage.image)
        # cv2.imshow(f'image_main_personHoldThingDetect{str(cid)}', cv2.resize(dataImage.image, (int(dataImage.image.shape[1]/2), int(dataImage.image.shape[0]/2))))
        # cv2.waitKey(1)
        coordinate_rois[dataImage.cid] = [[0,0], [dataImage.image.shape[1],0], [dataImage.image.shape[1], dataImage.image.shape[0]], [0, dataImage.image.shape[0]]]
        image, new_coordinate_rois[cid], coordinate_mapping_xyxys[cid] = crop_img_polygon(dataImage.image, coordinate_rois[dataImage.cid], coordinate_roi_border)#dataImage.image.copy()
        dtectBoxs_person = []
        dataTrackings_person.append(DataTracking(image, dtectBoxs_person, cid, type_id, count))
    t2 = time.time()
    # print('time process : ', t2 - t0)
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    dataTrackings_person_output = yolo_person.detect(dataTrackings_person)
    t3 = time.time()
    # print('time detect : ', t3-t2)
    dataTrackings_person_sort = pretreatment_tracking_loop_cid(dataTrackings_person_output, list_sort)
    t4 = time.time()
    # if debug_log:
    # draw_boxs_only_personHoldThong(dataTrackings_person_sort, 'sort_person')
    # print('time sort : ', t4-t3)
    t5 = time.time()
    batch_contours = bgshb.handle_batch_images(dataTrackings=dataTrackings_person_sort)
    t6 = time.time()
    dataTrackings_out = preprecess_output(batch_contours, dataTrackings_person_sort)
    t7 = time.time()
    # print('time image processing : ', t7-t5)
    dataTrackings_out_mapping = mapping_dataTrackings_polygon(dataTrackings_out, dataImages, coordinate_rois, coordinate_mapping_xyxys)
    # print('len(dataTrackings_out_mapping) :', len(dataTrackings_out_mapping))
    t8 = time.time()
    # if debug_log:
    # draw_boxs_only_personHoldThong(dataTrackings_out_mapping, 'warning')
    t9 = time.time()
    dict_tracking = {"holdThingDet": dataTrackings_out_mapping}

    if debug_log:
        dict_data = convertEvent.convert(copy.deepcopy(dict_tracking), coordinate_rois)#convert_event(copy.deepcopy(dict_tracking), coordinate_rois)
    else:
        dict_data = convertEvent.convert(dict_tracking, coordinate_rois)#convert_event(copy.deepcopy(dict_tracking), coordinate_rois)
    
    t1 = time.time()
    # print('time all main_personHoldThingDetect: ', t1-t0)
    name_folder = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")
    # os.mkdir('./event/' +name_folder)
    for cid in dict_data:
        image = dict_data[cid]["img"]
        datas = dict_data[cid]["data"]
        cv2.imwrite('./event_holdthingdetect/'+name_folder+'_image'+datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")+'.jpg', image)

    # cv2.waitKey(0)
    
    return dict_data

def main_test(dataImages, 
                coordinate_rois,
                list_sort,
                yolo_person,
                personFalse):
    if debog_main_log:
        print("MAIN test".center(100,"*"))
    t0 = time.time()

    dataTrackings_person = []
    coordinate_roi_border = 100
    new_coordinate_rois = {}
    coordinate_mapping_xyxys = {}

    for idx, dataImage in enumerate(dataImages): 
        # Debug_log(currentframe(), getframeinfo(currentframe()).filename, idx)
        tcheck = time.time()
        cid = dataImage.cid
        type_id = dataImage.type_id
        count = dataImage.count
        image, new_coordinate_rois[cid], coordinate_mapping_xyxys[cid] = crop_img_polygon(dataImage.image.copy(), coordinate_rois[dataImage.cid], coordinate_roi_border)#dataImage.image.copy()
        dtectBoxs_person = []
        dataTrackings_person.append(DataTracking(image, dtectBoxs_person, cid, type_id, count))
        # Debug_log(currentframe(), getframeinfo(currentframde()).filename, 'tcheck = ' +str(time.time()-tcheck))

    t1 = time.time()
    Debug_log(currentframe(), getframeinfo(currentframe()).filename, 't0 = ' +str(t1-t0))
    #detect person
    dataTrackings_person_output = yolo_person.detect(dataTrackings_person)
    # if debug_log:
    draw_boxs_only(copy.deepcopy(dataTrackings_person_output), 'detect')
    t1_ = time.time()
    dataTrackings_person_sort = pretreatment_tracking(dataTrackings_person_output, list_sort)
    
    # mapping and output
    dataTrackings_person_sort_mapping = mapping_dataTrackings_polygon(dataTrackings_person_sort, dataImages, coordinate_rois, coordinate_mapping_xyxys)
    draw_boxs_only(copy.deepcopy(dataTrackings_person_sort), 'sort')
    dataTrackingsOutput = personFalse.checkPersonFalse(dataTrackings_person_sort,dataTrackings_person_output)
    
    draw_boxs_only(copy.deepcopy(dataTrackingsOutput), 'dataTrackingsOutput')
    dict_data = {}
    if cv2.waitKey(0) == 27:
        sys.exit()
    return dict_data



def CountPerson(dataImages, 
                coordinate_rois,
                coodiPersonIns,
                coodiPersonOuts,
                list_sort,
                yolo_person,
                person_ins,
                person_outs,
                checkIDIn,
                checkIDOut):

    '''
        
    '''
    print("MAIN test".center(100,"*"))
    t0 = time.time()

    dataTrackings_person = []
    coordinate_roi_border = 100
    new_coordinate_rois = {}
    coordinate_mapping_xyxys = {}

    for idx, dataImage in enumerate(dataImages): 
        # Debug_log(currentframe(), getframeinfo(currentframe()).filename, idx)
        tcheck = time.time()
        cid = dataImage.cid
        type_id = dataImage.type_id
        count = dataImage.count
        dtectBoxs_person = []
        dataTrackings_person.append(DataTracking(dataImage.image, dtectBoxs_person, cid, type_id, count))
        # Debug_log(currentframe(), getframeinfo(currentframde()).filename, 'tcheck = ' +str(time.time()-tcheck))

    t1 = time.time()
    Debug_log(currentframe(), getframeinfo(currentframe()).filename, 't0 = ' +str(t1-t0))
    #detect person
    dataTrackings_person_output = yolo_person.detect(dataTrackings_person)
    # if debug_log:
    # draw_boxs_only(copy.deepcopy(dataTrackings_person_output), 'detect')
    t1_ = time.time()
    dataTrackings_person_sort = pretreatment_tracking(dataTrackings_person_output, list_sort)
    
    # mapping and output

    dataTrackings_person_sort = centerInROI(dataTrackings_person_sort, coodiPersonIns, coodiPersonOuts)

    for idx, dataTracking in enumerate(dataTrackings_person_sort):
        dtectBoxs = dataTrackings_person_sort[idx].dtectBoxs
        cid = dataTrackings_person_sort[idx].cid
        
        if cid not in checkIDIn:
            checkIDIn[cid] = {}
        if cid not in checkIDOut:
            checkIDOut[cid] = {}
        if cid not in person_ins:
            person_ins[cid] = 0
        if cid not in person_outs:
            person_outs[cid] = 0
        
        for dtectBox in dtectBoxs:
            id = dtectBox.id_tracking
            comeIn = dtectBox.ComeIn

            if comeIn == True:
                if id not in checkIDIn[cid]:
                    if id not in checkIDOut[cid]:
                        person_ins[cid]+=1
                        checkIDIn[cid][id] = time.time()
            elif comeIn == False:
                if id not in checkIDOut[cid]:
                    if id not in checkIDIn[cid]:
                        person_outs[cid]+=1
                        checkIDOut[cid][id] = time.time() 
    time_del = 5
    key_outs = {}
    for key_cid in checkIDOut:
        for keyOut in checkIDOut[key_cid]:
            if (time.time() - checkIDOut[key_cid][keyOut]) > time_del:
                if key_cid not in key_outs:
                    key_outs[key_cid] = [keyOut]
                else:
                    key_outs[key_cid].append(keyOut)
    
    for key_cid in key_outs:
        del checkIDOut[key_cid]

    key_ins = {}
    for key_cid in checkIDIn:
        for keyIn in checkIDIn[key_cid]:
            if (time.time() - checkIDIn[key_cid][keyIn]) > time_del:
                if key_cid not in key_ins:
                    key_ins[key_cid] = [keyIn]
                else:
                    key_ins[key_cid].append(keyIn)
    
    for key_cid in key_ins:
        del checkIDIn[key_cid]

    print('final person_ins, person_outs : ', person_ins, person_outs)
    draw_boxs_only_Count(copy.deepcopy(dataTrackings_person_sort), 'sort', person_ins, person_outs)


    #check line
    
    dict_data = {}
    if cv2.waitKey(1) == 27:
        sys.exit()
    return dict_data, person_ins, person_outs




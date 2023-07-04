import cv2
import numpy as np
import os

#debug log
from inspect import currentframe, getframeinfo
import datetime
import sys
debug_log = False
# Debug_log(currentframe(), getframeinfo(currentframe()).filename)
def Debug_log(cf, filename, name = None):
    if debug_log:
        ct = datetime.datetime.now()
        print(f'================ [{ct}] file {filename} , line : {cf.f_lineno} {name}')

def show_debug_log(list_dataTrackings):
    cid = 1#list_dataTrackings[0][0].cid

    Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'cid = '+str(cid))
    imgs = []
    for idx, dataTrackings in enumerate(list_dataTrackings):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'idx = '+str(idx))
        if len(dataTrackings) != 0:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'len(dataTrackings) = '+str(len(dataTrackings)))
            for i in range(len(dataTrackings)):
                if dataTrackings[i].cid == cid:
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'idx = '+str(idx))
                    imgs.append(cv2.resize(dataTrackings[i].frame, (450,350), interpolation = cv2.INTER_AREA))
        else:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'idx = '+str(idx))
            imgs.append(np.zeros((350,450,3), np.uint8))
    # if mode==0:
    zero_img = np.zeros((350,450,3), np.uint8)
    Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'len(imgs) = '+str(len(imgs)))
    im_h1 = cv2.hconcat([imgs[0], imgs[1], imgs[2]])
    if len(imgs)==6:
        im_h2 = cv2.hconcat([imgs[3], imgs[4], imgs[5]])
    elif len(imgs)==4:
        im_h2 = cv2.hconcat([imgs[3], zero_img, zero_img])
    img = cv2.vconcat([im_h1, im_h2])

    cv2.imshow('TBA', img)
    if cv2.waitKey(1) == 27:
        sys.exit()

def draw_box(dataTracking):
    image = dataTracking.frame.copy()
    dtectBoxs = dataTracking.dtectBoxs
    for dtectBox in dtectBoxs:
        x1, y1, x2, y2 = dtectBox.bbox[0], dtectBox.bbox[1], dtectBox.bbox[2], dtectBox.bbox[3]
        # print('x1, y1, x2, y2 : ', x1, y1, x2, y2)
        conf =  dtectBox.class_conf
        label = dtectBox.name_class
        id = dtectBox.id_tracking
        if dtectBox.class_ids != None:
            class_ids = dtectBox.class_ids
            str_class_ids = '-'.join(str(e) for e in class_ids)
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, str_class_ids + '-' + str(id))
        cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)
        if id==None:
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 1)
        elif  id!=None:
            cv2.putText(image, label + '-' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 1)                
        elif class_ids!=None:
            cv2.putText(image, str_class_ids + '-' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 1)
        else:
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 1)
        # cv2.imshow()
    return image

def draw_boxs_only(dataTrackings, txt_title):
    for idx, dataTracking in enumerate(dataTrackings):
        image = dataTrackings[idx].frame.copy()
        # cv2.imshow('image_'+txt_title, cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2))) )
        # cv2.waitKey(0)
        dtectBoxs = dataTrackings[idx].dtectBoxs
        name_show = str(dataTracking.cid)
        # cv2.imshow(str(dataTracking.cid) + '___', image)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename,  'len(dtectBoxs) :' + str(len(dtectBoxs)))
        for dtectBox in dtectBoxs:
            x1, y1, x2, y2 = int(dtectBox.bbox[0]), int(dtectBox.bbox[1]), int(dtectBox.bbox[2]), int(dtectBox.bbox[3])
            # print('x1, y1, x2, y2 : ', x1, y1, x2, y2)
            conf =  dtectBox.class_conf
            label = dtectBox.name_class
            id = dtectBox.id_tracking
            # print('*conf : ', conf)
            # print('*label : ', label)
            # print('*id : ', id)
            Debug_log(currentframe(), getframeinfo(currentframe()).filename,  'id-' + str(id))
            if dtectBox.class_ids != None:
                class_ids = dtectBox.class_ids
                # print('type(class_ids) : ', type(class_ids))
                # print('len(class_ids) : ', len(class_ids))
                # print('class_ids : ', class_ids)
                str_class_ids = '-'.join(str(e) for e in class_ids)
                # label = str_class_ids
                Debug_log(currentframe(), getframeinfo(currentframe()).filename, str_class_ids + '-' + str(id))
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)
            # print('**label : ', label)
            if id==None:
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
            elif  id!=None:
                cv2.putText(image, label + '-' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)                
            elif class_ids!=None:
                cv2.putText(image, str_class_ids + '_' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2)
            else:
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 2)
        dataTrackings[idx].frame = image
        # dict_video[name_show].write(image)
        resized_image = cv2.resize(image, (int(image.shape[1]*5/6), int(image.shape[0]*5/6))) 
        # cv2.imwrite(f'./event/draw_boxs_only_{txt_title}_{name_show}.jpg', image)
        cv2.imshow(txt_title + '_'+str(dataTracking.cid), resized_image)
        if cv2.waitKey(1) == 27:
            sys.exit()

def draw_boxs_only_Count(dataTrackings, txt_title, person_ins, person_outs):
    for idx, dataTracking in enumerate(dataTrackings):
        image = dataTrackings[idx].frame.copy()
        # cv2.imshow('image_'+txt_title, cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2))) )
        # cv2.waitKey(0)
        dtectBoxs = dataTrackings[idx].dtectBoxs
        name_show = str(dataTracking.cid)
        # cv2.imshow(str(dataTracking.cid) + '___', image)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename,  'len(dtectBoxs) :' + str(len(dtectBoxs)))
        for dtectBox in dtectBoxs:
            x1, y1, x2, y2 = int(dtectBox.bbox[0]), int(dtectBox.bbox[1]), int(dtectBox.bbox[2]), int(dtectBox.bbox[3])
            # print('x1, y1, x2, y2 : ', x1, y1, x2, y2)
            conf =  dtectBox.class_conf
            label = dtectBox.name_class
            id = dtectBox.id_tracking
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)
            if dtectBox.ComeIn == True:
                cv2.putText(image, label + '-' + str(id) + '-IN', (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
            elif dtectBox.ComeIn == False:
                cv2.putText(image, label + '-' + str(id) + '-OUT', (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
            else:
                cv2.putText(image, label + '-' + str(id) , (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
        cv2.putText(image, 'peron in' + ' :' + str(person_ins) , (100, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
        cv2.putText(image, 'person out' + ' :' + str(person_outs) , (100, 250), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
        dataTrackings[idx].frame = image
        # dict_video[name_show].write(image)
        resized_image = cv2.resize(image, (int(image.shape[1]), int(image.shape[0]))) 
        # cv2.imwrite(f'./event/draw_boxs_only_{txt_title}_{name_show}.jpg', image)
        cv2.imshow(txt_title + '_'+str(dataTracking.cid), resized_image)
        if cv2.waitKey(1) == 27:
            sys.exit()

def draw_boxs_only_fence(dataTrackings, txt_title):
    for idx, dataTracking in enumerate(dataTrackings):
        image = dataTrackings[idx].frame.copy()
        # cv2.imshow('image_'+txt_title, cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2))) )
        # cv2.waitKey(0)
        dtectBoxs = dataTrackings[idx].dtectBoxs
        name_show = str(dataTracking.cid)
        count = str(dataTracking.count)
        # cv2.imshow(str(dataTracking.cid) + '___', image)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename,  'len(dtectBoxs) :' + str(len(dtectBoxs)))
        for dtectBox in dtectBoxs:
            x1, y1, x2, y2 = int(dtectBox.bbox[0]), int(dtectBox.bbox[1]), int(dtectBox.bbox[2]), int(dtectBox.bbox[3])
            # print('x1, y1, x2, y2 : ', x1, y1, x2, y2)
            conf =  dtectBox.class_conf
            label = dtectBox.name_class
            id = dtectBox.id_tracking
            Debug_log(currentframe(), getframeinfo(currentframe()).filename,  'id-' + str(id))
            if dtectBox.class_ids != None:
                class_ids = dtectBox.class_ids
                # print('type(class_ids) : ', type(class_ids))
                # print('len(class_ids) : ', len(class_ids))
                # print('class_ids : ', class_ids)
                str_class_ids = '-'.join(str(e) for e in class_ids)
                label = str_class_ids
                Debug_log(currentframe(), getframeinfo(currentframe()).filename, str_class_ids + '-' + str(id))
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)
            if id==None:
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
            elif  id!=None:
                cv2.putText(image, label + '-' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)                
            elif class_ids!=None:
                cv2.putText(image, str_class_ids + '-' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
            else:
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
        dataTrackings[idx].frame = image
        # dict_video[name_show].write(image)
        resized_image = cv2.resize(image, (int(image.shape[1]*5/6), int(image.shape[0]*5/6))) 
        cv2.imwrite(f'./event_fence/{txt_title}_{name_show}_{str(count)}.jpg', image)
        # cv2.imshow(txt_title + '_'+str(dataTracking.cid), resized_image)
        # if cv2.waitKey(1) == 27:
        #     sys.exit()

def draw_boxs_only_personHoldThong(dataTrackings, txt_title):
    for idx, dataTracking in enumerate(dataTrackings):
        image = dataTrackings[idx].frame.copy()
        dtectBoxs = dataTrackings[idx].dtectBoxs
        name_show = str(dataTracking.cid)
        # cv2.imshow(str(dataTracking.cid) + '___', image)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename,  'len(dtectBoxs) :' + str(len(dtectBoxs)))
        for dtectBox in dtectBoxs:
            x1, y1, x2, y2 = int(dtectBox.bbox[0]), int(dtectBox.bbox[1]), int(dtectBox.bbox[2]), int(dtectBox.bbox[3])
            # print('x1, y1, x2, y2 : ', x1, y1, x2, y2)
            conf =  dtectBox.class_conf
            label = dtectBox.name_class
            id = dtectBox.id_tracking
            Debug_log(currentframe(), getframeinfo(currentframe()).filename,  'id-' + str(id))
            if dtectBox.class_ids != None:
                class_ids = dtectBox.class_ids
                # print('type(class_ids) : ', type(class_ids))
                # print('len(class_ids) : ', len(class_ids))
                # print('class_ids : ', class_ids)
                str_class_ids = '-'.join(str(e) for e in class_ids)
                label = str_class_ids
                Debug_log(currentframe(), getframeinfo(currentframe()).filename, str_class_ids + '-' + str(id))
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)
            if id==None:
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
            elif  id!=None:
                cv2.putText(image, label + '-' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)                
            elif class_ids!=None:
                cv2.putText(image, str_class_ids + '-' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
            else:
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
        dataTrackings[idx].frame = image
        # dict_video[name_show].write(image)
        resized_image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2))) 
        # cv2.imwrite(f'./event_holdthingdetect/draw_boxs_only_{txt_title}_{name_show}.jpg', image)
        cv2.imshow(txt_title + '_'+str(dataTracking.cid), resized_image)
        if cv2.waitKey(1) == 27:
            sys.exit()

def draw_boxs(dataTrackings, coordinate_rois, txt_title):
    for idx, dataTracking in enumerate(dataTrackings):
        image = dataTrackings[idx].frame.copy()
        dtectBoxs = dataTrackings[idx].dtectBoxs
        coordinate_roi = coordinate_rois[dataTracking.cid]
        x1c, y1c, x2c, y2c = coordinate_roi
        for dtectBox in dtectBoxs:
            x1, y1, x2, y2 = int(dtectBox.bbox[0]), int(dtectBox.bbox[1]), int(dtectBox.bbox[2]), int(dtectBox.bbox[3])
            # print('x1, y1, x2, y2 : ', x1, y1, x2, y2)
            conf =  dtectBox.class_conf
            label = dtectBox.name_class
            id = dtectBox.id_tracking
            if dtectBox.class_ids != None:
                class_ids = dtectBox.class_ids
                # print('type(class_ids) : ', type(class_ids))
                # print('len(class_ids) : ', len(class_ids))
                # print('class_ids : ', class_ids)
                str_class_ids = '-'.join(str(e) for e in class_ids)
                label = str_class_ids
                Debug_log(currentframe(), getframeinfo(currentframe()).filename, str_class_ids + '-' + str(id))
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)
            if id==None:
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
            elif  id!=None:
                cv2.putText(image, label + '-' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)                
            elif class_ids!=None:
                cv2.putText(image, str_class_ids + '-' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
            else:
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
        cv2.rectangle(image, (x1c, y1c), (x2c, y2c), (0,0,255), 2)
        cv2.putText(image, txt_title, (int(image.shape[1]/2), 100), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), 3)
        dataTrackings[idx].frame = image

def get_draw_boxs(dataImages, list_dataTrackings, coordinate_rois):
    list_img_ret = {}
    for dataImage in dataImages: 
         #Check img existed in list
        f_existed_img = False
        if list_img_ret is dict:
            for key in list(list_img_ret.keys()) :
                if key == dataImage.cid:
                    f_existed_img = True
        if f_existed_img is False:
            list_img_ret[dataImage.cid] = dataImage.image
            coordinate_roi = coordinate_rois[dataImage.cid]
            pts = np.array(coordinate_roi,np.int32)
            list_img_ret[dataImage.cid] = cv2.polylines(list_img_ret[dataImage.cid], [pts],True, (0, 0, 255), 3)               

    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    count_i = 0
    for dataTrackings in list_dataTrackings:
        for idx, dataTracking in enumerate(dataTrackings):
            #Check img existed in list
            f_existed_img = False
            if type(list_img_ret) is dict:
                for key in list(list_img_ret.keys()) :
                    if key == dataTracking.cid:
                        f_existed_img = True
            if f_existed_img is False:
                continue

            dtectBoxs = dataTrackings[idx].dtectBoxs
            count = str(dataTracking.count)

            for dtectBox in dtectBoxs:
                x1, y1, x2, y2 = int(dtectBox.bbox[0]), int(dtectBox.bbox[1]), int(dtectBox.bbox[2]), int(dtectBox.bbox[3])

                conf =  dtectBox.class_conf
                label = dtectBox.name_class
                id = dtectBox.id_tracking

                if dtectBox.class_ids != None:
                    class_ids = dtectBox.class_ids
                    str_class_ids = '-'.join(str(e) for e in class_ids)
                    label = str_class_ids

                cv2.rectangle(list_img_ret[dataTracking.cid] , (x1, y1), (x2, y2), colors[count_i], 2)
                if id==None:
                    cv2.putText(list_img_ret[dataTracking.cid] , label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, colors[count_i], 2)
                elif  id!=None:
                    cv2.putText(list_img_ret[dataTracking.cid] , label + '-' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, colors[count_i], 2)                
                elif class_ids!=None:
                    cv2.putText(list_img_ret[dataTracking.cid] , str_class_ids + '-' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, colors[count_i], 2)
                else:
                    cv2.putText(list_img_ret[dataTracking.cid] , label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, colors[count_i], 2)
            # cv2.imshow(str(dataTracking.cid), list_img_ret[dataTracking.cid])
        count_i+=1

    return list_img_ret

def draw_boxs_HSV(dataTrackings, coordinate_rois, txt_title):
    for idx, dataTracking in enumerate(dataTrackings):
        image = dataTrackings[idx].frame.copy()
        dtectBoxs = dataTrackings[idx].dtectBoxs
        for dtectBox in dtectBoxs:
            x1, y1, x2, y2 = int(dtectBox.bbox[0]), int(dtectBox.bbox[1]), int(dtectBox.bbox[2]), int(dtectBox.bbox[3])
            # print('x1, y1, x2, y2 : ', x1, y1, x2, y2)
            conf =  dtectBox.class_conf
            label = dtectBox.name_class
            id = dtectBox.id_tracking
            if dtectBox.class_ids != None:
                class_ids = dtectBox.class_ids
                str_class_ids = '-'.join(str(e) for e in class_ids)
                label = str_class_ids
                Debug_log(currentframe(), getframeinfo(currentframe()).filename, str_class_ids + '-' + str(id))
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)
            if id==None:
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
            elif  id!=None:
                cv2.putText(image, label + '-' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)                
            elif class_ids!=None:
                cv2.putText(image, str_class_ids + '-' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
            else:
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
        cv2.putText(image, txt_title, (int(image.shape[1]/2), 100), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), 3)
        dataTrackings[idx].frame = image

def draw_boxs_polygon(dataTrackings, coordinate_rois, coordinate_mapping_xyxys, txt_title):
    index_height = 5
    for idx, dataTracking in enumerate(dataTrackings):
        image = dataTrackings[idx].frame.copy()
        dtectBoxs = dataTrackings[idx].dtectBoxs
        coordinate_roi = coordinate_rois[dataTracking.cid]
        coordinate_mapping_xyxy = coordinate_mapping_xyxys[dataTracking.cid]
        pts = np.array(coordinate_roi,np.int32)
        # pts = pts.reshape((-1, 1, 2))                 
        image = cv2.polylines(image, [pts],True, (0, 0, 255), 3)

        for dtectBox in dtectBoxs:
            x1, y1, x2, y2 = int(dtectBox.bbox[0]), int(dtectBox.bbox[1]), int(dtectBox.bbox[2]), int(dtectBox.bbox[3])
            # print('x1, y1, x2, y2 : ', x1, y1, x2, y2)
            conf =  dtectBox.class_conf
            label = dtectBox.name_class
            id = dtectBox.id_tracking
            check_class_ids = None
            if dtectBox.class_ids != None:
                class_ids = dtectBox.class_ids
                check_class_ids = str(dtectBox.check_class_ids)
                # print('check_class_ids : ',check_class_ids)
                str_class_ids = '-'.join(str(e) for e in class_ids)
                label = str_class_ids
                Debug_log(currentframe(), getframeinfo(currentframe()).filename, str_class_ids + '-' + str(id))
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)
            if id==None:
                cv2.putText(image, label, (x1, y1-index_height), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 5)
            elif  id!=None:
                cv2.putText(image, label + '-' + str(id)+str(check_class_ids), (x1, y1-index_height), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 5)                
            elif class_ids!=None:
                cv2.putText(image, str_class_ids + '-' + str(id) + '-' +check_class_ids, (x1, y1-index_height), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 5)
            else:
                cv2.putText(image, label, (x1, y1-index_height), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 5)
        cv2.rectangle(image, (coordinate_mapping_xyxy[0], coordinate_mapping_xyxy[1]), (coordinate_mapping_xyxy[2], coordinate_mapping_xyxy[3]), (0,0,255), 2)
        cv2.putText(image, txt_title, (int(image.shape[1]/2), 100), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 6)
        
        dataTrackings[idx].frame = image

def draw_boxs_helmet(dataTrackings, coordinate_rois, txt_title, violates):
    for idx, dataTracking in enumerate(dataTrackings):
        image = dataTrackings[idx].frame.copy()
        dtectBoxs = dataTrackings[idx].dtectBoxs
        coordinate_roi = coordinate_rois[dataTracking.cid]
        x1c, y1c, x2c, y2c = coordinate_roi
        for dtectBox in dtectBoxs:
            x1, y1, x2, y2 = dtectBox.bbox[0], dtectBox.bbox[1], dtectBox.bbox[2], dtectBox.bbox[3]
            # print('x1, y1, x2, y2 : ', x1, y1, x2, y2)
            conf =  dtectBox.class_conf
            label = dtectBox.name_class
            id = dtectBox.id_tracking
            if dtectBox.class_ids != None:
                class_ids = dtectBox.class_ids
                # print('type(class_ids) : ', type(class_ids))
                # print('len(class_ids) : ', len(class_ids))
                # print('class_ids : ', class_ids)
                str_class_ids = '-'.join(str(e) for e in class_ids)
                label = str_class_ids
                Debug_log(currentframe(), getframeinfo(currentframe()).filename, str_class_ids + '-' + str(id))
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)
            if id==None:
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
            elif  id!=None:
                cv2.putText(image, label + '-' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)                
            elif class_ids!=None:
                cv2.putText(image, str_class_ids + '-' + str(id), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
            else:
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
        cv2.rectangle(image, (x1c, y1c), (x2c, y2c), (0,0,255), 2)
        cv2.putText(image, txt_title, (int(image.shape[1]/2), 100), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), 3)
        cv2.putText(image, str(violates[0][0]), (int(image.shape[1]/2), int(image.shape[1]/2)), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), 3)
        dataTrackings[idx].frame = image

def save_event(dict_data):
    name_folder = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")
    # os.mkdir('./event/' +name_folder)
    for cid in dict_data:
        image = dict_data[cid]["img"]
        datas = dict_data[cid]["data"]
        cv2.imwrite('./event/'+name_folder+'_image'+datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")+'.jpg', image)
        # print('datas : ', datas)
        # for data in datas:
        #     # print('dataaaa : ', data['typeObj'])
        #     txt = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")
        #     txt+=data['typeObj']
        #     cv2.imwrite('./event/'+name_folder+'/'+txt+'.jpg', data['cropImg'])
        break

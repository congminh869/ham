import cv2
import copy
import numpy as np
from shapely.geometry import Polygon,box,LineString
from include.yolo import DtectBox

#SORT
import sys
import skimage
sys.path.insert(0, './sort')
from sort import Sort
from include.yolo import YOLOv562, DataTracking


#debug log
from inspect import currentframe, getframeinfo
import datetime

debug_log = False
# Debug_log(currentframe(), getframeinfo(currentframe()).filename)
def Debug_log(cf, filename, name = None):
    if debug_log:
        ct = datetime.datetime.now()
        print(f'================ [{ct}] file {filename} , line : {cf.f_lineno} {name}')


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
    # if mode==1:
    #     zero_img = np.zeros((350,450,3), np.uint8)
    #     im_h1 = cv2.hconcat([imgs[0], imgs[1], zero_img])
    #     im_h2 = cv2.hconcat([imgs[2], imgs[3], zero_img])
    #     img = cv2.vconcat([im_h1, im_h2])

    #     cv2.imshow('turnel', img)
    #     cv2.waitKey(1)
    #     video_write.write(img)

def crop_img_polygon_old1(img, coordinate_roi, coordinate_roi_border):
    Debug_log(currentframe(), getframeinfo(currentframe()).filename, img.shape)
    polygon = np.array(coordinate_roi)

    rect = cv2.boundingRect(polygon)  # Hinh chu nhat bao quanh polygon
    x,y,w,h = rect
    # print('rect : ', rect)
    # cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,0), 2)
    x1 = x - coordinate_roi_border
    y1 = y - coordinate_roi_border
    x2 = x + w + coordinate_roi_border
    y2 = y + h + coordinate_roi_border

    if x1<0:
        x1 = 10
    if y1<0:
        y1=10
    if x2>img.shape[1]:
        x2 = img.shape[1]-10
    if y2>img.shape[0]:
        y2=img.shape[0]-10

    x, y = x1, y1 
    h = y2-y1
    w = x2-x1 

    croped = img[y:y+h, x:x+w].copy()
    polygon = polygon - polygon.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    # cv2.drawContours(mask, [polygon], -1, (255, 255, 255), -1, cv2.LINE_AA)

    # black_bg = cv2.bitwise_and(croped, croped, mask=mask)
     

    coordinate_mapping_xy = np.array([x,y])
    coordinate_mapping_xyxy = [x,y, x+w, y+h]
    new_coordinate_roi = np.array(coordinate_roi) - coordinate_mapping_xy

    #show
    # print('np.array(coordinate_roi) : ', np.array(coordinate_roi))
    # print('new_coordinate_roi : ', new_coordinate_roi)

    # frame = [cv2.putText(img, ''.join(str(e) for e in center_coordinate), center_coordinate, cv2.FONT_HERSHEY_SIMPLEX, 
    #                1, (255, 0, 0), 2, cv2.LINE_AA) for center_coordinate in coordinate_roi]
    # print('x1, y1, x2, y2: ', x1, y1, x2, y2)
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
    # resized_image = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4))) 
    # cv2.imshow('frame',resized_image)

    # image = cv2.polylines(croped, [np.array(new_coordinate_roi,np.int32)],True, (0, 0, 255), 3)
    # resized_croped = cv2.resize(image, (int(croped.shape[1]/4), int(croped.shape[0]/4))) 
    # cv2.imshow('image_croped',resized_croped)

    # resized = cv2.resize(croped, (960,640), interpolation = cv2.INTER_AREA)
    # cv2.imshow('crop_polygon',resized)
    
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     sys.exit()
    return croped, new_coordinate_roi, coordinate_mapping_xyxy

def crop_img_polygon_old2(img, coordinate_roi, coordinate_roi_border):
    '''
    older version before day 30/4-1/5
    '''
    # print('\n\n\n ============>  shape' ,img.shape)    (1520, 2688, 3)


    Debug_log(currentframe(), getframeinfo(currentframe()).filename, img.shape)
    polygon = np.array(coordinate_roi,np.int32) 
    # centroid polygon
    center_x = int(np.mean(polygon[:, 0]))
    center_y = int(np.mean(polygon[:, 1]))
    
    new_polygon = polygon.copy()
    
    height,width,c = img.shape 

    for i in range(len(polygon)):        
        # K/c center -> point
        distance = np.sqrt((polygon[i][0] - center_x)**2 + (polygon[i][1] - center_y)**2) # sqrt((x1-x2)^2 + (y1-y2)^2)
        
        new_x = polygon[i][0] + int(coordinate_roi_border * (polygon[i][0] - center_x) / distance) 
        new_y = polygon[i][1] + int(coordinate_roi_border * (polygon[i][1] - center_y) / distance)

        if new_x < 0:
            new_x = 10
        if new_y < 0:
            new_y = 10
        if new_x > width:
            new_x = width-10
        if new_y > height:
            new_y = height-10
        
        
        new_polygon[i] = [new_x, new_y]

    # pts_old = np.array(polygon,np.int32) 
    pts_new = np.array(new_polygon,np.int32)  
    
    # draw
    # img = cv2.polylines(img,[pts_new],True,(0,255,255),3)
    img = cv2.polylines(img,[polygon],True, (0, 0, 255), 3)
    

    rect = cv2.boundingRect(new_polygon)  # Hinh chu nhat bao quanh new_polygon
    x,y,w,h = rect

    croped = img[y:y+h, x:x+w].copy()
    new_polygon = new_polygon - new_polygon.min(axis=0)
    #print("\n\n",new_polygon, "\n\n")

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [new_polygon.astype(int)], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # cv2.imshow('croped',croped)
    black_bg = croped#cv2.bitwise_and(croped, croped, mask=mask)
    # cv2.imshow('black_bg',black_bg)
    # cv2.waitKey(0)

    coordinate_mapping_xyxy = [x,y, x+w, y+h]
    coordinate_mapping_xy = np.array([x,y])
    new_coordinate_roi = np.array(coordinate_roi) - coordinate_mapping_xy
    MIN_X, MIN_Y, _, _ = coordinate_mapping_xyxy
    # for point in range(len(coordinate_roi)):
    #     coordinate_roi[point][0] -= MIN_X
    #     coordinate_roi[point][1] -= MIN_Y
        
    # new_coordinate_roi = np.array(coordinate_roi) - coordinate_mapping_xy

    
    #show
    # print('np.array(coordinate_roi) : ', np.array(coordinate_roi))
    # print('new_coordinate_roi : ', new_coordinate_roi)

    # frame = [cv2.putText(img, ''.join(str(e) for e in center_coordinate), center_coordinate, cv2.FONT_HERSHEY_SIMPLEX, 
    #                1, (255, 0, 0), 2, cv2.LINE_AA) for center_coordinate in coordinate_roi]
    # print('x1, y1, x2, y2: ', x1, y1, x2, y2)
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
    # resized_image = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4))) 
    # cv2.imshow('frame',resized_image)

    # image = cv2.polylines(black_bg, [np.array(polygon,np.int32)],True, (0, 0, 255), 3)
    # resized_croped = cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4))) 
    # cv2.imshow('image_croped',resized_croped)

    # resized = cv2.resize(croped, (960,640), interpolation = cv2.INTER_AREA)
    # cv2.imshow('crop_polygon',resized)
    
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     sys.exit()
    return black_bg, new_coordinate_roi, coordinate_mapping_xyxy

def crop_img_polygon(img, coords, border):
    '''
        code by Son after 30-4-2023

    '''
    def intersection_line(line1,line2):
        x1, y1, x2, y2, x3, y3, x4, y4 = line1[0][0],line1[0][1],line1[1][0],line1[1][1],\
                                            line2[0][0],line2[0][1],line2[1][0],line2[1][1]
        # Tính các hệ số a, b, c, d, e, f
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        d = y4 - y3
        e = x3 - x4
        f = x4 * y3 - x3 * y4

        # Tính tọa độ giao điểm (x, y)
        if a*e - b*d == 0:
            return None
        else:
            x = (float(b*f) - float(c*e)) / (float(a*e) - float(b*d))
            y = (float(c*d) - float(a*f)) / (float(a*e) - float(b*d))     
            # print("x,y = ",x,y)
            return (int(x), int(y))

    def draw_rec(p1,p2,border):
        x1, y1 = p1
        x2, y2 = p2
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        height = border
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        rect = ((x1 + x2) / 2, (y1 + y2) / 2), (length, height), angle
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box

    def get_bouding_poly(coords,border,width,height):
        polygon = np.array(coords)
        list_line = []
        for i in range(len(polygon)):
            if i < len(polygon)-1:
                p1 = polygon[i]
                p2 = polygon[i+1]
            if i == len(polygon)-1:    
                p1 = polygon[i]
                p2 = polygon[0]      
            box = draw_rec(p1,p2,border)
            rec1 = Polygon([p1,p2,(box[3][0],box[3][1]),(box[0][0],box[0][1])])
            rec2 = Polygon([p1,p2,(box[2][0],box[2][1]),(box[1][0],box[1][1])])
            if rec1.intersects(Polygon(coords)):    
                list_line.append(((box[2][0],box[2][1]), (box[1][0],box[1][1])))
            else:
                list_line.append(((box[3][0],box[3][1]), (box[0][0],box[0][1])))
        list_point = []
        for i in range(len(list_line)):
            if i < len(list_line) - 1:
                l1 = list_line[i]
                l2 = list_line[i+1]
            else:
                l1 = list_line[i]
                l2 = list_line[0]
            list_point.append(intersection_line(l1,l2))
        return list_point

    def get_crop_img(polygon1,polygon2):
        intersection_polygon = polygon1.intersection(polygon2)
        # Lấy tọa độ các đỉnh của phần giao nhau
        intersection_vertices = np.array(intersection_polygon.exterior.coords, dtype=np.int32)
        # Tìm bounding box của phần giao nhau
        x, y, w, h = cv2.boundingRect(intersection_vertices)
        crop_img = img[y:y+h, x:x+w]
        # Tạo mask đa giác bao quanh phần cắt
        mask = np.zeros_like(crop_img)
        # Vẽ đa giác bao quanh phần cắt
        cv2.fillPoly(mask, [intersection_vertices - (x, y)], (255, 255, 255))
        crop_img = cv2.bitwise_and(crop_img, mask)

        mapping_xyxy = [x,y, x+w, y+h]
        return crop_img, mapping_xyxy

    
    height,width,c = img.shape 
    bounding_poly = get_bouding_poly(coords,border,width,height)
    # time_2 = time.time()
    
    polygon = np.array(coords)
    pts1 = np.array(polygon,np.int32) # old
    pts2 = np.array(bounding_poly,np.int32) # new

    # img = cv2.polylines(img,[pts1],True,(0,0,255),4)
    # cv2.polylines(img, [pts2], isClosed=True, color=(255,0, 0), thickness=4)
    
    # Tìm phần giao bounding polygon với ảnh gốc
    img_shape = np.array([[0,0],[width,0],[width,height],[0,height]], np.int32)
    polygon1 = Polygon(pts2)
    polygon2 = Polygon(img_shape) 
    

    crop_img, mapping_xyxy = get_crop_img(polygon1,polygon2)
    new_coord = polygon - np.array([mapping_xyxy[0],mapping_xyxy[1]])
    return crop_img, new_coord, mapping_xyxy

def crop_img_polygon_fence(img, coordinate_roi, coordinate_roi_border):

    # print('\n\n\n ============>  shape' ,img.shape)    (1520, 2688, 3)


    Debug_log(currentframe(), getframeinfo(currentframe()).filename, img.shape)
    polygon = np.array(coordinate_roi,np.float32) 
    # centroid polygon
    center_x = int(np.mean(polygon[:, 0]))
    center_y = int(np.mean(polygon[:, 1]))
    
    new_polygon = polygon.copy()
    
    height,width,c = img.shape 

    for i in range(len(polygon)):        
        # K/c center -> point
        distance = np.sqrt((polygon[i][0] - center_x)**2 + (polygon[i][1] - center_y)**2) # sqrt((x1-x2)^2 + (y1-y2)^2)
        
        new_x = polygon[i][0] + int(coordinate_roi_border * (polygon[i][0] - center_x) / distance) 
        new_y = polygon[i][1] + int(coordinate_roi_border * (polygon[i][1] - center_y) / distance)

        if new_x < 0:
            new_x = 10
        if new_y < 0:
            new_y = 10
        if new_x > width:
            new_x = width-10
        if new_y > height:
            new_y = height-10
        
        
        new_polygon[i] = [new_x, new_y]

    # pts_old = np.array(polygon,np.int32) 
    pts_new = np.array(new_polygon,np.int32)  
    
    # draw
    # img = cv2.polylines(img,[pts_new],True,(0,255,255),3)
    # img = cv2.polylines(img,[polygon],True,(0, 0, 255),3)
    

    rect = cv2.boundingRect(new_polygon)  # Hinh chu nhat bao quanh new_polygon
    x,y,w,h = rect

    croped = img[y:y+h, x:x+w].copy()
    new_polygon = new_polygon - new_polygon.min(axis=0)
    #print("\n\n",new_polygon, "\n\n")

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [new_polygon.astype(int)], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # cv2.imshow('croped',croped)
    black_bg = cv2.bitwise_and(croped, croped, mask=mask)
    # cv2.imshow('black_bg',black_bg)
    # cv2.waitKey(0)

    coordinate_mapping_xyxy = [x,y, x+w, y+h]
    coordinate_mapping_xy = np.array([x,y])
    new_coordinate_roi = np.array(coordinate_roi) - coordinate_mapping_xy
    MIN_X, MIN_Y, _, _ = coordinate_mapping_xyxy
    # for point in range(len(coordinate_roi)):
    #     coordinate_roi[point][0] -= MIN_X
    #     coordinate_roi[point][1] -= MIN_Y
        
    # new_coordinate_roi = np.array(coordinate_roi) - coordinate_mapping_xy

    
    #show
    # print('np.array(coordinate_roi) : ', np.array(coordinate_roi))
    # print('new_coordinate_roi : ', new_coordinate_roi)

    # frame = [cv2.putText(img, ''.join(str(e) for e in center_coordinate), center_coordinate, cv2.FONT_HERSHEY_SIMPLEX, 
    #                1, (255, 0, 0), 2, cv2.LINE_AA) for center_coordinate in coordinate_roi]
    # print('x1, y1, x2, y2: ', x1, y1, x2, y2)
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
    # resized_image = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4))) 
    # cv2.imshow('frame',resized_image)

    # image = cv2.polylines(black_bg, [np.array(polygon,np.int32)],True, (0, 0, 255), 3)
    # resized_croped = cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4))) 
    # cv2.imshow('image_croped',resized_croped)

    # resized = cv2.resize(croped, (960,640), interpolation = cv2.INTER_AREA)
    # cv2.imshow('crop_polygon',resized)
    
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     sys.exit()
    return black_bg, new_coordinate_roi, coordinate_mapping_xyxy

def crop_img(img, coordinate_roi):
    x, y, x2, y2 = coordinate_roi
    w = x2 -x
    h = y2 - y
    crop_img = img[y:y+h, x:x+w]
    return  crop_img

def mapping_coordinate(dtectBoxs, coordinate_roi):
    dtectBoxs_person = copy.deepcopy(dtectBoxs)
    x1c, y1c, x2c, y2c = coordinate_roi
    for idx, dtectBox in enumerate(dtectBoxs):
        if dtectBox.bbox!=None:
            x1, y1, x2, y2 = dtectBox.bbox[0], dtectBox.bbox[1], dtectBox.bbox[2], dtectBox.bbox[3]
            x1_mapping = x1c+x1
            y1_mapping = y1c+y1
            x2_mapping = x1c+x2
            y2_mapping = y1c+y2
            dtectBoxs_person[idx].bbox = [x1_mapping, y1_mapping, x2_mapping, y2_mapping]
    return dtectBoxs_person

def mapping_coordinate_polygon(dtectBoxs, coordinate_roi, coordinate_mapping_xy):
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

def mapping_dataTrackings_polygon(dataTrackings, dataImages, coordinate_rois, coordinate_mapping_xys):
    '''
        function mapping image_crop's coordinate with image_src' coordinate
    '''
    dataTrackings_mapping = []
    for idx, dataTracking in enumerate(dataTrackings): 
        cid = dataTracking.cid
        type_id = dataTracking.type_id
        count = dataTracking.count
        image = dataImages[idx].image#copy.deepcopy(dataImages[idx].image)
        coordinate_roi = coordinate_rois[dataTracking.cid]

        coordinate_mapping_xy = coordinate_mapping_xys[dataTracking.cid]
        dtectBoxs = dataTracking.dtectBoxs
        dtectBoxs_mapping = mapping_coordinate_polygon(dtectBoxs, coordinate_roi, coordinate_mapping_xy)
        dataTrackings_mapping.append(DataTracking(image, dtectBoxs_mapping, cid, type_id, count))
    return dataTrackings_mapping

#################################################
def mapping_dataTrackings(dataTrackings, dataImages, coordinate_rois):
    '''
        function mapping image_crop's coordinate with image_src' coordinate
    '''
    dataTrackings_mapping = []
    for idx, dataTracking in enumerate(dataTrackings): 
        cid = dataTracking.cid
        type_id = dataTracking.type_id
        count = dataTracking.count
        image = dataImages[idx].image#copy.deepcopy(dataImages[idx].image)
        coordinate_roi = coordinate_rois[dataTracking.cid]
        dtectBoxs = dataTracking.dtectBoxs
        dtectBoxs_mapping = mapping_coordinate(dtectBoxs, coordinate_roi)
        dataTrackings_mapping.append(DataTracking(image, dtectBoxs_mapping, cid, type_id, count))
    return dataTrackings_mapping

def get_rects(dataImages, coordinate_rois, labels_allow_helmet):
    rects_list = []
    labels = []
    for dataImage in dataImages:
        x1, y1, x2, y2 = coordinate_rois[dataImage.cid]
        w = x2-x1
        h = y2-y1
        rects_list.append([[x1, y1, w, h]])

        labels.append(labels_allow_helmet[dataImage.cid])

    return rects_list, labels


def process_tracking(dets_to_sort, sort_tracker, cid):
    tracked_dets = sort_tracker.update(dets_to_sort)
    return tracked_dets

def tracking_to_datatracking(dict_tracked_dets, dataTrackings):
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    for idx, dataTracking in enumerate(dataTrackings): 
        dtectBoxs = []
        cid = dataTrackings[idx].cid
        tracked_dets = dict_tracked_dets[cid]

        for det in tracked_dets:
            conf = det[-1]
            # det = list(map(int,det))
            # for i in range(0, 4):
            #     if det[i]<0:
            #         det[i]=0
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            id = det[8]

            bbox = [x1, y1, x2, y2]
            name_class = det[9]
            id_tracking = id
            class_conf = conf 
            #list: 0:chuadung   1:mu_bh_trang 2:mu_bh_vang  3:mu_thuong   4:ao_dacam    5:ao_trang    6:ao_khac 7:quan_dacam  8:quan_khac   9:balo
            class_ids = det[10] #for detect uniform 

            # print('name_class : ', name_class)
            # print('id_tracking : ', id_tracking)
            # print('class_conf : ', class_conf)
            dtectBox = DtectBox()
            dtectBox.bbox = bbox
            dtectBox.name_class = name_class
            dtectBox.id_tracking = id_tracking 
            dtectBox.class_conf = class_conf 
            dtectBox.class_ids = class_ids 
            dtectBoxs.append(dtectBox)
        dataTrackings[idx].dtectBoxs = dtectBoxs
    return dataTrackings

def pretreatment_tracking(dataTrackings, list_sort):
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    dets_to_sort_cid = {}
    list_cid = []
    for idx, dataTracking in enumerate(dataTrackings):
        cid = dataTracking.cid
        dets_to_sort = np.empty((0,7))
        for dtectBox in dataTracking.dtectBoxs:
            x1, y1, x2, y2 = dtectBox.bbox
            conf = dtectBox.class_conf
            b_kps = [dtectBox.name_class, [dtectBox.class_ids]] #['ss',None]#['label', [2,3,4]]#
            dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, 0, b_kps])))
        dets_to_sort_cid[cid] = dets_to_sort
        list_cid.append(cid)

    dict_tracked_dets = {}
    for cid in list_cid:
        dict_tracked_det = process_tracking(dets_to_sort_cid[cid], list_sort[cid], cid)
        dict_tracked_dets[cid] = dict_tracked_det

    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    # dict_tracked_dets = {cid : process_tracking(dets_to_sort_cid[cid], list_sort[cid], cid) for cid in list_sort}
    dataTrackings_person_sort = tracking_to_datatracking(dict_tracked_dets, copy.deepcopy(dataTrackings))
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    return dataTrackings_person_sort

def pretreatment_tracking_loop_cid(dataTrackings, list_sort):
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    for idx, dataTracking in enumerate(dataTrackings):
        dtectBoxs = []
        cid = dataTracking.cid
        dets_to_sort = np.empty((0,7))
        image = dataTracking.frame
        _count = dataTracking.count
        # print('*******************************',cid, _count)
        for dtectBox in dataTracking.dtectBoxs:
            x1, y1, x2, y2 = dtectBox.bbox
            conf = dtectBox.class_conf
            b_kps = [dtectBox.name_class, [dtectBox.class_ids]] #['ss',None]#['label', [2,3,4]]#
            dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, 0, b_kps])))
        tracked_dets = list_sort[cid].update(dets_to_sort)
        for det in tracked_dets:
            conf = det[-1]
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            id = det[8]

            bbox = [x1, y1, x2, y2]
            name_class = det[9]
            id_tracking = id
            class_conf = conf 
            #list: 0:chuadung   1:mu_bh_trang 2:mu_bh_vang  3:mu_thuong   4:ao_dacam    5:ao_trang    6:ao_khac 7:quan_dacam  8:quan_khac   9:balo
            class_ids = det[10] #for detect uniform 

            dtectBox = DtectBox()
            dtectBox.bbox = bbox
            dtectBox.name_class = name_class
            dtectBox.id_tracking = id_tracking 
            dtectBox.class_conf = class_conf 
            dtectBox.class_ids = class_ids 
            dtectBoxs.append(dtectBox)

            if debug_log:
                cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)
                cv2.putText(image, str(id_tracking), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
        resized_image = cv2.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3))) 
        # cv2.imshow(str(_count) + str(cid), resized_image)

        dataTrackings[idx].dtectBoxs = dtectBoxs
    # if cv2.waitKey(0) == 27:
    #     sys.exit()

    return dataTrackings

def load_sort(cid):
    #load model sort
        # sort
    sort_max_age = 30
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh,
                       cid = cid) # {plug into parser}
    
    return sort_tracker

def test_crop_and_mapping_img(dataImages, coordinate_rois, yolo_person):
    '''
        test main
    '''
    #detect person
    Debug_log(currentframe(), getframeinfo(currentframe()).filename,'xoa')
    dataTrackings_person = []
    coordinate_roi_border = 20
    for idx, dataImage in enumerate(dataImages): 
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, idx)
        cid = dataImage.cid
        type_id = dataImage.type_id
        count = dataImage.count
        image = crop_img(dataImage.image.copy(), coordinate_rois[dataImage.cid])#dataImage.image.copy()
        dtectBoxs_person = []
        dataTrackings_person.append(DataTracking(image, dtectBoxs_person, cid, type_id, count))

    #model detect person: input dataTrackings_person
    dataTrackings_person_output = yolo_person.detect(dataTrackings_person)
    dataTracking_person_warning = ROI(dataTrackings_person_output, coordinate_roi_border, coordinate_rois=coordinate_rois)

    
    dataTrackings_person_mapping = mapping(dataTracking_person_warning, dataImages, coordinate_rois)
    draw_boxs(dataTrackings_person_mapping)
    
    for i in range(len(dataTrackings_person_mapping)):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, i)
        resized = cv2.resize(dataTrackings_person_mapping[i].frame, (640,360), interpolation = cv2.INTER_AREA)
        str_show = "person" + str(dataTrackings_person_mapping[i].cid);
        cv2.imshow(str_show, resized)
        cv2.waitKey(1)


def convert_event(dict_tracking, coordinate_rois):
    '''
        check event in dict_tracking
        {"cid": {"img": image, "data": data},
         "cid": {"img": image, "data": data}, 
         ...}
         data = {"typeObj":  type_warning, "warning": txt_warnings[type_warning], "cropImg": crop_img}

         **NOTE convert Hat to person
    '''

    dict_data = {}
    type_warnings = ["person", "uniform", "vihecle", "holdThingDet", "hat"]
    txt_warnings = {
                    "person": "Nguoi ra vao", 
                    "vehicle": "Xe ra vao", 
                    "holdThingDet": "Nguoi mang vat the", 
                    "hat": "Mu", 
                    "uniform": "Dong phuc", 
                    }
    Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'str(len(dict_tracking)) '+ str(len(dict_tracking)))
    for idx_tw, type_warning in enumerate(type_warnings):
        if type_warning in dict_tracking:
            # print()
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'type_warning '+ type_warning)
            dataTrackings = dict_tracking[type_warning]
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'str(len(dataTrackings)) '+ str(len(dataTrackings)))
            
            for idx, dataTracking in enumerate(dataTrackings):
                datas = []
                cid = dataTrackings[idx].cid
                dtectBoxs = dataTrackings[idx].dtectBoxs
                Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'str(len(dtectBoxs)) '+ str(len(dtectBoxs)))
                
                image = dataTrackings[idx].frame
                
                # if cid not in dict_data:
                #     image = dataTrackings[idx].frame
                # else:
                #     image = dict_data[cid]["img"]
                for dtectBox in dtectBoxs:
                    x1, y1, x2, y2 = int(dtectBox.bbox[0]), int(dtectBox.bbox[1]), int(dtectBox.bbox[2]), int(dtectBox.bbox[3])
                    x = int(x1)
                    y = int(y1)
                    w = int(x2-x1)
                    h = int(y2 - y1)
                    crop_img = image[y:y+h, x:x+w]
                    data = {"typeObj":  type_warning, "warning": txt_warnings[type_warning], "cropImg": crop_img}
                    datas.append(data)
                    #draw image put txt and draw reg
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)

                    cv2.putText(image, txt_warnings[type_warning], (x1, y1-30*idx_tw), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)

                Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'str(len(datas)) '+ str(len(datas)))
                # cv2.imshow('data_event', cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2))) )
                # cv2.waitKey(0)
                if len(datas)!=0:
                    if cid not in dict_data:
                        dict_data[cid] = {"img": image, "data": datas}
                    else:
                        dict_data[cid]["data"] = dict_data[cid]["data"]+datas
                
                # resized = cv2.resize(image, (int(image.shape[1]/3),int(image.shape[0]/3)), interpolation = cv2.INTER_AREA)
                # cv2.imshow(str(cid), image)
                # if cv2.waitKey(0) == 27:
                #     sys.exit()

    return dict_data

def bb_intersection_over_union(boxA, boxB):
    '''
        boxA, boxB = detection.gt, detection.pred
    '''
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def check_is_high_HSV_and_convert_event(dataTrackings, coordinate_rois):
    '''
        check ROI
        find IOU threshold

        {"cid": {"img": image, "data": data},
         "cid": {"img": image, "data": data}, 
         ...}

         “data” :[{“id”: 47_C, “open”: True/False }]

         coordinate_rois{cid: [{id: [x1, y1, x2, y2]}, {id: [x1, y1, x2, y2]}, ...],
                        cid: [{id: [x1, y1, x2, y2]}, {id: [x1, y1, x2, y2]}, ...],
                        ...}

    '''
    dict_data={}
    iou_threshold = 0.4
    for idx, dataTracking in enumerate(dataTrackings):
        dtectBoxs = dataTrackings[idx].dtectBoxs
        cid = dataTrackings[idx].cid
        if cid not in dict_data:
            dict_data[cid] = {"img": dataTrackings[idx].frame, "data": []} #dataTrackings[idx].frame
        
        idxbbs = [] # check duplicate bounding box for id
        for id in coordinate_rois[cid]:
            is_high = False
            ious = {}
            box_gt = coordinate_rois[cid][id]
            for idxbb, dtectBox in enumerate(dtectBoxs):
                if idxbb in idxbbs:
                    continue
                box_pred = dtectBoxs[idxbb].bbox
                iou = bb_intersection_over_union(box_gt, box_pred)
                # print('id iou ', id, iou)
                if iou>iou_threshold:
                    dict_data[cid]["data"].append({"id": id, "open": True})
                    is_high = True
                    idxbbs.append(idxbb)
                    break
            if is_high == False:
                dict_data[cid]["data"].append({"id": id, "open": False})
    # print(dict_data)            
    return dict_data

def check_is_clock_and_convert_event(dataTrackings):
    '''
        check ROI
        find IOU threshold
        {0: 'I', 1: "O'"}

        {"cid": {"img": image, "data": data},
         "cid": {"img": image, "data": data}, 
         ...}

         “data” :[{“id”: 47_C, “open”: True/False }]

         coordinate_rois{cid: [{id: [x1, y1, x2, y2]}, {id: [x1, y1, x2, y2]}, ...],
                        cid: [{id: [x1, y1, x2, y2]}, {id: [x1, y1, x2, y2]}, ...],
                        ...}

    '''
    name = ['I',"O'"]
    dict_data={}
    iou_threshold = 0.4
    data = []
    for idx, dataTracking in enumerate(dataTrackings):
        dtectBoxs = dataTrackings[idx].dtectBoxs
        cid = dataTrackings[idx].cid
        frame = dataTrackings[idx].frame
        for len_dtectBoxs in range(len(dtectBoxs)):
            x1, y1, x2, y2 = dtectBoxs[len_dtectBoxs].bbox
            name_class = dtectBoxs[len_dtectBoxs].name_class
            if name_class == name[0]:
                data = [{0: 0, "isHigh": True}]
                cv2.putText(frame, "True", (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
            else:
                data = [{0: 0, "isHigh": False}]
                cv2.putText(frame, "false", (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
            

        if cid not in dict_data:
            dict_data[cid] = {"img": frame, "data": data} #dataTrackings[idx].frame

    #     resized = cv2.resize(dict_data[cid]["img"], (int(dict_data[cid]["img"].shape[1]/3),int(dict_data[cid]["img"].shape[0]/3)), interpolation = cv2.INTER_AREA)
    #     cv2.imshow(str(cid), dict_data[cid]["img"])
    # if cv2.waitKey(1) == 27:
    #     sys.exit()
    # print(dict_data)            
    return dict_data

def convert_envet_fence(dataTrackings_person, dataTrackings_FS):
    '''
        dict_data = {"cid": {"img": image, "data": data},
                     "cid": {"img": image, "data": data}, 
                     ...}
        data = [{"typeObj": person, "num": 1},
                {"typeObj": fireSmoke, "num": 1}]
    '''
    dict_data = {}
    # print('len(dataTrackings_person) : ', len(dataTrackings_person))
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    for idx in range(len(dataTrackings_person)):
        # Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        num_persons = len(dataTrackings_person[idx].dtectBoxs)
        # print('num_persons : ', num_persons)
        num_FSs = len(dataTrackings_FS[idx].dtectBoxs)
        cid = dataTrackings_person[idx].cid
        frame = dataTrackings_person[idx].frame
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'idx: {idx}, len(person) : {str(num_persons)}' )
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'idx: {idx}, len(fireSmoke) : {str(num_FSs)}' )
        for num_person in range(num_persons):
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            x1,y1,x2,y2 = dataTrackings_person[idx].dtectBoxs[num_person].bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
            cv2.putText(frame, 'person', (x1, y1-30), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)


        for num_FSs in range(num_FSs):
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            x1,y1,x2,y2 = dataTrackings_FS[idx].dtectBoxs[num_FSs].bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
            cv2.putText(frame, 'fire_and_smoke', (x1, y1-30), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
        if num_persons>0 or num_FSs>0:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            if cid not in dict_data:
                dict_data[cid] = {"img": frame, "data": [{"typeObj": "person", "num": num_persons},
                                                         {"typeObj": "fireSmoke", "num": num_FSs},]} #dataTrackings[idx].frame
            name_folder = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")
            # cv2.imwrite(f'./output/fire_smoke_{name_folder}.jpg', frame)
    #     resized = cv2.resize(dict_data[cid]["img"], (int(dict_data[cid]["img"].shape[1]/3),int(dict_data[cid]["img"].shape[0]/3)), interpolation = cv2.INTER_AREA)
    #     cv2.imshow(str(cid), resized)
    # if cv2.waitKey(1) == 27:
    #     sys.exit()
    return dict_data

def meger_vehicle_person(dataTrackings_person, dataTrackings_vehicle):
    for idx in range(len(dataTrackings_person)):
        if len(dataTrackings_vehicle[idx].dtectBoxs)>0:
            dataTrackings_person[idx].dtectBoxs +=dataTrackings_vehicle[idx].dtectBoxs
    return dataTrackings_person

def split_vehicle_person(dataTrackings):
    dataTracking_person_warnings = []
    dataTracking_vihecle_warnings = []
    check_person = False
    check_vehicle = False
    dtecBoxs_person = []
    dtecBoxs_vihecle = []
    for idx_cid, dataTracking in enumerate(dataTrackings):
        # image_person = copy.deepcopy(dataTracking.frame)
        # image_vehicle = copy.deepcopy(dataTracking.frame)
        for idx, dtecBox in enumerate(dataTracking.dtectBoxs):
            # x1, y1, x2, y2 = dtecBox.bbox[0], dtecBox.bbox[1], dtecBox.bbox[2], dtecBox.bbox[3]
            if dtecBox.name_class == 'person':
                dtecBoxs_person.append(dtecBox)
                check_person = True
            else:
                dtecBoxs_vihecle.append(dtecBox)
                check_vehicle = True
        if check_person:
            dataTracking_person_warning = copy.deepcopy(dataTrackings[idx_cid])
            dataTracking_person_warning.dtectBoxs = dtecBoxs_person
            # for idx, dtecBox in enumerate(dtecBoxs_person):
                # x1, y1, x2, y2 = dtecBox.bbox[0], dtecBox.bbox[1], dtecBox.bbox[2], dtecBox.bbox[3]
                # cv2.rectangle(image_person, (x1, y1), (x2, y2), (255,255,0), 2)
                # print('person')
                # cv2.imshow('person', image_person)
                # cv2.waitKey(0)
            dataTracking_person_warnings.append(dataTracking_person_warning)
            check_person = False
        
        if check_vehicle:
            dataTracking_vihecle_warning = copy.deepcopy(dataTrackings[idx_cid])
            dataTracking_vihecle_warning.dtectBoxs = dtecBoxs_vihecle
            # for idx, dtecBox in enumerate(dtecBoxs_vihecle):
                # x1, y1, x2, y2 = dtecBox.bbox[0], dtecBox.bbox[1], dtecBox.bbox[2], dtecBox.bbox[3]
                # cv2.rectangle(image_vehicle, (x1, y1), (x2, y2), (0,255,0), 2)
                # print('vihecle')
                # cv2.imshow('vihecle', image_vehicle)
                # cv2.waitKey(0)
            dataTracking_vihecle_warnings.append(dataTracking_vihecle_warning)
            check_vehicle = False
        
        # for idx, dataTracking in enumerate(dataTracking_person_warnings):
        #     image = dataTracking_person_warnings[idx].frame.copy()
        #     dtectBoxs = dataTracking_person_warnings[idx].dtectBoxs
        #     name_show = str(dataTracking.cid)
        #     # cv2.imshow(str(dataTracking.cid) + '___', image)
        #     Debug_log(currentframe(), getframeinfo(currentframe()).filename,  'len(dtectBoxs) :' + str(len(dtectBoxs)))
        #     for dtectBox in dtectBoxs:
        #         x1, y1, x2, y2 = int(dtectBox.bbox[0]), int(dtectBox.bbox[1]), int(dtectBox.bbox[2]), int(dtectBox.bbox[3])
        #         cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)
        #         cv2.imshow('person', image)
        #         cv2.waitKey(0)

        dtecBoxs_person = []
        dtecBoxs_vihecle = []

    # for idx, dataTracking in enumerate(dataTracking_person_warnings):
    #     image = dataTracking_person_warnings[idx].frame.copy()
    #     dtectBoxs = dataTracking_person_warnings[idx].dtectBoxs
    #     name_show = str(dataTracking.cid)
    #     # cv2.imshow(str(dataTracking.cid) + '___', image)
    #     Debug_log(currentframe(), getframeinfo(currentframe()).filename,  'len(dtectBoxs) :' + str(len(dtectBoxs)))
    #     for dtectBox in dtectBoxs:
    #         x1, y1, x2, y2 = int(dtectBox.bbox[0]), int(dtectBox.bbox[1]), int(dtectBox.bbox[2]), int(dtectBox.bbox[3])
    #         cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)
    #         cv2.imshow('person', image)
    #         cv2.waitKey(0)

    # for idx, dataTracking in enumerate(dataTracking_vihecle_warnings):
    #     image = dataTracking_vihecle_warnings[idx].frame.copy()
    #     dtectBoxs = dataTracking_vihecle_warnings[idx].dtectBoxs
    #     name_show = str(dataTracking.cid)
    #     # cv2.imshow(str(dataTracking.cid) + '___', image)
    #     Debug_log(currentframe(), getframeinfo(currentframe()).filename,  'len(dtectBoxs) :' + str(len(dtectBoxs)))
    #     for dtectBox in dtectBoxs:
    #         x1, y1, x2, y2 = int(dtectBox.bbox[0]), int(dtectBox.bbox[1]), int(dtectBox.bbox[2]), int(dtectBox.bbox[3])
    #         cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)
    #         cv2.imshow('vihecle', image)
    #         cv2.waitKey(0)

    

    return dataTracking_person_warnings, dataTracking_vihecle_warnings

def dataImageToDataTracking_reg(dataImages):
    dataTrackings= []
    for idx, dataImage in enumerate(dataImages): 
        cid = dataImage.cid
        type_id = dataImage.type_id
        count = dataImage.count
        image = dataImage.image
        dtectBoxs_person = []
        dataTrackings.append(DataTracking(image, dtectBoxs_person, cid, type_id, count))

    return dataTrackings


# def PersonFalse(dataTrackings):
#     pass

if __name__ == '__main__':
    main()


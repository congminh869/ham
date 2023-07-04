import cv2
import numpy as np
from include.yolo import DtectBox, DataTracking
from shapely.geometry import Point, Polygon

def draw_box(dataTracking_warning, coordinate_roi):
    frame = dataTracking_warning.frame
    dtecBoxs = dataTracking_warning.dtectBoxs
    for dtecBox in dtecBoxs:
        x1, y1, x2, y2 = int(dtecBox.bbox[0]), int(dtecBox.bbox[1]), int(dtecBox.bbox[2]), int(dtecBox.bbox[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
    # cv2.rectangle(frame, (coordinate_roi[0], coordinate_roi[1]), (coordinate_roi[2], coordinate_roi[3]), (255,255,0), 2)
    dataTracking_warning.frame = frame
    return dataTracking_warning

def check_ROI(box, coordinate_roi):
    check_violation = False
    flexible_pixel = 10
    x1, y1, x2, y2 = box
    x1c, y1c, x2c, y2c = coordinate_roi
    x1cfp, y1cfp, x2cfp, y2cfp = x1c+flexible_pixel, y1c+flexible_pixel, x2c-flexible_pixel, y2c-flexible_pixel
    if (((x1<x1c<x2) or (x1<x2c<x2)) and ((y1c<y1<y2c) or (y1c<y2<y2c))) or \
         (((y1<y1c<y2) or (y1<y2c<y2)) and ((x1c<x1<x2c) or (x1c<x2<x2c))):
         check_violation = True
    #ROI shrinks flexible_pixel
    if (((x1<x1cfp<x2) or (x1<x2cfp<x2)) and ((y1cfp<y1<y2cfp) or (y1cfp<y2<y2cfp))) or \
         (((y1<y1cfp<y2) or (y1<y2cfp<y2)) and ((x1cfp<x1<x2cfp) or (x1cfp<x2<x2cfp))):
         check_violation = True
    return check_violation

def ROI(dataTrackings, coordinate_rois={1:[500, 300, 948, 638]}):
    '''
        
    '''
    dataTracking_warning_draws = []
    for idx, dataTracking in enumerate(dataTrackings): 
        dtecBoxs_warning = []
        dtecBoxs = dataTracking.dtectBoxs
        coordinate_roi = coordinate_rois[dataTracking.cid]
        for dtecBox in dtecBoxs:
            if check_ROI(dtecBox.bbox, coordinate_roi):
                #warning 
                dtecBoxs_warning.append(dtecBox)
        dataTrackings[idx].dtectBoxs = dtecBoxs_warning
        dataTracking_warning = dataTrackings[idx]
        dataTracking_warning.frame = dataTracking.frame.copy()
        # dataTracking_warning = DataTracking(dataTracking.frame.copy(), dtecBoxs_warning)
        dataTracking_warning_draw = draw_box(dataTracking_warning, coordinate_roi)
        dataTracking_warning_draws.append(dataTracking_warning_draw)
    return dataTracking_warning_draws

def ROI(dataTrackings, coordinate_roi_border, coordinate_rois={1:[500, 300, 948, 638]}):
    '''
        
    '''
    dataTracking_warning_draws = []
    for idx, dataTracking in enumerate(dataTrackings): 
        dtecBoxs_warning = []
        dtecBoxs = dataTracking.dtectBoxs
        h,w,c = dataTracking.frame.shape
        coordinate_roi = [coordinate_roi_border,coordinate_roi_border,w-coordinate_roi_border,h-coordinate_roi_border]#coordinate_rois[dataTracking.cid]
        for dtecBox in dtecBoxs:
            if check_ROI(dtecBox.bbox, coordinate_roi):
                #warning 
                dtecBoxs_warning.append(dtecBox)
        dataTrackings[idx].dtectBoxs = dtecBoxs_warning
        dataTracking_warning = dataTrackings[idx]
        dataTracking_warning.frame = dataTracking.frame.copy()
        # dataTracking_warning = DataTracking(dataTracking.frame.copy(), dtecBoxs_warning)
        dataTracking_warning_draw = draw_box(dataTracking_warning, coordinate_roi)
        dataTracking_warning_draws.append(dataTracking_warning_draw)
    return dataTracking_warning_draws

def mapping_coordinate_roi(coordinate_roi):
    MIN_X = 1000000
    MIN_Y = 1000000
    for i in range(len(coordinate_roi)):
        xc = coordinate_roi[i][0]
        yc = coordinate_roi[i][1]      
        if xc < MIN_X:
            MIN_X = xc
        if yc < MIN_Y:
            MIN_Y = yc
            
    return MIN_X,MIN_Y,coordinate_roi

def check_ROI_polygon(box, polygon):
    check_violation = False
    x1, y1, x2, y2 = box
    
    # 4 dinh cua bbox
    p1 = Point(x1,y1)
    p2 = Point(x2,y2)
    p3 = Point(x2,y1)
    p4 = Point(x1,y2)
    
    # Neu cham -> warning
    if polygon.contains(p1) and polygon.contains(p2) and polygon.contains(p3) and polygon.contains(p4):
        check_violation = False
    else:
        check_violation = True
    
    return check_violation

def ROI_polygon(dataTrackings, coordinate_roi_border, coordinate_rois=[]):
    dataTracking_warning_draws = []
    for idx, dataTracking in enumerate(dataTrackings): 
        dtecBoxs_warning = []
        dtecBoxs = dataTracking.dtectBoxs
        coordinate_roi = coordinate_rois[dataTracking.cid] 
        # resized = cv2.resize(dataTracking.frame, (960,640), interpolation = cv2.INTER_AREA)
        # cv2.imshow('crop',resized)
        polygon = np.array(coordinate_roi,np.int32)
        polygon = Polygon(polygon)
        for dtecBox in dtecBoxs:
            if check_ROI_polygon(dtecBox.bbox, polygon):
                #warning 
                dtecBoxs_warning.append(dtecBox)
                               
        dataTrackings[idx].dtectBoxs = dtecBoxs_warning
        dataTracking_warning = dataTrackings[idx]
        dataTracking_warning.frame = dataTracking.frame.copy()
        # dataTracking_warning = DataTracking(dataTracking.frame.copy(), dtecBoxs_warning)
        dataTracking_warning_draw = draw_box(dataTracking_warning, coordinate_roi)
        dataTracking_warning_draws.append(dataTracking_warning_draw)
    return dataTracking_warning_draws

def CenterInROI(box, polygon):
    check_violation = False
    x1, y1, x2, y2 = box

    center = Point(int((x1+x2)/2), int((y1+y2)/2))
    
    # 4 dinh cua bbox
    p1 = Point(x1,y1)
    p2 = Point(x2,y2)
    p3 = Point(x2,y1)
    p4 = Point(x1,y2)
    
    # Neu cham -> warning
    if polygon.contains(center):
        check_violation = True
    else:
        check_violation = False
    
    return check_violation

def centerInROI(dataTrackings, coordIns, coordOuts):
    dataTracking_warning_draws = []
    for idx, dataTracking in enumerate(dataTrackings): 
        dtecBoxs_warning = []
        dtecBoxs = dataTracking.dtectBoxs
        coordIn = coordIns[dataTracking.cid] 
        coordOut = coordOuts[dataTracking.cid] 

        # print('coordIn : ', coordIn)
        # print('coordOut : ', coordOut)
        polygonIn = np.array(coordIn,np.int32)
        polygonIn = Polygon(polygonIn)

        polygonOut = np.array(coordOut,np.int32)
        polygonOut = Polygon(polygonOut)

        for idxbb, dtecBox in enumerate(dtecBoxs):
            if CenterInROI(dtecBox.bbox, polygonIn):
                #warning 
                dataTrackings[idx].dtectBoxs[idxbb].ComeIn = True
            elif CenterInROI(dtecBox.bbox, polygonOut):
                dataTrackings[idx].dtectBoxs[idxbb].ComeIn = False
        #draw polygon
        # cv2.imshow('a',dataTrackings[idx].frame)
        # cv2.waitKey(0)
        cv2.drawContours(dataTrackings[idx].frame, [np.array(coordIn)], -1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.drawContours(dataTrackings[idx].frame, [np.array(coordOut)], -1, (0, 0, 255), 2, cv2.LINE_AA)
    return dataTrackings

if __name__ == '__main__':
    frame = cv2.imread('/home/minhssd/Pictures/Screenshot from person3.mp4.png')

    x1, y1, x2, y2 = 295, 151, 948, 638

    cv2.regtagle(frame, (x1, y1), (x2, y2), (255,0,0),2)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)


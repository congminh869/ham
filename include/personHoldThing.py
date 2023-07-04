import cv2
import copy
import numpy as np
import time
from multiprocessing.pool import Pool
# from util import DataTracking, DtectBox
#debug log
from inspect import currentframe, getframeinfo
import datetime

debug_log = False
# Debug_log(currentframe(), getframeinfo(currentframe()).filename)
def Debug_log(cf, filename, name = None):
    if debug_log:
        ct = datetime.datetime.now()
        print(f'================ [{ct}] file {filename} , line : {cf.f_lineno} {name}')
        
class BGS_HB:
    def __init__(self, min_area=1000, gaussian_blur_kernel=3, dilate_kernel=3,
                    dilate_iterations=6, low_threshold=15, high_threshold=255, clean_bg_num=5):
        self.min_area = min_area
        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.dilate_kernel = dilate_kernel
        self.dilate_iterations = dilate_iterations
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.clean_bg_num = clean_bg_num
        self.background_images = {}

    def handle_batch_images(self, dataTrackings):
        images = []
        for i, dataTracking in enumerate(dataTrackings):
            cid, frame_count, image = dataTracking.cid, i, dataTracking.frame
            # print("*"*50)
            # print("frame_count: ", frame_count, dataTracking.count)
            # print("*"*50)
            
            
            image = cv2.resize(image,(640, 480))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.GaussianBlur(image, (self.gaussian_blur_kernel, self.gaussian_blur_kernel), 0)
            
            if self.background_images.get(cid) is None:
                self.background_images[cid] = {}
                self.background_images[cid][frame_count] = image.copy()

            images.append((cid, frame_count, image))

        potential_boxes = {}
        for image in images:
            cid, potential_box = self.handle_one_image(*image)
            potential_boxes[cid] = potential_box
        
        return potential_boxes

    
    def handle_one_image(self, cid, frame_count, image):
        count_idx = frame_count
        background_image = None
        while count_idx > frame_count - self.clean_bg_num:
            count_idx -= 1
            if self.background_images[cid].get(count_idx) is not None:
                background_image = self.background_images[cid].get(count_idx)
                break

        potential_box=[]
        self.background_images[cid][frame_count] = image

        if background_image is not None:
            frame = cv2.absdiff(background_image, image)
            frame = cv2.dilate(frame, np.ones((self.dilate_kernel, self.dilate_kernel)), iterations=self.dilate_iterations)
            _, frame = cv2.threshold(frame, self.low_threshold, self.high_threshold, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)
                    rx,ry,rw,rh = cv2.boundingRect(approx)
                    potential_box.append([rx, ry, rx+rw, ry+rh])

            self.clean_background(cid, frame_count)

        return (cid, frame_count), potential_box

    def clean_background(self, cid, frame_count):
        for key in list(self.background_images[cid].keys()):
            if key < frame_count - self.clean_bg_num:
                del self.background_images[cid][key]

    def remove_source(self, cid):
        del self.background_images[cid]



def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain[1]) / 2, (img1_shape[0] - img0_shape[0] * gain[0]) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, [0, 2]] /= gain[1]
    coords[:, [1, 3]] /= gain[0]
    clip_coords(coords, img0_shape)
    return coords.astype(np.int64)


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2

def check_bbox_inside(bb1, bb2, iou_threshold=0.85, min_threshold=0.05):
    x11, y11, x12, y12 = bb1
    x21, y21, x22, y22 = bb2
    
    assert x11 < x12
    assert y11 < y12
    assert x21 < x22
    assert y21 < y22

    # determine the coordinates of the intersection rectangle
    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (x12 - x11) * (y12 - y11)
    bb2_area = (x22 - x21) * (y22 - y21)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area

    return intersection_area > bb2_area * iou_threshold and bb2_area > bb1_area * min_threshold

def preprecess_output(batch_contours, dataTrackings, high_threshold=1.2):
    # high_threshold = 1.2
    person_bboxes = {}
    for i, dataTracking in enumerate(dataTrackings):
        person_bboxes[(dataTracking.cid, i)] = [dtectBox.bbox for dtectBox in dataTracking.dtectBoxs]
        # person_bboxes[dataTracking.cid].appe
    
    # print('batch_contours = ',batch_contours)
    
    #
    dict_PersonHoldThing = {}
    for pair_key, contours in batch_contours.items():
        ccid, frame_count = pair_key
        frame = copy.deepcopy(dataTrackings[frame_count].frame)
        # frame2 = copy.deepcopy(dataTrackings[ccid].frame)
        for (hho_bbox, person_bbox, idx) in [(hho_bbox, person_bbox, idx) for hho_bbox in contours for idx, person_bbox in enumerate(person_bboxes[pair_key])]:
            # print('hho_bbox, person_bbox, idx : ', hho_bbox, person_bbox, idx)
            # hho_bbox = scale_coords((480, 640), np.array([hho_bbox]).astype(np.float64), frame.shape, None)[0]
            x11, y11, x12, y12 = hho_bbox
            x21, y21, x22, y22 = person_bbox
            differ_high = y21 - y11
            person_height = y22 - y21
            
            #check_bbox_inside and check object taller than allow
            if check_bbox_inside(hho_bbox, person_bbox, iou_threshold=0.85, min_threshold=0.05) and differ_high / person_height > high_threshold:
                # cv2.rectangle(frame, (x11, y11), (x12, y12), (0,255,0), 3)
                # cv2.rectangle(frame, (x21, y21), (x22, y22), (0,255,255), 3)
                # cv2.imwrite(f"./output/result_{time.time()}.jpg", frame)
                # cv2.imshow('frame',cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4))))
                if (ccid, frame_count) in dict_PersonHoldThing:
                    dict_PersonHoldThing[(ccid, frame_count)].append(idx)
                else:
                    dict_PersonHoldThing[(ccid, frame_count)] = [idx]
                Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'ccid : {ccid}, idx : {idx}')
                # x1, y1, x2, y2 = dataTrackings[ccid].dtectBoxs[idx].bbox
                # cv2.rectangle(frame2, (x1, y1), (x2, y2), (0,255,255), 3)
                # cv2.imshow('frame2',cv2.resize(frame2, (int(frame2.shape[1]/4), int(frame2.shape[0]/4))))
                # cv2.waitKey(1)

    dataTracking_person_warnings = []
    for idx_cid in dict_PersonHoldThing:
        # print(idx_cid)
        ccid, frame_count = idx_cid
        frame3 = copy.deepcopy(dataTrackings[frame_count].frame)
        dataTracking_person_HTD = []
        for idx_dtb in dict_PersonHoldThing[idx_cid]:
            # print(idx_dtb)
            # print(len(dataTrackings[frame_count].dtectBoxs))
            dataTracking_person_HTD.append(dataTrackings[frame_count].dtectBoxs[idx_dtb])
            x1, y1, x2, y2 = dataTrackings[frame_count].dtectBoxs[idx_dtb].bbox
            cv2.rectangle(frame3, (x1, y1), (x2, y2), (0,255,255), 3)
            # cv2.imshow('frame3'+str(idx_dtb),cv2.resize(frame3, (int(frame3.shape[1]/3), int(frame3.shape[0]/3))))
            # cv2.waitKey(1)
        
        dataTracking_person_warning = copy.deepcopy(dataTrackings[frame_count])
        dataTracking_person_warning.dtectBoxs = dataTracking_person_HTD
        dataTracking_person_warnings.append(dataTracking_person_warning)
    
    return dataTracking_person_warnings

if __name__ == '__main__':
    bgshb = BGS_HB()
    
    cap = cv2.VideoCapture('videos/gay_1.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame,(640, 480))
        draw_frame = frame.copy()
        cid = 10
        dataTrackings = [DataTracking(frame, [], cid, 123, 0)]
        batch_contours = bgshb.handle_batch_images(dataTrackings=dataTrackings)
        # batch_contours = bgshb.handle_batch_images_parallel(dataTrackings=dataTrackings)
        for ccid, contours in batch_contours.items():
            for contour in contours:
                rx1, ry1, rx2, ry2 = contour
                cv2.rectangle(draw_frame, (rx1, ry1), (rx2, ry2), (0,255,0), 3)

        key = cv2.waitKey(1)
        if key==27:
            break
        cv2.imshow("show", draw_frame)
         
    cap.release()
    cv2.destroyAllWindows()



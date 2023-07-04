#!/usr/bin/env python
 
import cv2
import time
from shapely.geometry import Point, Polygon
import numpy as np
def crop_img_polygon(img, coordinate_roi):

    polygon = np.array(coordinate_roi)
    rect = cv2.boundingRect(polygon)  # Hinh chu nhat bao quanh polygon
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()
    polygon = polygon - polygon.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [polygon], -1, (255, 255, 255), -1, cv2.LINE_AA)

    # black_bg = cv2.bitwise_and(croped, croped, mask=mask)
    

    resized = cv2.resize(croped, (960,640), interpolation = cv2.INTER_AREA)
    cv2.imshow('crop',resized)
    cv2.waitKey(0)
    return  croped

def draw_cricle(frame, center_coordinates):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # fontScale
    fontScale = 2
       
    # Blue color in BGR
    color = (0, 0, 255)
      
    # Line thickness of 2 px
    thickness = 2
    a = [cv2.putText(frame, '1', center_coordinate, font, 
                   fontScale, color, thickness, cv2.LINE_AA) for center_coordinate in center_coordinates]

    frame = cv2.polylines(frame, [np.array(center_coordinates,np.int32)],True, (0, 0, 255), 3)
    # polygon_ = Polygon(np.array(center_coordinates))
    # p1 = Point(348*4,168*4) # in
    # p2 = Point(73*4,264*4) # out
    # cv2.circle(frame, (348*4,168*4), 10, color, thickness)
    # cv2.circle(frame, (73*4,264*4), 10, color, thickness)
    # print('polygon_.contains(p1) : ', polygon_.contains(p1))
    # print('polygon_.contains(p2) : ', polygon_.contains(p2))

def check_resized():
    image = cv2.imread('/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/image/3person.jpg')
    h, w, c = image.shape
    w_scale = 640/w
    h_scale = 640/h
    x1_scale, y1_scale, x2_scale, y2_scale = 124, 178, 295, 290
    x1, y1, x2, y2 = int(x1_scale/w_scale), int(y1_scale/h_scale), int(x2_scale/w_scale), int(y2_scale/h_scale)
    resized_image = cv2.resize(image, (640, 640), interpolation = cv2.INTER_AREA) 


    # cv2.rectangle(resized_image, (x1_scale, y1_scale), (x2_scale, y2_scale), (255,255,0), 2)
    # cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)
    # cv2.imshow('image', image)
    # cv2.imshow('resized_image', resized_image)
    # cv2.waitKey(0)
    cv2.imwrite('3person.jpg', resized_image)





def video():
    # Start default camera
    cap = cv2.VideoCapture('/home/evnadmin/Documents/AI_hoabinh/video/video_test.mkv');
    count_frame = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    print(size)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('=================fps : ', fps)
    while(cap.isOpened()):

        ret, frame = cap.read()
        if count_frame>=9040:
            if ret == True:
                # center_coordinates = np.array([[37, 338],[50, 35],[342, 10],[650, 31], [604, 322],[340, 368]])*4
                # print(center_coordinates)
                # draw_cricle(frame, center_coordinates)
                out.write(frame)
                resized_image = cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4))) 
                cv2.imshow('frame',resized_image)
                print('frame.shape : ', frame.shape)

                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            else:
                break
        count_frame+=1
        print('count_frame : ', count_frame)

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__' :
    # video()
    video()
    # img = cv2.imread("/home/minhssd/Pictures/HSV_test.png", cv2.IMREAD_COLOR)
    # cv2.imshow("image", img)
    # cv2.waitKey(0)

 
    
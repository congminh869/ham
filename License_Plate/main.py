import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.abspath(os.path.join(__dir__, '../LP_detect_module')))
print('=================================')
print(os.path.join(__dir__, '../LP_detect_module'))
print(os.path.abspath(os.path.join(__dir__, '../LP_detect_module')))
print(sys.path.append(os.path.abspath(os.path.join(__dir__, '../LP_detect_module'))))
# sys.path.append(os.path.abspath(os.path.join(__dir__, '../LP_detect_module')))
# sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../LP_detect_module')))

sys.path.append('/home/evnadmin/Documents/AI_hoabinh/yoloHubLoad/')
from LP_detect_module import FaceDetect

# from include.onebatch import main, License_Plate
# import cv2
# import time
# import sys
# #debug log
# from inspect import currentframe, getframeinfo
# import datetime

# debug_log = False

# def Debug_log(cf, filename, name = None):
#     if debug_log:
#         ct = datetime.datetime.now()
#         print(f'[{ct}] file {filename} , line : {cf.f_lineno} {name}')

# class DataImage:
#     cid = -1
#     type_id = -1
#     count = -1
#     image = None
    
#     def __init__(self, _cid, _type_id, _count, _image):
#         self.cid = _cid
#         self.type_id = _type_id
#         self.count = _count
#         self.image = _image 

# def test():
#     #[0,0, frame.shape[1],frame.shape[0]]
#     while True:
#         filename_video = "/home/mq/Documents/AI_hoabinh/video_test/plate/IMG_0706.MOV"
#         input_video = cv2.VideoCapture(filename_video)
#         if input_video.isOpened() == False:
#             print("Video not found")
#             sys.exit(1)
#         else:
#             # Read until the video is completed
#             dataImages = []
#             cid = 0
#             coordinate_rois = {}

#             count_plate_per_id = {}
#             buffer_plate_per_id = {}
#             lp_rets = []
#             count = 0
#             while(input_video.isOpened()):
#                 # Capture frame by frame
#                 ret, frame = input_video.read()
#                 if ret == True:
#                     print('count : ', count)
#                     count+=1
#                     if count > 100:
#                         dataImages.append(DataImage(cid,1,1,frame))
#                         coordinate_rois[cid] = [0,0,640, 640]#[0,0,frame.shape[1], frame.shape[0]]#[1300,834, 1960,1474]#[1230,834, 3369,1935]
#                         count_plate_per_id , buffer_plate_per_id , lp_rets, _, _=main(dataImages, coordinate_rois, count_plate_per_id , buffer_plate_per_id , lp_rets)
#                         dataImages = []
#                         for lp_ret in lp_rets:
#                             txt = lp_ret['txt']
#                             x1,y1,x2,y2 = lp_ret['box_kps']
#                             image = lp_ret['frame']
#                             Debug_log(currentframe(), getframeinfo(currentframe()).filename)
#                             # print(f'id: {lp_ret["id"]}, txt : {txt}, box_kps: [{x1}, {y1}, {x2}, {y2}] ')
#                             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                             cv2.putText(image, txt, (x1, y1-10),
#                                     cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
#                             name = './out_img/'+str(txt)+'_'+str(lp_ret['time'])+'.jpg'
#                             #cv2.imwrite(name, image)
#                         print('*****************************************************************')
#                 else:
#                     break

#         input_video.release()
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     license_Plate = License_Plate()

#     while True:
#         filename_video = "/home/mq/Documents/AI_hoabinh/video_test/plate/IMG_0706.MOV"
#         input_video = cv2.VideoCapture(filename_video)
#         if input_video.isOpened() == False:
#             print("Video not found")
#             sys.exit(1)
#         else:
#             # Read until the video is completed
#             dataImages = []
#             cid = 0
#             coordinate_rois = {}

#             count_plate_per_id = {}
#             buffer_plate_per_id = {}
#             lp_rets = []
#             count = 0
#             while(input_video.isOpened()):
#                 # Capture frame by frame
#                 ret, frame = input_video.read()
#                 if ret == True:
#                     print('count : ', count)
#                     count+=1
#                     if count > 100:
#                         dataImages.append(DataImage(cid,1,1,frame))
#                         coordinate_rois[cid] = [0,0,640, 640]#[0,0,frame.shape[1], frame.shape[0]]#[1300,834, 1960,1474]#[1230,834, 3369,1935]
#                         license_Plate.detect(dataImages, coordinate_rois)
#                         cv2.imshow('src', cv2.resize(frame, (int(frame.shape[1]/3),int(frame.shape[0]/3)), interpolation = cv2.INTER_AREA))
#                         dataImages = []
#                 else:
#                     break
#     
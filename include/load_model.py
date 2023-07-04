import cv2
import sys 
import yaml
import numpy as np

#region Phat hien vat the cao
from include.personHoldThingDetect import personHoldThing_All
from include.yolo import YOLOv562
from include.model_paddle_class import paddleClas, infer_paddleClass

from datetime import datetime

now = datetime.now() # current date and time

with open('./config/config_camera.yaml', 'r') as file:
    config_mode = yaml.safe_load(file)
mode = config_mode['mode']['number']
filename_video = config_mode['filename_video']['path']#"./image_test/person/person1.mp4"

with open('./config/config_main_include.yaml', 'r') as file:
    configuration = yaml.safe_load(file)
# if mode==0 or mode==1 or mode==2 or mode==6:

frame_width = int(450*3)
frame_height = int(350*2)
   
size = (frame_width, frame_height)
date_time = now.strftime("%m_%d_%Y__%H_%M_%S")
video_write = cv2.VideoWriter('./video_record/'+date_time+'.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)



if mode==0:
    ####################load model######################
    num_camera = 3#3
    #load model detect person
    print('LOAD MODEL PERSON')
    weights_person = configuration['weights_person']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_person']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_person']['iou_thres']
    conf_thres = configuration['weights_person']['conf_thres']
    img_size = configuration['weights_person']['img_size']
    max_det=configuration['weights_person']['max_det']
    agnostic_nms=configuration['weights_person']['agnostic_nms']
    yolo_person = YOLOv562(weights_person, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)

    #load model detect hat
    print('LOAD MODEL HAT')
    weights_hat = configuration['weights_hat']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_hat']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_hat']['iou_thres']
    conf_thres = configuration['weights_hat']['conf_thres']
    img_size = configuration['weights_hat']['img_size']
    max_det=configuration['weights_hat']['max_det']
    agnostic_nms=configuration['weights_hat']['agnostic_nms']
    yolo_hat = YOLOv562(weights_hat, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)
    
    #load model detect vehicle
    print('LOAD MODEL VEHICLE')
    weights_vehicle = configuration['weights_vehicle']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_vehicle']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_vehicle']['iou_thres']
    conf_thres = configuration['weights_vehicle']['conf_thres']
    img_size = configuration['weights_vehicle']['img_size']
    max_det=configuration['weights_vehicle']['max_det']
    agnostic_nms=configuration['weights_vehicle']['agnostic_nms']
    yolo_vehicle = YOLOv562(weights_vehicle, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)

    #region khoi tao phat hien vat cao
    personHoldThingDetect_video_1 = personHoldThing_All()
    # main_TBA(dataImages, coordinate_rois)
    
if mode==1:
    ##############################load model##############################
    num_camera = 1#18
    #load model detect person
    print('LOAD MODEL PERSON')
    weights_person = configuration['weights_person']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_person']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_person']['iou_thres']
    conf_thres = configuration['weights_person']['conf_thres']
    img_size = configuration['weights_person']['img_size']
    max_det=configuration['weights_person']['max_det']
    agnostic_nms=configuration['weights_person']['agnostic_nms']
    yolo_person = YOLOv562(weights_person, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)

    #load model detect hat
    print('LOAD MODEL HAT')
    weights_hat = configuration['weights_hat']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_hat']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_hat']['iou_thres']
    conf_thres = configuration['weights_hat']['conf_thres']
    img_size = configuration['weights_hat']['img_size']
    max_det=configuration['weights_hat']['max_det']
    agnostic_nms=configuration['weights_hat']['agnostic_nms']
    yolo_hat = YOLOv562(weights_hat, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)

    #load model detect uniform
    engine = paddleClas()
    ######################################################################
    # main_tunnel(dataImages, coordinate_rois)
if mode==2:
    num_camera = 1#20
    #load model detect person
    print('LOAD MODEL PERSON')
    weights_person = configuration['weights_person']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_person']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_person']['iou_thres']
    conf_thres = configuration['weights_person']['conf_thres']
    img_size = configuration['weights_person']['img_size']
    max_det=configuration['weights_person']['max_det']
    agnostic_nms=configuration['weights_person']['agnostic_nms']
    yolo_person = YOLOv562(weights_person, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)

    ##########################load model
    print('LOAD MODEL FIRE AND SMOKE')
    #load model detect fire and smoke
    weights_FS = configuration['weights_FS']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_FS']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_FS']['iou_thres']
    conf_thres = configuration['weights_FS']['conf_thres']
    img_size = configuration['weights_FS']['img_size']
    max_det=configuration['weights_FS']['max_det']
    agnostic_nms=configuration['weights_FS']['agnostic_nms']
    yolo_FS = YOLOv562(weights_FS, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)
    print('load model done')
    ####################################
    # main_fence(dataImages)
if mode==3:
    num_camera = 1
    #load model detect High volt switch
    print('LOAD MODEL HIGH VOLT SWITCH')
    weights_HVS = configuration['weights_HVS']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_HVS']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_HVS']['iou_thres']
    conf_thres = configuration['weights_HVS']['conf_thres']
    img_size = configuration['weights_HVS']['img_size']
    max_det=configuration['weights_HVS']['max_det']
    agnostic_nms=configuration['weights_HVS']['agnostic_nms']
    yolo_HVS = YOLOv562(weights_HVS, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)
    # main_high_volt_switch(dataImages, coordinate_rois)
if mode==4:
    num_camera = 4
    #load model detect person
    print('LOAD MODEL PERSON')
    weights_person = configuration['weights_person']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_person']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_person']['iou_thres']
    conf_thres = configuration['weights_person']['conf_thres']
    img_size = configuration['weights_person']['img_size']
    max_det=configuration['weights_person']['max_det']
    agnostic_nms=configuration['weights_person']['agnostic_nms']
    yolo_person = YOLOv562(weights_person, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)

    #load model detect belt
    print('LOAD MODEL BELT')
    weights_belt = configuration['weights_belt']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_belt']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_belt']['iou_thres']
    conf_thres = configuration['weights_belt']['conf_thres']
    img_size = configuration['weights_belt']['img_size']
    max_det=configuration['weights_belt']['max_det']
    agnostic_nms=configuration['weights_belt']['agnostic_nms']
    yolo_belt = YOLOv562(weights_belt, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)
    # main_belt(dataImages, coordinate_rois)
if mode==5:
    num_camera = 1
    #load model detect clock
    print('LOAD MODEL CLOCK')
    weights_clock = configuration['weights_clock']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_clock']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_clock']['iou_thres']
    conf_thres = configuration['weights_clock']['conf_thres']
    img_size = configuration['weights_clock']['img_size']
    max_det=configuration['weights_clock']['max_det']
    agnostic_nms=configuration['weights_clock']['agnostic_nms']
    yolo_clock = YOLOv562(weights_clock, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)
    # main_clock(dataImages, coordinate_rois)
if mode==6:
    num_camera = 1
    #load model detect person
    print('LOAD MODEL PERSON')
    weights_person = configuration['weights_person']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_person']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_person']['iou_thres']
    conf_thres = configuration['weights_person']['conf_thres']
    img_size = configuration['weights_person']['img_size']
    max_det=configuration['weights_person']['max_det']
    agnostic_nms=configuration['weights_person']['agnostic_nms']
    yolo_person = YOLOv562(weights_person, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)

    print('LOAD MODEL FIRE AND SMOKE')
    #load model detect fire and smoke
    weights_FS = configuration['weights_FS']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_FS']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_FS']['iou_thres']
    conf_thres = configuration['weights_FS']['conf_thres']
    img_size = configuration['weights_FS']['img_size']
    max_det=configuration['weights_FS']['max_det']
    agnostic_nms=configuration['weights_FS']['agnostic_nms']
    yolo_FS = YOLOv562(weights_FS, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)
    print('load model done')

    # weights_person = './weight/crowdhuman_yolov5m.pt'
    # yolo_person = YOLOv562(weights_person)

    #load model detect hat
    print('LOAD MODEL HAT')
    weights_hat = configuration['weights_hat']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_hat']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_hat']['iou_thres']
    conf_thres = configuration['weights_hat']['conf_thres']
    img_size = configuration['weights_hat']['img_size']
    max_det=configuration['weights_hat']['max_det']
    agnostic_nms=configuration['weights_hat']['agnostic_nms']
    yolo_hat = YOLOv562(weights_hat, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)


    #load model detect vehicle
    print('LOAD MODEL VEHICLE')
    weights_vehicle = configuration['weights_vehicle']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_vehicle']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_vehicle']['iou_thres']
    conf_thres = configuration['weights_vehicle']['conf_thres']
    img_size = configuration['weights_vehicle']['img_size']
    max_det=configuration['weights_vehicle']['max_det']
    agnostic_nms=configuration['weights_vehicle']['agnostic_nms']
    yolo_vehicle = YOLOv562(weights_vehicle, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)

    #load model detect High volt switch
    print('LOAD MODEL HIGH VOLT SWITCH')
    weights_HVS = configuration['weights_HVS']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_HVS']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_HVS']['iou_thres']
    conf_thres = configuration['weights_HVS']['conf_thres']
    img_size = configuration['weights_HVS']['img_size']
    max_det=configuration['weights_HVS']['max_det']
    agnostic_nms=configuration['weights_HVS']['agnostic_nms']
    yolo_HVS = YOLOv562(weights_HVS, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)

    #load model detect belt
    print('LOAD MODEL BELT')
    weights_belt = configuration['weights_belt']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_belt']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_belt']['iou_thres']
    conf_thres = configuration['weights_belt']['conf_thres']
    img_size = configuration['weights_belt']['img_size']
    max_det=configuration['weights_belt']['max_det']
    agnostic_nms=configuration['weights_belt']['agnostic_nms']
    yolo_belt = YOLOv562(weights_belt, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)

    #load model detect clock
    print('LOAD MODEL CLOCK')
    weights_clock = configuration['weights_clock']['weights'] #'./weight/crowdhuman_yolov5m.pt'
    classes=configuration['weights_clock']['classes']
    device=configuration['device']['device']
    iou_thres = configuration['weights_clock']['iou_thres']
    conf_thres = configuration['weights_clock']['conf_thres']
    img_size = configuration['weights_clock']['img_size']
    max_det=configuration['weights_clock']['max_det']
    agnostic_nms=configuration['weights_clock']['agnostic_nms']
    yolo_clock = YOLOv562(weights_clock, classes=classes, device=device, iou_thres = iou_thres, conf_thres = conf_thres, img_size = img_size, max_det=max_det, agnostic_nms=agnostic_nms)

    #region khoi tao phat hien vat cao
    personHoldThingDetect_video_1 = personHoldThing_All()

    #load model detect uniform
    engine = paddleClas()
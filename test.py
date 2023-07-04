from inspect import currentframe, getframeinfo
import datetime
debug_log = True

def Debug_log(cf, filename, name = None):
    if debug_log:
        ct = datetime.datetime.now()
        print(f'================ [{ct}] file {filename} , line : {cf.f_lineno} {name}')
Debug_log(currentframe(), getframeinfo(currentframe()).filename)

from AI_Interface import AI_TBA,AI_TUNNEL, AI_LICENSE_PLATE, \
                        AI_HSV, AI_CLOCK, AI_FENCE, AI_BELT, \
                        AI_PersonHoldThingDetect, AI_TESTSORT,\
                        AI_LP_LIVE, AI_COUNTPERSON
Debug_log(currentframe(), getframeinfo(currentframe()).filename)
import cv2
import time
import sys
Debug_log(currentframe(), getframeinfo(currentframe()).filename)
import numpy as np
Debug_log(currentframe(), getframeinfo(currentframe()).filename)

from License_Plate.include.models_retina import model_retina
Debug_log(currentframe(), getframeinfo(currentframe()).filename)
class DataImage:
    cid = -1
    type_id = -1
    count = -1
    image = None
    
    def __init__(self, _cid, _type_id, _count, _image):
        self.cid = _cid
        self.type_id = _type_id
        self.count = _count
        self.image = _image

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

def test_hsv():
    main = AI_CLOCK()
    main.Init()

    num_camera = [0,1,2]

    filename_video = "/home/mq/Documents/Son/yoloHubLoad/image_test/person/belt3.mp4"

    input_video = cv2.VideoCapture(filename_video)
    if input_video.isOpened() == False:
        print("Video not found")
        sys.exit(1)
    else:
        # Read until the video is completed
        dataImages = []
        cid = 0
        coordinate_rois = {}
        labels_allow_helmet = {}
        count_frame = 0
        while(input_video.isOpened()):
            # Capture frame by frame
            frame = cv2.imread("clock_I.jpg", cv2.IMREAD_COLOR)

            if True:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                print('num_camera : ', num_camera)
                for cid in num_camera:
                    dataImages.append(DataImage(cid,1,1,frame))
                    coordinate_rois[cid] = {0: [96,114,201,196], 1: [319,238,426,312], 2: [554,363,647,433]}

                dict_data = main.Detect(dataImages)
                dataImages = []

                count_frame +=1
                print('count_frame : ', count_frame)
            else:
                print('break')
                break

def video(filename_video):
    tunnel = AI_TUNNEL()
    tunnel.Init()

    tba = AI_TBA()
    tba.Init()

    hsv = AI_HSV()
    hsv.Init()

    belt = AI_BELT()
    belt.Init()

    clock = AI_CLOCK()
    clock.Init()

    fence = AI_FENCE()
    fence.Init()

    plate = AI_LICENSE_PLATE()
    plate.Init()


    num_camera = [0]

    # filename_video = '/home/evnadmin/Documents/AI_hoabinh/video/videotest.mp4'

    input_video = cv2.VideoCapture(filename_video)
    if input_video.isOpened() == False:
        print("Video not found")
        sys.exit(1)
    else:
        # Read until the video is completed
        dataImages = []
        cid = 0
        coordinate_rois = {}
        labels_allow_helmet = {}
        labels_allow_uniform = {}

        coordinate_rois_HSV = {}


        coordinate_rois_reg={}
        count_frame = 0
        while(input_video.isOpened()):
            # Capture frame by frame
            ret, frame = input_video.read()

            if ret == True:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # print('num_camera : ', num_camera)
                for cid in num_camera:
                    dataImages.append(DataImage(cid,1,1,frame))
                    coordinate_rois[cid] = [[205*4, 203*4],[169*4,67*4],[296*4,15*4],[550*4,29*4], [708*4,86*4],[906*4,262*4], [552*4,408*4], [225*4,324*4]]
                    labels_allow_helmet[cid] = [1,2,3,4,5,6]# [[0]]
                    labels_allow_uniform[cid] = [[6],[8]]# [quan=[4,5,6], ao=[7,8]]

                    coordinate_rois_HSV[cid] = {0: [96,114,201,196], 1: [319,238,426,312], 2: [554,363,647,433]}

                    coordinate_rois_reg[cid] = [0,0,frame.shape[1], frame.shape[1]]

                    # if count_frame>=5:
                    #     num_camera = [0,2]

                dict_data = tba.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
                # show(dict_data)
                dict_data = tunnel.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
                
                #
                dict_data = hsv.Detect(dataImages, coordinate_rois_HSV, labels_allow_helmet, labels_allow_uniform)

                #reg
                dict_data = belt.Detect(dataImages, coordinate_rois_reg, labels_allow_helmet, labels_allow_uniform)
                dict_data = plate.Detect(dataImages, coordinate_rois_reg, labels_allow_helmet, labels_allow_uniform)

                #full image
                dict_data = clock.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
                dict_data = fence.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)

                dataImages = []

                count_frame +=1
                # print('count_frame : ', count_frame)
            else:
                print('break')
                break 

def tba_(filename_video, video = False):
    tba = AI_TBA()
    tba.Init()
    print('check tba')

    num_camera = [0]#,2,3,4,5,6,7,8,9,10,11,12,13]

    # filename_video = '/home/evnadmin/Documents/AI_hoabinh/video/person.jpg'

    if video:
        cap = cv2.VideoCapture(filename_video)
    else:
        frame = cv2.imread(filename_video)
    # Read until the video is completed
    dataImages = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 0
    while True:
        if video:
            ret, frame = cap.read()
            if ret!=True:
                continue
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print('num_camera : ', num_camera)
        if count_frame%15==0:
            for cid in num_camera:
                dataImages.append(DataImage(cid,1,1,frame))
                coordinate_rois[cid] = [[182,498],[202,185],[466,95],[892,142], [1136,280],[1134,489], [924,658], [318,662]]
                # coordinate_rois[cid] = [[205*4, 203*4],[169*4,67*4],[296*4,15*4],[550*4,29*4], [708*4,86*4],[906*4,262*4], [552*4,408*4], [225*4,324*4]]
                #0,4,6->0, 3->1, 1,2,5->2
                labels_allow_helmet[cid] = [1]# [[0]] , {0: 'None', 1: 'Red', 2: 'Yellow', 3: 'White', 4: 'Blue', 5: 'Orange', 6: 'Others'}
                labels_allow_uniform[cid] = [[6],[8]]# [quan=[4,5,6], ao=[7,8]]

                coordinate_rois_HSV[cid] = {0: [96,114,201,196], 1: [319,238,426,312], 2: [554,363,647,433]}

                coordinate_rois_reg[cid] = [0,0,frame.shape[1], frame.shape[1]]

                # if count_frame>=5:
                #     num_camera = [0,2]

            dict_data = tba.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
            # show(dict_data)

            dataImages = []

        count_frame +=1

def TbaLive(filename_video, video = False):
    tba = AI_TBA()
    tba.Init(stream=True)
    print('check tba')

    num_camera = [0]#,2,3,4,5,6,7,8,9,10,11,12,13]

    # filename_video = '/home/evnadmin/Documents/AI_hoabinh/video/person.jpg'

    if video:
        cap = cv2.VideoCapture(filename_video)
    else:
        frame = cv2.imread(filename_video)
    # Read until the video is completed
    dataImages = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 0
    while True:
        if video:
            ret, frame = cap.read()
            if ret!=True:
                continue
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print('num_camera : ', num_camera)
        if count_frame%15==0:
            for cid in num_camera:
                dataImages.append(DataImage(cid,1,1,frame))
                coordinate_rois[cid] = [[182,498],[202,185],[466,95],[892,142], [1136,280],[1134,489], [924,658], [318,662]]
                # coordinate_rois[cid] = [[205*4, 203*4],[169*4,67*4],[296*4,15*4],[550*4,29*4], [708*4,86*4],[906*4,262*4], [552*4,408*4], [225*4,324*4]]
                #0,4,6->0, 3->1, 1,2,5->2
                labels_allow_helmet[cid] = [1]# [[0]] , {0: 'None', 1: 'Red', 2: 'Yellow', 3: 'White', 4: 'Blue', 5: 'Orange', 6: 'Others'}
                labels_allow_uniform[cid] = [[6],[8]]# [quan=[4,5,6], ao=[7,8]]

                coordinate_rois_HSV[cid] = {0: [96,114,201,196], 1: [319,238,426,312], 2: [554,363,647,433]}

                coordinate_rois_reg[cid] = [0,0,frame.shape[1], frame.shape[1]]

                # if count_frame>=5:
                #     num_camera = [0,2]

            dict_data = tba.detectLiveStr(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
            # show(dict_data)

            dataImages = []

        count_frame +=1

def tunnel_(filename_video, video = False):
    tba = AI_TUNNEL()
    tba.Init()


    num_camera = [0]#,1,2,3]#,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

    # filename_video = '/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/test_sort/image_test/person/1.png'
    if video:
        cap = cv2.VideoCapture(filename_video)
    else:
        frame = cv2.imread(filename_video)
    # Read until the video is completed
    dataImages = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 0
    while True:
        if video:
            ret, frame = cap.read()
            if ret!=True:
                continue
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print('num_camera : ', num_camera)
        if count_frame%15==0:
            for cid in num_camera:
                dataImages.append(DataImage(cid,1,1,frame))
                coordinate_rois[cid] = [[0,0], [frame.shape[1], 0], [frame.shape[1], frame.shape[0]], [0, frame.shape[0]]] #[[316,482],[202,185],[466,95],[892,142], [1136,280],[1134,489], [924,658], [318,662]]
                labels_allow_helmet[cid] = [1,2,3,4,5,6]# [[0]]
                labels_allow_uniform[cid] = [[4],[7]]# [quan=[4,5,6], ao=[7,8]]

                coordinate_rois_HSV[cid] = {0: [96,114,201,196], 1: [319,238,426,312], 2: [554,363,647,433]}

                coordinate_rois_reg[cid] = [0,0,frame.shape[1], frame.shape[1]]

                # if count_frame>=5:
                #     num_camera = [0,2]
            t1 = time.time()
            dict_data = tba.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
            # show(dict_data)
            t2 = time.time()
            print('test total time : ', t2 - t1)
            print('count_frame : ', count_frame)
            dataImages = []

        count_frame +=1


def hsv_(filename_video='/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/data/data_dongho_moi/data/images/I_105_1.jpg'):
    tba = AI_HSV()
    tba.Init()


    num_camera = [0]

    # filename_video = '/home/evnadmin/Documents/AI_hoabinh/video/person.jpg'

    frame = cv2.imread(filename_video)
    # Read until the video is completed
    dataImages = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 0
    while True:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print('num_camera : ', num_camera)
        for cid in num_camera:
            dataImages.append(DataImage(cid,1,1,frame))
            coordinate_rois[cid] = [[205*4, 203*4],[169*4,67*4],[296*4,15*4],[550*4,29*4], [708*4,86*4],[906*4,262*4], [552*4,408*4], [225*4,324*4]]
            labels_allow_helmet[cid] = [1,2,3,4,5,6]# [[0]]
            labels_allow_uniform[cid] = [[6],[8]]# [quan=[4,5,6], ao=[7,8]]

            coordinate_rois_HSV[cid] = {0: [96,114,201,196], 1: [319,238,426,312], 2: [554,363,647,433]}

            coordinate_rois_reg[cid] = [0,0,frame.shape[1], frame.shape[1]]

            # if count_frame>=5:
            #     num_camera = [0,2]

        dict_data = tba.Detect(dataImages, coordinate_rois_HSV, labels_allow_helmet, labels_allow_uniform)
        # show(dict_data)

        dataImages = []

        count_frame +=1


def belt_(filename_video = '/media/minhssd/Disk2T/Documents/thuy_dien_hoa_binh/code/video_test/belt3.mp4'):
    tba = AI_BELT()
    tba.Init()


    num_camera = [0]

    # filename_video = '/home/evnadmin/Documents/AI_hoabinh/video/person.jpg'

    # frame = cv2.imread(filename_video)
    cap = cv2.VideoCapture(filename_video)
    # Read until the video is completed
    dataImages = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 0
    while True:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print('num_camera : ', num_camera)
        ret, frame = cap.read()
        h, w, c =  frame.shape
        if ret == True:
            for cid in num_camera:
                dataImages.append(DataImage(cid,1,1,frame))
                coordinate_rois[cid] = [[100, 100],[w,100],[w,h],[100,h]]#, [708*4,86*4],[906*4,262*4], [552*4,408*4], [225*4,324*4]]

                # if count_frame>=5:
                #     num_camera = [0,2]

            dict_data = tba.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
            # show(dict_data)

            dataImages = []

        count_frame +=1


def clock_(filename_video):
    tba = AI_CLOCK()
    tba.Init()


    num_camera = [0]

    # filename_video = '/home/evnadmin/Documents/AI_hoabinh/video/person.jpg'

    frame = cv2.imread(filename_video)
    # Read until the video is completed
    dataImages = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 0
    while True:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print('num_camera : ', num_camera)
        for cid in num_camera:
            dataImages.append(DataImage(cid,1,1,frame))
            coordinate_rois[cid] = [[205*4, 203*4],[169*4,67*4],[296*4,15*4],[550*4,29*4], [708*4,86*4],[906*4,262*4], [552*4,408*4], [225*4,324*4]]
            labels_allow_helmet[cid] = [1,2,3,4,5,6]# [[0]]
            labels_allow_uniform[cid] = [[6],[8]]# [quan=[4,5,6], ao=[7,8]]

            coordinate_rois_HSV[cid] = {0: [96,114,201,196], 1: [319,238,426,312], 2: [554,363,647,433]}

            coordinate_rois_reg[cid] = [0,0,frame.shape[1], frame.shape[1]]

            # if count_frame>=5:
            #     num_camera = [0,2]

        dict_data = tba.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
        # show(dict_data)

        dataImages = []

        count_frame +=1



def fence_(filename_video):
    tba = AI_FENCE()
    tba.Init()


    num_camera = [0]#,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    # filename_video = '/home/evnadmin/Documents/AI_hoabinh/video/person.jpg'

    frame = cv2.imread(filename_video)
    # Read until the video is completed
    dataImages = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 0
    while True:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print('num_camera : ', num_camera)
        for cid in num_camera:
            dataImages.append(DataImage(cid,1,1,frame))
            coordinate_rois[cid] = [[193, 144],[332, 53],[471,236],[476,476], [478, 584],[300, 623], [138,499]]
            labels_allow_helmet[cid] = [1,2,3,4,5,6]# [[0]]
            labels_allow_uniform[cid] = [[6],[8]]# [quan=[4,5,6], ao=[7,8]]

            coordinate_rois_HSV[cid] = {0: [96,114,201,196], 1: [319,238,426,312], 2: [554,363,647,433]}

            coordinate_rois_reg[cid] = [0,0,frame.shape[1], frame.shape[1]]

            # if count_frame>=5:
            #     num_camera = [0,2]

        dict_data = tba.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
        # show(dict_data)

        dataImages = []

        count_frame +=1


def fence_event_test():
    tba = AI_FENCE()
    tba.Init()


    num_camera = [0]

    # filename_video = '/home/evnadmin/Documents/AI_hoabinh/video/person.jpg'

    frame_1 = cv2.imread('/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/image/1person.jpg')
    frame_2 = cv2.imread('/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/image/2person.jpg')
    frame_3 = cv2.imread('/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/image/3person.jpg')
    # Read until the video is completed
    dataImages = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 0
    while True:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print('num_camera : ', num_camera)
        if True:
            dataImages.append(DataImage(0,1,1,frame_1))
            dataImages.append(DataImage(0,1,1,frame_1))
            dataImages.append(DataImage(0,1,1,frame_1))
            dataImages.append(DataImage(0,1,1,frame_1))
            dataImages.append(DataImage(0,1,1,frame_1))
            dataImages.append(DataImage(0,1,1,frame_1))
            dataImages.append(DataImage(0,1,1,frame_1))
            dataImages.append(DataImage(0,1,1,frame_1))
            dataImages.append(DataImage(0,1,1,frame_1))
            dataImages.append(DataImage(0,1,1,frame_1))

            dataImages.append(DataImage(1,1,1,frame_3))
            dataImages.append(DataImage(1,1,1,frame_3))
            dataImages.append(DataImage(1,1,1,frame_3))
            dataImages.append(DataImage(1,1,1,frame_3))
            dataImages.append(DataImage(1,1,1,frame_3))
            dataImages.append(DataImage(1,1,1,frame_3))
            dataImages.append(DataImage(1,1,1,frame_3))
            dataImages.append(DataImage(1,1,1,frame_3))

            dataImages.append(DataImage(3,1,1,frame_2))
            dataImages.append(DataImage(3,1,1,frame_2))
            dataImages.append(DataImage(3,1,1,frame_2))
            dataImages.append(DataImage(3,1,1,frame_2))
            dataImages.append(DataImage(3,1,1,frame_2))
            dataImages.append(DataImage(3,1,1,frame_2))
            dataImages.append(DataImage(3,1,1,frame_2))
            
            coordinate_rois[0] = [[10, 10],[600,10],[600,600],[10,600]]#, [708*4,86*4],[906*4,262*4], [552*4,408*4], [225*4,324*4]]
            coordinate_rois[1] = [[10, 10],[600,10],[600,600],[10,600]]#, [708*4,86*4],[906*4,262*4], [552*4,408*4], [225*4,324*4]]
            coordinate_rois[3] = [[10, 10],[600,10],[600,600],[10,600]]#, [708*4,86*4],[906*4,262*4], [552*4,408*4], [225*4,324*4]]

            # if count_frame>=5:
            #     num_camera = [0,2]

        dict_data = tba.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
        # show(dict_data)

        dataImages = []

        count_frame +=1

def plate_(filename_video='/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/plate/oto_edit.avi'):
    import os
    import shutil
    # if os.path.isdir('event_plate'):
    #     shutil.rmtree('event_plate')
    #     os.mkdir('./event_plate/')

    #     os.mkdir('./event_plate/data_event/')
    #     os.mkdir('./event_plate/detect/')
    #     os.mkdir('./event_plate/event/')
    #     os.mkdir('./event_plate/main/')
    #     os.mkdir('./event_plate/reg/')
    #     os.mkdir('./event_plate/raw/')
    # else:
    #     os.mkdir('./event_plate/')
    #     os.mkdir('./event_plate/data_event/')
    #     os.mkdir('./event_plate/detect/')
    #     os.mkdir('./event_plate/event/')
    #     os.mkdir('./event_plate/main/')
    #     os.mkdir('./event_plate/reg/')
    #     os.mkdir('./event_plate/rawS/')


    tba = AI_LICENSE_PLATE()
    tba.Init()


    num_camera = [0]

    # filename_video = '/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/image/bienso.jpg'

    # frame = cv2.imread(filename_video)
    img = cv2.imread('/home/minhssd/Pictures/plate_3.jpg')
    cap = cv2.VideoCapture(filename_video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    # Read until the video is completed
    dataImages = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 0
    while True:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print('num_camera : ', num_camera)

        ret, frame = cap.read()
        # if count_frame==0:
        # cv2.imwrite('img_plate.jpg', frame)
        if ret :#and count_frame>=3250:#>=48:#%1==0:
            for cid in num_camera:
                dataImages.append(DataImage(cid,1,1,frame))
                coordinate_rois[cid] = [[205*4, 203*4],[169*4,67*4],[296*4,15*4],[550*4,29*4], [708*4,86*4],[906*4,262*4], [552*4,408*4], [225*4,324*4]]
                labels_allow_helmet[cid] = [1,2,3,4,5,6]# [[0]]
                labels_allow_uniform[cid] = [[6],[8]]# [quan=[4,5,6], ao=[7,8]]

                coordinate_rois_HSV[cid] = {0: [96,114,201,196], 1: [319,238,426,312], 2: [554,363,647,433]}

                coordinate_rois_reg[cid] = [0,0,frame.shape[1], frame.shape[1]]

                # if count_frame>=5:
                #     num_camera = [0,2]

            dict_data = tba.Detect(dataImages, coordinate_rois_reg, labels_allow_helmet, labels_allow_uniform)
            # show(dict_data)

            dataImages = []
        if ret==False:#if False:#count_frame==0:
            for cid in num_camera:
                dataImages.append(DataImage(cid,1,1,img))
                coordinate_rois[cid] = [[205*4, 203*4],[169*4,67*4],[296*4,15*4],[550*4,29*4], [708*4,86*4],[906*4,262*4], [552*4,408*4], [225*4,324*4]]
                labels_allow_helmet[cid] = [1,2,3,4,5,6]# [[0]]
                labels_allow_uniform[cid] = [[6],[8]]# [quan=[4,5,6], ao=[7,8]]

                coordinate_rois_HSV[cid] = {0: [96,114,201,196], 1: [319,238,426,312], 2: [554,363,647,433]}

                coordinate_rois_reg[cid] = [0,0,img.shape[1], img.shape[1]]

                # if count_frame>=5:
                #     num_camera = [0,2]

            dict_data = tba.Detect(dataImages, coordinate_rois_reg, labels_allow_helmet, labels_allow_uniform)
            # show(dict_data)

            dataImages = []
        count_frame +=1
        print('count_frame : ', count_frame)

def PlateLive(filename_video='/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/plate/oto_edit.avi'):
    tba = AI_LP_LIVE()
    tba.Init()


    num_camera = [0]

    # filename_video = '/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/image/bienso.jpg'

    # frame = cv2.imread(filename_video)
    img = cv2.imread('/home/minhssd/Pictures/plate_3.jpg')
    cap = cv2.VideoCapture(filename_video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    # Read until the video is completed
    dataImages = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 0
    while True:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print('num_camera : ', num_camera)

        ret, frame = cap.read()
        # if count_frame==0:
        # cv2.imwrite('img_plate.jpg', frame)
        if ret and count_frame>=35:#>=48:#%1==0:
            for cid in num_camera:
                dataImages.append(DataImage(cid,1,1,frame))
                coordinate_rois[cid] = [[205*4, 203*4],[169*4,67*4],[296*4,15*4],[550*4,29*4], [708*4,86*4],[906*4,262*4], [552*4,408*4], [225*4,324*4]]
                labels_allow_helmet[cid] = [1,2,3,4,5,6]# [[0]]
                labels_allow_uniform[cid] = [[6],[8]]# [quan=[4,5,6], ao=[7,8]]

                coordinate_rois_HSV[cid] = {0: [96,114,201,196], 1: [319,238,426,312], 2: [554,363,647,433]}

                coordinate_rois_reg[cid] = [0,0,frame.shape[1], frame.shape[1]]

                # if count_frame>=5:
                #     num_camera = [0,2]

            dict_data = tba.Detect(dataImages, coordinate_rois_reg, labels_allow_helmet, labels_allow_uniform)
            # show(dict_data)

            dataImages = []
        if ret==False:#if False:#count_frame==0:
            for cid in num_camera:
                dataImages.append(DataImage(cid,1,1,img))
                coordinate_rois[cid] = [[205*4, 203*4],[169*4,67*4],[296*4,15*4],[550*4,29*4], [708*4,86*4],[906*4,262*4], [552*4,408*4], [225*4,324*4]]
                labels_allow_helmet[cid] = [1,2,3,4,5,6]# [[0]]
                labels_allow_uniform[cid] = [[6],[8]]# [quan=[4,5,6], ao=[7,8]]

                coordinate_rois_HSV[cid] = {0: [96,114,201,196], 1: [319,238,426,312], 2: [554,363,647,433]}

                coordinate_rois_reg[cid] = [0,0,img.shape[1], img.shape[1]]

                # if count_frame>=5:
                #     num_camera = [0,2]

            dict_data = tba.Detect(dataImages, coordinate_rois_reg, labels_allow_helmet, labels_allow_uniform)
            # show(dict_data)

            dataImages = []
        count_frame +=1
        print('count_frame : ', count_frame)

def plate_retina(filename_video='/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/plate/oto_edit.avi'):
    tba = AI_LICENSE_PLATE()
    tba.Init()


    num_camera = [0]

    # filename_video = '/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/image/bienso.jpg'

    # frame = cv2.imread(filename_video)
    cap = cv2.VideoCapture(filename_video)
    # Read until the video is completed
    dataImages = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 0
    while True:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print('num_camera : ', num_camera)
        ret, frame = cap.read()
        if ret:
            for cid in num_camera:
                dataImages.append(DataImage(cid,1,1,frame))
                coordinate_rois[cid] = [[205*4, 203*4],[169*4,67*4],[296*4,15*4],[550*4,29*4], [708*4,86*4],[906*4,262*4], [552*4,408*4], [225*4,324*4]]
                labels_allow_helmet[cid] = [1,2,3,4,5,6]# [[0]]
                labels_allow_uniform[cid] = [[6],[8]]# [quan=[4,5,6], ao=[7,8]]

                coordinate_rois_HSV[cid] = {0: [96,114,201,196], 1: [319,238,426,312], 2: [554,363,647,433]}

                coordinate_rois_reg[cid] = [0,0,frame.shape[1], frame.shape[1]]

                # if count_frame>=5:
                #     num_camera = [0,2]

            dict_data = tba.Detect(dataImages, coordinate_rois_reg, labels_allow_helmet, labels_allow_uniform)
            # show(dict_data)

            dataImages = []

        count_frame +=1

def personHoldThingDetect_(filename_video, video = False):
    tba = AI_PersonHoldThingDetect()
    tba.Init()
    print('check tba')

    num_camera = [0]

    # filename_video = '/home/evnadmin/Documents/AI_hoabinh/video/person.jpg'

    if video:
        cap = cv2.VideoCapture(filename_video)
    else:
        frame = cv2.imread(filename_video)
    # Read until the video is completed
    dataImages = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 0
    while True:
        if video:
            ret, frame = cap.read()
            if ret!=True:
                continue
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print('num_camera : ', num_camera)
        # print('count_frame = ',count_frame)
        dataImages.append(DataImage(0,1,count_frame,frame))
        
        coordinate_rois[0] = [[0,0],[0,2500],[2500,2500],[2500,0]]

        # if count_frame>=5:
        #     num_camera = [0,2]
        if len(dataImages)==5:
            dict_data = tba.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
            # show(dict_data)

            dataImages = []

        count_frame +=1

def test_loop_ip_():
    tba = AI_TBA()
    tba.Init()


    num_camera = [0,1,3]

    filename_video1 = '/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/ROI.mp4'
    filename_video2 = '/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/video_test_hoabinh_TBA4.avi'
    filename_video3 = '/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/not_hat.mp4'

    cap1 = cv2.VideoCapture(filename_video1)
    cap2 = cv2.VideoCapture(filename_video2)
    cap3 = cv2.VideoCapture(filename_video3)

    if (cap3.isOpened()== False): 
        print("Error opening video stream or file")
        # sys.exit()

    dataImages = []
    dataImages1 = []
    dataImages2 = []
    dataImages3 = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 1

    coordinate_rois[0] = [[316,482],[202,185],[466,95],[892,142], [1136,280],[1134,489], [924,658], [318,662]]
    coordinate_rois[1] = [[ 148, 1352],
                         [ 200,  140],
                         [1368,   40],
                         [2600,  124],
                         [2416, 1288],
                         [1360, 1472]]
    coordinate_rois[2] = [[316,482],[202,185],[466,95],[892,142], [1136,280],[1134,489], [924,658], [318,662]]
    labels_allow_helmet[0] = [1,2,3,4,5,6]# [[0]]
    labels_allow_helmet[1] = [1,2,3,4,5,6]# [[0]]
    labels_allow_helmet[2] = [1,2,3,4,5,6]# [[0]]
    count1 = 0
    count2 = 0
    count3 = 0
    loop = 30
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        # print(ret1, ret2, ret3)

        if ret1:
            if len(dataImages1)<=loop:
                dataImages1.append(DataImage(0,1,count1,frame1))
                count1+=1
                # print('check 1')
        if ret2:
            if len(dataImages1)<=loop:
                dataImages2.append(DataImage(1,1,count2,frame2))
                count2+=1
                # print('check 2')
        if ret3:
            if len(dataImages1)<=loop:
                dataImages3.append(DataImage(2,1,count3,frame3))
                count3+=1
                # print('check 3')
        # print('len(dataImages1) : ', len(dataImages1))
        # print('len(dataImages2) : ', len(dataImages2))
        # print('len(dataImages3) : ', len(dataImages3))
        if len(dataImages1)==loop or len(dataImages2)==loop or len(dataImages3)==loop:
            dataImages = dataImages1 + dataImages2 + dataImages3

            dict_data = tba.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
            # show(dict_data)
            dataImages = []
            dataImages1 = []
            dataImages2 = []
            dataImages3 = []
            count1 = 0
            count2 = 0
            count3 = 0



        count_frame +=1


def test_loop_ip_fence():
    tba = AI_FENCE()
    tba.Init()


    num_camera = [0,1,3]

    filename_video1 = '/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/ROI.mp4'
    filename_video2 = '/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/video_test_hoabinh_TBA4.avi'
    filename_video3 = '/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/not_hat.mp4'

    cap1 = cv2.VideoCapture(filename_video1)
    cap2 = cv2.VideoCapture(filename_video2)
    cap3 = cv2.VideoCapture(filename_video3)

    if (cap3.isOpened()== False): 
        print("Error opening video stream or file")
        # sys.exit()

    dataImages = []
    dataImages1 = []
    dataImages2 = []
    dataImages3 = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 1

    coordinate_rois[0] = [[316,482],[202,185],[466,95],[892,142], [1136,280],[1134,489], [924,658], [318,662]]
    coordinate_rois[1] = [[ 148, 1352],
                         [ 200,  140],
                         [1368,   40],
                         [2600,  124],
                         [2416, 1288],
                         [1360, 1472]]
    coordinate_rois[2] = [[316,482],[202,185],[466,95],[892,142], [1136,280],[1134,489], [924,658], [318,662]]
    labels_allow_helmet[0] = [1,2,3,4,5,6]# [[0]]
    labels_allow_helmet[1] = [1,2,3,4,5,6]# [[0]]
    labels_allow_helmet[2] = [1,2,3,4,5,6]# [[0]]
    count1 = 0
    count2 = 0
    count3 = 0
    loop = 2
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        # print(ret1, ret2, ret3)

        if ret1:
            if len(dataImages1)<=loop:
                dataImages1.append(DataImage(0,1,count1,frame1))
                count1+=1
                # print('check 1')
        if ret2:
            if len(dataImages1)<=loop:
                dataImages2.append(DataImage(1,1,count2,frame2))
                count2+=1
                # print('check 2')
        if ret3:
            if len(dataImages1)<=loop:
                dataImages3.append(DataImage(2,1,count3,frame3))
                count3+=1
                # print('check 3')
        # print('len(dataImages1) : ', len(dataImages1))
        # print('len(dataImages2) : ', len(dataImages2))
        # print('len(dataImages3) : ', len(dataImages3))
        if len(dataImages1)==loop or len(dataImages2)==loop or len(dataImages3)==loop:
            dataImages = dataImages1 + dataImages2 + dataImages3

            dict_data = tba.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
            # show(dict_data)
            dataImages = []
            dataImages1 = []
            dataImages2 = []
            dataImages3 = []
            count1 = 0
            count2 = 0
            count3 = 0



        count_frame +=1

def test_loop_ip_HoldThingDetect():
    tba = AI_PersonHoldThingDetect()
    tba.Init()


    num_camera = [0,1,3]

    filename_video1 = '/home/minhssd/AI_hoabinh/thang_3.mp4'
    filename_video2 = '/home/minhssd/AI_hoabinh/thang_3.mp4'
    filename_video3 = '/home/minhssd/AI_hoabinh/holdThingDet.avi'

    cap1 = cv2.VideoCapture(filename_video1)
    cap2 = cv2.VideoCapture(filename_video2)
    cap3 = cv2.VideoCapture(filename_video3)

    if (cap3.isOpened()== False): 
        print("Error opening video stream or file")
        # sys.exit()

    dataImages = []
    dataImages1 = []
    dataImages2 = []
    dataImages3 = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 1

    coordinate_rois[0] =np.array([[38, 489],[23,9],[949,20],[933,520]])*4
    coordinate_rois[1] = np.array([[38, 489],[23,9],[949,20],[933,520]])*4
    coordinate_rois[2] = np.array([[38, 489],[23,9],[949,20],[933,520]])*4
    labels_allow_helmet[0] = [1,2,3,4,5,6]# [[0]]
    labels_allow_helmet[1] = [1,2,3,4,5,6]# [[0]]
    labels_allow_helmet[2] = [1,2,3,4,5,6]# [[0]]
    count1 = 0
    count2 = 0
    count3 = 0
    loop = 10
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        # print(ret1, ret2, ret3)

        if ret1:
            if len(dataImages1)<=loop:
                dataImages1.append(DataImage(0,1,count1,frame1))
                count1+=1
                # print('check 1')
        if ret2:
            if len(dataImages1)<=loop:
                dataImages2.append(DataImage(1,1,count2,frame2))
                count2+=1
                # print('check 2')
        if ret3:
            if len(dataImages1)<=loop:
                dataImages3.append(DataImage(2,1,count3,frame3))
                count3+=1
                # print('check 3')
        # print('len(dataImages1) : ', len(dataImages1))
        # print('len(dataImages2) : ', len(dataImages2))
        # print('len(dataImages3) : ', len(dataImages3))
        if len(dataImages1)==loop or len(dataImages2)==loop or len(dataImages3)==loop:
            dataImages = dataImages1 + dataImages2 + dataImages3

            dict_data = tba.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
            # show(dict_data)
            dataImages = []
            dataImages1 = []
            dataImages2 = []
            dataImages3 = []
            count1 = 0
            count2 = 0
            count3 = 0



        count_frame +=1


def test_loop_ip_tba():
    tba = AI_TBA()
    tba.Init()


    num_camera = [0,1,3]

    filename_video1 = '/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/ROI.mp4'
    filename_video2 = '/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/full_hat.mp4'
    filename_video3 = '/media/minhssd/New Volume/work/Documents/thuy_dien_hoa_binh/gpu/video_test/full_hat.mp4'

    cap1 = cv2.VideoCapture(filename_video1)
    cap2 = cv2.VideoCapture(filename_video2)
    cap3 = cv2.VideoCapture(filename_video3)

    if (cap3.isOpened()== False): 
        print("Error opening video stream or file")
        # sys.exit()

    dataImages = []
    dataImages1 = []
    dataImages2 = []
    dataImages3 = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 1

    coordinate_rois[0] =[[182,498],[202,185],[466,95],[892,142], [1136,280],[1134,489], [924,658], [318,662]]
    coordinate_rois[1] = [[182,498],[202,185],[466,95],[892,142], [1136,280],[1134,489], [924,658], [318,662]]
    coordinate_rois[2] = [[182,498],[202,185],[466,95],[892,142], [1136,280],[1134,489], [924,658], [318,662]]
    labels_allow_helmet[0] = [0]# [[0]]
    labels_allow_helmet[1] = [0]# [[0]]
    labels_allow_helmet[2] = [0]# [[0]]
    count1 = 0
    count2 = 0
    count3 = 0
    loop = 10
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        # print(ret1, ret2, ret3)

        if ret1:
            if len(dataImages1)<=loop:
                dataImages1.append(DataImage(0,1,count1,frame1))
                count1+=1
                # print('check 1')
        if ret2:
            if len(dataImages1)<=loop:
                dataImages2.append(DataImage(1,1,count2,frame2))
                count2+=1
                # print('check 2')
        if ret3:
            if len(dataImages1)<=loop:
                dataImages3.append(DataImage(2,1,count3,frame3))
                count3+=1
                # print('check 3')
        # print('len(dataImages1) : ', len(dataImages1))
        # print('len(dataImages2) : ', len(dataImages2))
        # print('len(dataImages3) : ', len(dataImages3))
        if len(dataImages1)==loop or len(dataImages2)==loop or len(dataImages3)==loop:
            dataImages = dataImages1 + dataImages2 + dataImages3

            dict_data = tba.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)
            # show(dict_data)
            dataImages = []
            dataImages1 = []
            dataImages2 = []
            dataImages3 = []
            count1 = 0
            count2 = 0
            count3 = 0



        count_frame +=1
def pretreatment_tracking_loop_cid(dataTrackings, list_sort):
    for idx, dataTracking in enumerate(dataTrackings):
        dtectBoxs = []
        cid = dataTracking.cid
        dets_to_sort = np.empty((0,7))
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
        dataTrackings[idx].dtectBoxs = dtectBoxs
    return dataTrackings
    

def check_fps_plate():
    model_retinaa = model_retina()
    # filename_video = "/media/minhssd/New Volume/work/Documents/License_Plate_Recognition/data_raw/data_video_duong_pho_Hanoi/tu_quay/IMG_0709.MOV"

    # input_video = cv2.VideoCapture(filename_video)
    # if input_video.isOpened() == False:
    #     print("Video not found")
    #     sys.exit(1)
    # else:
    #     while(input_video.isOpened()):
    #         # Capture frame by frame
    #         ret, frame = input_video.read()
    #         if ret:
    #             dets_plate = model_retina.detect_plate(frame, file_img = False) #[[x1, y1, x2, y2], box_kps, id, conf]
    print('**********************************************************************')
    frame = cv2.imread('/home/evnadmin/Documents/AI_hoabinh/video/bienso.jpg')
    # frame = cv2.imread('/home/minhssd/Pictures/bienso.jpg')
    # cv2.imshow('img', frame)
    # cv2.waitKey(0)
    while True:
        # Capture frame by frame
        t1 = time.time()
        dets_plate = model_retinaa.detect_plate(frame, file_img = False) #[[x1, y1, x2, y2], box_kps, id, conf]
        t2 = time.time()
        print('**********************************************************************', 1/(t2-t1))
    
    
def check_false_person(filename_video, video=False):
    testSort = AI_TESTSORT()
    testSort.Init()
    print('check tba')

    num_camera = [0]#,2,3,4,5,6,7,8,9,10,11,12,13]

    # filename_video = '/home/evnadmin/Documents/AI_hoabinh/video/person.jpg'

    if video:
        cap = cv2.VideoCapture(filename_video)
    else:
        frame = cv2.imread(filename_video)
    # Read until the video is completed
    dataImages = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}
    count_frame = 0
    while True:
        if video:
            ret, frame = cap.read()
            if ret!=True:
                continue
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print('num_camera : ', num_camera)
        if count_frame%2==0:
            for cid in num_camera:
                dataImages.append(DataImage(cid,1,1,frame))
                coordinate_rois[cid] = [[182,498],[202,185],[466,95],[892,142], [1136,280],[1134,489], [924,658], [318,662]]

            dict_data = testSort.Detect(dataImages, coordinate_rois, labels_allow_helmet, labels_allow_uniform)

            dataImages = []

        count_frame +=1


def CountPerson(filename_video, video=False):
    testSort = AI_COUNTPERSON()
    testSort.Init()
    print('check tba')

    num_camera = [0]#,2,3,4,5,6,7,8,9,10,11,12,13]

    # filename_video = '/home/evnadmin/Documents/AI_hoabinh/video/person.jpg'

    if video:
        cap = cv2.VideoCapture(filename_video)
    else:
        frame = cv2.imread(filename_video)
    # Read until the video is completed
    dataImages = []
    cid = 0
    coordinate_rois = {}
    labels_allow_helmet = {}
    labels_allow_uniform = {}

    coordinate_rois_HSV = {}


    coordinate_rois_reg={}

    coodiPersonIns={}
    coodiPersonOuts={}
    count_frame = 0
    while True:
        if video:
            ret, frame = cap.read()
            if ret!=True:
                continue
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print('num_camera : ', num_camera)
        if ret and count_frame%2==0:
            for cid in num_camera:
                dataImages.append(DataImage(cid,1,1,frame))
                coordinate_rois[cid] = [[0,0],[frame.shape[1],0],[frame.shape[1],frame.shape[0]],[0,frame.shape[0]]]
                coodiPersonIns[cid] =  [[185, 374],
                                        [285, 321],
                                        [611, 484],
                                        [599, 566]]
                coodiPersonOuts[cid] = [[87, 475],
                                        [173, 386],
                                        [583, 587],
                                        [569, 673]]
                                        # [[653, 535],
                                        # [649, 459],
                                        # [1004, 354],
                                        # [1074, 390]]

            dict_data = testSort.Detect(dataImages, coordinate_rois, coodiPersonIns, coodiPersonOuts, labels_allow_helmet, labels_allow_uniform)

            dataImages = []

        count_frame +=1
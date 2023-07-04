import cv2
from AI_Interface import AI_FaceDetect
import numpy
import time

RTSP_URL = "rtsp://admin:MQ123456@192.168.6.122:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
# RTSP_URL = "rtsp://admin:123@123a@192.168.6.142:554"
INPUT_VIDEO = "/home/mq/Videos/test2.mkv"


cap = cv2.VideoCapture(INPUT_VIDEO)

writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'),24,(1280,720))

ai_process = AI_FaceDetect()
ai_process.Init()
_count = 0 
while True:
    ret, frame = cap.read()
    if not ret:
        print("countinue")
        continue
    _count += 1
    print(_count)
    dataImage = DataImage()
    dataImage.cid = 0
    dataImage.type_id = 0
    dataImage.count = _count
    dataImage.image = frame

    _, img_outs = ai_process.Detect([dataImage], None)

    for img in img_outs:
        resized = cv2.resize(img.image, (1280,720), interpolation = cv2.INTER_AREA)
        writer.write(resized)
        # cv2.imshow("Show", resized)
        # if cv2.waitKey(30) == 27:
        #     break

    result = ai_process.GetResult()
    if len(result) > 0:
        print(len(result))     

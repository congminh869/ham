print('check 1')
from test import plate_, tba_, tunnel_, fence_, hsv_, belt_, clock_, \
                test_loop_ip_, video, personHoldThingDetect_, test_loop_ip_HoldThingDetect,\
                fence_event_test, check_fps_plate, plate_retina
import shutil
import os 
import time
print('check 2')
if __name__ == "__main__":
    if os.path.isdir('event'):
        shutil.rmtree('event')
        os.mkdir('./event/')
    else:
        os.mkdir('./event/')
        #'/home/evnadmin/Documents/AI_hoabinh/video/ROI.mp4'
    filename_video = '/home/evnadmin/Documents/AI_hoabinh/video/ROI.mp4'
    # test_hsv()
    #video()
    plate_('/home/evnadmin/Documents/AI_hoabinh/video/plate2.mkv')
    # plate_test_img()
    # plate_retina('/home/evnadmin/Documents/AI_hoabinh/video/plate2.mkv')
    # print('check 2')
    # check_fps_plate()
    # tba_(filename_video, video = True)
    # tunnel_(filename_video, video = True)
    # fence_('/home/evnadmin/Documents/AI_hoabinh/video/person.jpg')

    # hsv_()
    # belt_(filename_video)
    # clock_()

    # test_loop_ip_()
    # t1 = time.time()
    # time.sleep(5)
    # t2 = time.time()
    # print(t2-t1)
    # video(filename_video)
    # personHoldThingDetect_('/home/evnadmin/Documents/AI_hoabinh/video/hangraocut.mp4', video = True)
    # test_loop_ip_HoldThingDetect()
    # fence_event_test()

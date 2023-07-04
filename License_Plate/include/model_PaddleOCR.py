import os
import sys
import base64
import shutil
import subprocess
import threading

# __dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
# sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

# os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import json
import copy
import numpy as np
import json
import time
import logging
from PIL import Image

sys.path.insert(0, './License_Plate/PaddleOCR')
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image

####inclue
from tools.infer.include_align import *
from tools.infer.predict_rec import TextRecognizer
from tools.infer.predict_system import TextSystem
import argparse
#debug log
from inspect import currentframe, getframeinfo
import datetime

debug_log = False

def Debug_log(cf, filename, name = None):
    if debug_log:
        ct = datetime.datetime.now()
        print(f'================ [{ct}] file {filename} , line : {cf.f_lineno} {name}')

logger = get_logger()


class PaddleOcR_Reg:
    def __init__(self, IS_VISUALIZE=False, MODE_REC=True, MODE_DET=False, MODE_SYS=False, threshold=0.94):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        self.IS_VISUALIZE = IS_VISUALIZE
        self.MODE_REC = MODE_REC # mode recognition
        self.MODE_DET = MODE_DET #mode detect text
        self.MODE_SYS = MODE_SYS # mode both recognition and detect text 
        self.args = utility.parse_args()
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        if self.MODE_REC:
            self.args.rec_model_dir = './License_Plate/PaddleOCR/rec_lp_lite_80x240_only_reg_plate_training_d4m4y2023'
            self.args.rec_image_shape = "3, 48, 320"
            self.args.rec_char_dict_path = "./License_Plate/PaddleOCR/rec_lp_lite_80x240_only_reg_plate_training_d4m4y2023/en_dict.txt"
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)  
        self.threshold = 0.94 #check reg plate smaller then exit 
        #############load model paddle#############
        if self.MODE_SYS:
            print('run MODE_SYS')
            text_sys = TextSystem(args)
        if self.MODE_REC:
            print('run MODE_REC')
            self.text_recognizer = TextRecognizer(self.args)
            #3, 48, 320
            hight_ini = 80
            weigh_ini = 240
            img_ini = out_img = np.zeros((hight_ini,weigh_ini,3), np.uint8)
            rec_res_ini, _ = self.text_recognizer([img_ini])
            print('run MODE_REC done')
        if self.MODE_DET:
            print('run MODE_DET')
            text_detector = TextDetector(args)
        print('load model done 1')
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)

    def reg_plate(self, img_raw, box_kpss):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        image_decode = img_raw.copy()
        count_plate = 0
        results = []
        for box_kps in box_kpss:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            id = box_kps[2]
            kps_int = box_kps[1]
            kps = [(kps_int[0], kps_int[1]), (kps_int[2], kps_int[3]), (kps_int[4], kps_int[5]), (kps_int[6], kps_int[7]), (kps_int[8], kps_int[9])]
            x1, y1, x2, y2 = box_kps[0]

            # count_kp = 0
            coordinate_dict = {'top-right': (int(kps[1][0]-x1), int(kps[1][1]-y1)), 
                                'bottom-right': (int(kps[4][0]-x1), int(kps[4][1]-y1)), 
                                'top-left': (int(kps[0][0]-x1), int(kps[0][1]-y1)), 
                                'bottom-left': (int(kps[3][0]-x1), int(kps[3][1]-y1))}

            Debug_log(currentframe(), getframeinfo(currentframe()).filename, image_decode.shape)
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, box_kps[0])

            img_crop_lp = image_decode[y1:y2, x1:x2]
            img_copy = img_crop_lp.copy()
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, img_copy.shape)
            # cv2.imwrite('img_crop_lp_old.jpg', img_copy)
            #align image
            cropped_img = align_image(img_copy, coordinate_dict)
            # cv2.imwrite('cropped_img_new.jpg', cropped_img)

            ####################detect and recognize#############################
            img_list = []
            #detect and recognize in horizontal number plate 
            check_plate_sqare_, img_list = check_plate_sqare(cropped_img)
            if check_plate_sqare_ is not None:
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                img = check_plate_sqare_
                # cv2.imwrite('check_plate_sqare_0.jpg', img_list[0])
                # cv2.imwrite('check_plate_sqare_1.jpg', img_list[1])
            else:
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                img = cropped_img
                img_list = [cropped_img]
            starttime = time.time()
            if self.MODE_SYS: #If you combine the plate into 1 line, it will be considered as 1 word sometime 2 word
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                dt_boxes, rec_res = text_sys(img)
                txt = ''
                for text, score in rec_res:
                    txt = str(text) + ':' + str(round(score, 2)) + '--'
                if txt == '':
                    txt = 'Null'
                image_decode = cv2.putText(image_decode, txt, (x1-5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
                #save img crop
                mode_sys(dt_boxes, rec_res, img, args, count_plate)
            if self.MODE_REC:
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                # print('wait text_recognizer')
                rec_res, _ = self.text_recognizer(img_list)
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                txt_result, check_acc, arv_acc = mode_rec(rec_res, self.threshold)
                # if check_acc: #samller threshold
                #     continue

                result_format_check = check_format_plate(txt_result)
                if result_format_check==False:
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'CHECK FORMAT PLATE FALSE')
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename, txt_result)
                    # print('CHECK FORMAT PLATE FALSE', txt_result)
                # else:
                txt_result = check_format_plate_append(txt_result) # return lp + '#' or None
                if txt_result == None:
                    continue
                # txt_result+= ' ' + str(round(arv_acc, 4))
                # image_decode = cv2.putText(image_decode, txt_result, (x1-5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)
                
            if self.MODE_DET:
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                dt_boxes, _ = text_detector(img)
            
            elapse = time.time() - starttime
            # print('time inferen det and rec : ', elapse)
            #####################################################################
            count_plate += 1
            # results.append([id, txt_result, [x1, y1, x2, y2], [], [], []])
            results.append([id, txt_result, [x1, y1, x2, y2], cropped_img, kps, img_raw])
            if self.IS_VISUALIZE:
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                print("IS_VISUALIZE or SAVE_VIDEO")
                cv2.rectangle(image_decode, (x1, y1), (x2, y2), (255,0,0), 2)
                for kp in kps: 
                    cv2.circle(image_decode, kp, 9, (255,255,0), -1)
        return results



if __name__ == '__main__':
    img_paths = './image_crop/*.jpg'
    reg_plate = PaddleOcR_Reg()
    for i in range(0,100):
        path =  "./image_crop/1669623067_4507804.jpg"
        frame = cv2.imread(path)
        box_kpss = [[[873, 450, 914, 479], [881, 456, 909, 458, 894, 465, 879, 472, 907, 474], 11, 0.9968008995056152], 
                    [[71, 470, 117, 508], [82, 478, 110, 481, 94, 489, 78, 498, 107, 501], 10, 0.9967106580734253], 
                    [[786, 445, 808, 463], [791, 452, 802, 451, 797, 455, 792, 458, 802, 458], 9, 0.9540412425994873], 
                    [[1552, 475, 1577, 497], [1560, 483, 1570, 483, 1565, 487, 1559, 490, 1570, 490], 8, 0.8629198670387268], 
                    [[1378, 461, 1400, 479], [1385, 468, 1394, 468, 1389, 471, 1385, 473, 1394, 473], 7, 0.9246069192886353], 
                    [[1316, 458, 1337, 476], [1323, 465, 1331, 465, 1327, 467, 1323, 470, 1331, 470], 6, 0.9785090088844299], 
                    [[1446, 477, 1476, 502], [1454, 485, 1469, 486, 1461, 490, 1454, 495, 1469, 495], 5, 0.9813259840011597], 
                    [[1244, 506, 1276, 533], [1251, 513, 1270, 512, 1261, 519, 1252, 526, 1270, 525], 4, 0.9985417127609253], 
                    [[1084, 552, 1138, 593], [1093, 561, 1130, 561, 1111, 572, 1093, 585, 1130, 585], 3, 0.99471515417099], 
                    [[903, 624, 973, 685], [916, 631, 968, 636, 938, 655, 909, 674, 960, 679], 2, 0.999784529209137], 
                    [[277, 496, 366, 565], [288, 503, 359, 507, 322, 531, 284, 555, 356, 559], 1, 0.9999070167541504]]
        results = reg_plate.reg_plate(frame, box_kpss)
        
        for result in results:
            id = result[0]
            txt_result = str(id) +'-'+ result[1]
            x1, y1, x2, y2 = result[2]
            cropped_img_height, cropped_img_width, _ = result[3].shape
            x_crop = result[4][0][0]
            y_crop = result[4][0][1]
            frame[y_crop:y_crop+cropped_img_height, x_crop:x_crop+cropped_img_width] = result[3]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(frame, (x_crop, y_crop), (x_crop+cropped_img_width, y_crop+cropped_img_height), (147, 125, 255), 2)
            cv2.line(frame, (x_crop, y_crop+int(cropped_img_height/2)), (x_crop+cropped_img_width, y_crop+int(cropped_img_height/2)), (147, 125, 255), 2)
            cx = x1
            cy = y1 - 12
            cv2.putText(frame, txt_result, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            window_name = 'frame'
        # cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow(window_name, frame)
        # cv2.waitKey(0)

        print('*********************************************************************************')


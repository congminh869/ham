from __future__ import print_function
#debug log
from inspect import currentframe, getframeinfo
import datetime
import sys
#SORT
import skimage
sys.path.insert(0, './sort')
from sort import *

debug_log = False

def Debug_log(cf, filename, name = None):
    if debug_log:
        ct = datetime.datetime.now()
        print(f'================ [{ct}] file {filename} , line : {cf.f_lineno} {name}')
Debug_log(currentframe(), getframeinfo(currentframe()).filename)
import os
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import time

Debug_log(currentframe(), getframeinfo(currentframe()).filename)
sys.path.insert(0, './License_Plate/retinaface')
from data_retinaface import cfg_mnet, cfg_re50
from layers_retinaface.functions.prior_box import PriorBox
from utils_retinaface.nms.py_cpu_nms import py_cpu_nms
from models_retinaface.retinaface import RetinaFace
from utils_retinaface.box_utils import decode, decode_landm


Debug_log(currentframe(), getframeinfo(currentframe()).filename)
SHOW_IMG = False

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./License_Plate/retinaface/weights_retinaface/data_extend_tri_tu_Minh29_9_final/mobilenet0.25_epoch_235.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.7, type=float, help='visualization_threshold')
args = parser.parse_args()
# debug_log = True
class model_retina:
    def __init__(self):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        self.cfg = None
        if args.network == "mobile0.25":
            self.cfg = cfg_mnet
        elif args.network == "resnet50":
            self.cfg = cfg_re50
        # net and model
        self.device = "cuda:0"#torch.device("cpu" if args.cpu else "cuda")
        self.retinaPlateNet = RetinaFace(cfg=self.cfg, phase = 'test')
        self.retinaPlateNet = self.load_model(self.retinaPlateNet, args.trained_model, args.cpu)
        self.retinaPlateNet.eval()
        print('Finished loading model!')
        cudnn.benchmark = True
        
        self.retinaPlateNet = self.retinaPlateNet.to(self.device)
        self.resize = 1

        #load model sort
        # sort
        sort_max_age = 30
        sort_min_hits = 2
        sort_iou_thresh = 0.2
        self.sort_tracker = Sort(max_age=sort_max_age,
                           min_hits=sort_min_hits,
                           iou_threshold=sort_iou_thresh) # {plug into parser}

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}


    def load_model(self, model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        if load_to_cpu:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            # device = 1#torch.cuda.current_device()
            print('retina load device : ', self.device)
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, self.device)
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(self.device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        return model


    def detect_plate(self, image_path, file_img = True, check_sort = True):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        if file_img:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            img_raw = image_path
        img_raw_heigh, img_raw_width, img_raw_depth = img_raw.shape
        # print(img_raw.shape)
        # cv2.imshow('img_raw', img_raw)
        # cv2.waitKey(0)

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        tic = time.time()
        loc, conf, landms = self.retinaPlateNet(img)  # forward pass
        # print('retinaPlateNet forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        results = []
        if check_sort:
            #tracking
            dets_to_sort = np.empty((0,7))
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                conf = float(b[4])
                
                # print('type(conf) : ', type(conf))
                for i in range(0, len(b)):
                    if b[i]<0:
                        b[i]=0
                b = list(map(int, b))
                b_kps = b[5:]
                dets_to_sort = np.vstack((dets_to_sort, np.array([b[0], b[1], b[2], b[3], conf, 0, b_kps])))

            tracked_dets = self.sort_tracker.update(dets_to_sort)
            # print('tracked_dets : ', tracked_dets)
            
            for det in tracked_dets:
                conf = det[-1]
                det = list(map(int,det))
                for i in range(0, 4):
                    if det[i]<0:
                        det[i]=0
                x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                id = det[8]
                box_kps = det[9:-1]
                Debug_log(currentframe(), getframeinfo(currentframe()).filename, len(box_kps))
                Debug_log(currentframe(), getframeinfo(currentframe()).filename, det)
                Debug_log(currentframe(), getframeinfo(currentframe()).filename, box_kps)
                
                text = str(id) + '_'+ str(conf)

                results.append([[x1, y1, x2, y2], box_kps, id, conf])
                if SHOW_IMG:
                    cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cx = x1
                    cy = y1 + 12
                    cv2.putText(img_raw, text, (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                    # landms
                    cv2.circle(img_raw, (box_kps[0], box_kps[1]), 1, (0, 0, 255), 4)
                    cv2.circle(img_raw, (box_kps[2], box_kps[3]), 1, (0, 255, 255), 4)
                    cv2.circle(img_raw, (box_kps[4], box_kps[5]), 1, (255, 0, 255), 4)
                    cv2.circle(img_raw, (box_kps[6], box_kps[7]), 1, (0, 255, 0), 4)
                    cv2.circle(img_raw, (box_kps[8], box_kps[9]), 1, (255, 0, 0), 4)
                    cv2.imshow('img', img_raw)
                    cv2.waitKey(0)
        else:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                conf = float(b[4])
                
                # print('type(conf) : ', type(conf))
                for i in range(0, len(b)):
                    if b[i]<0:
                        b[i]=0
                b = list(map(int, b))
                b_kps = b[5:]

                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                results.append([[x1, y1, x2, y2], b_kps, None, conf])
        return results
import os
import sys
import numpy
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))
# print('=================================')
# print(os.path.join(__dir__, '../../LP_detect_module'))
# print(os.path.abspath(os.path.join(__dir__, '../../LP_detect_module')))
# print(sys.path.append(os.path.abspath(os.path.join(__dir__, '../../LP_detect_module'))))
# # sys.path.append(os.path.abspath(os.path.join(__dir__, '../../LP_detect_module')))
# sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../LP_detect_module')))
import LP_detect_module.FaceDetect as FaceDetect
class ModelRetinaCpp:
    def __init__(self):
        print('path config : ', os.path.abspath(os.getcwd())+'/config/config_LP.txt')
        self.face_detecter = FaceDetect.FaceDetectRetinaTRT(os.path.abspath(os.getcwd())+'/config/config_LP.yaml')
    def detect_plate(self, image, file_img = True, check_sort = True):
        cid = 0
        np_image_data = numpy.asarray(image)
        m = FaceDetect.Mat3b.from_array(np_image_data)
        ret = self.face_detecter.Detect(cid, m)
        '''
            cv::Rect_<float> bbox;
            int camera_id;
            int class_id;
            int id_tracking;
            float class_confidence;
            std::vector<cv::Point> landmark_points;
            float mask_confidence;
            std::string name;
            int isRecognized;
            int isUnknow;
            std::chrono::steady_clock::time_point time_in;
            double live_time;
            float area;
            bool exist;
            uint64_t count_frame;
            cv::Mat img;
            cv::Mat img_face;
        '''
        # print('check 1')
        results = []
        for i in ret:
            x, y, w, h = int(i.bbox.x), int(i.bbox.y), int(i.bbox.width), int(i.bbox.height)
            x1, y1, x2, y2 = int(i.bbox.x), int(i.bbox.y), int(i.bbox.width+ i.bbox.x), int(i.bbox.height+ i.bbox.y)
            # results.append([[x1, y1, x2, y2], b_kps, None, conf])
            image = cv2.rectangle(image, (int(i.bbox.x), int(i.bbox.y)), (int(i.bbox.width + i.bbox.x), int(i.bbox.height + i.bbox.y)), (255, 0, 0), 2)
            image = cv2.putText(image, str(i.id_tracking), (int(i.bbox.x), int(i.bbox.y) - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
            # print('check 2')
            # print(i.landmark_points)
            # print(i.landmark_points)
            # print(len(i.landmark_points))
            # print('check 3')
            b_kps = []
            for point in i.landmark_points:
                # print('point', (int(i.bbox.x) + point.x, int(i.bbox.y) +point.y))
                # print('check 4')
                b_kps.append(int(i.bbox.x) + point.x)
                b_kps.append(int(i.bbox.y) +point.y)
                # image = cv2.putText(image, str(count_p), (int(i.bbox.x) + point.x, int(i.bbox.y) +point.y - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
                image = cv2.circle(image, (int(i.bbox.x) + point.x, int(i.bbox.y) +point.y), radius=5, color=(0,0, 255), thickness=-1)
        
            results.append([[x1, y1, x2, y2], b_kps, None, 0.9])

        # cv2.imshow('image', cv2.resize(image, (720,720), interpolation = cv2.INTER_AREA))
        # if cv2.waitKey(1) == 27:
        #     sys.exit() 
        return results
def main_cpp():
    model_retina = ModelRetinaCpp()
    filename_video = "/home/evnadmin/Documents/AI_hoabinh/video/plate1.mkv"

    input_video = cv2.VideoCapture(filename_video)
    if input_video.isOpened() == False:
        print("Video not found")
        sys.exit(1)
    else:
        count=0
        t =0
        while(input_video.isOpened()):
            # Capture frame by frame
            ret, frame = input_video.read()
            if ret:
                t1 = time.time()
                results = model_retina.detect_plate(frame, 0) #[[x1, y1, x2, y2], box_kps, id, conf]
                t2 = time.time()
                print('results : ', results)
                print('             time infer ', int(1/(t2-t1)))
                t += int(1/(t2-t1))
                count+=1
                # print('time infer : ', int(t/count))
            else:
                break


    


def main():
    model_retina = model_retina()
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
    while True:
        # Capture frame by frame
        t1 = time.time()
        dets_plate = model_retina.detect_plate(frame, file_img = False) #[[x1, y1, x2, y2], box_kps, id, conf]
        t2 = time.time()
        print('**********************************************************************')

if __name__ == "__main__":
    main_cpp()
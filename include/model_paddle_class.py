# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
#debug log
from inspect import currentframe, getframeinfo
import datetime

debug_log = False
# Debug_log(currentframe(), getframeinfo(currentframe()).filename)
def Debug_log(cf, filename, name = None):
    if debug_log:
        ct = datetime.datetime.now()
        print(f'================ [{ct}] file {filename} , line : {cf.f_lineno} {name}')


import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../PaddleClas/')))

from ppcls.utils import config
from ppcls.engine.engine import Engine

def check_uniform(dataTrackings, labels_allow_uniform):
    Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'check_uniform')
    '''
        4:ao_dacam     
        5:ao_trang 
        6:ao_khac 

        7:quan_dacam  
        8:quan_khac 

        => check follow "ao"
        if one true is true 
        false and false => false
         
    '''
    dataTracking_uniform_false = copy.deepcopy(dataTrackings)
    # print('labels_allow_uniform : ', labels_allow_uniform)
    for idx, dataTracking in enumerate(dataTrackings):
        cid = dataTrackings[idx].cid
        label_allow = labels_allow_uniform[cid]
        label_coats = label_allow[0]
        label_pants = label_allow[1]
        # print('label_coats : ', label_coats)
        # print('label_pants : ', label_pants)
        dataTracking_uniform_false[idx].dtectBoxs = []

        for i, dtectBox in enumerate(dataTrackings[idx].dtectBoxs):
            class_id = dataTrackings[idx].dtectBoxs[i].class_ids
            # print('class_id : ', class_id)
            if class_id==None:
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                dataTrackings[idx].dtectBoxs[i].check_class_ids = False
                dataTracking_uniform_false[idx].dtectBoxs.append(dataTrackings[idx].dtectBoxs[i])
            else:
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                check_coat = [item in class_id for item in label_coats]
                check_pant = [item in class_id for item in label_pants]
                if True in check_coat:
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                    dataTrackings[idx].dtectBoxs[i].check_class_ids = True
                elif True in check_pant:
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                    dataTrackings[idx].dtectBoxs[i].check_class_ids = True
                else:
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                    dataTrackings[idx].dtectBoxs[i].check_class_ids = False
                    dataTracking_uniform_false[idx].dtectBoxs.append(dataTrackings[idx].dtectBoxs[i])


    return dataTrackings, dataTracking_uniform_false


def paddleClas(config_dir = "./config/MobileNetV1_multilabel.yaml", weight_dir= '../../weight/weight_paddle_class_v23/best_model', image_dir = 'Infer.infer_imgs=./PaddleClas/data/image/'):
    args = config.parse_args()
    print('__dir__ : ', os.path.abspath(os.path.join(__dir__, '../../weight/weight_paddle_class_v23/best_model')))
    args.config = config_dir
    args.override = [f"Arch.pretrained={os.path.abspath(os.path.join(__dir__, weight_dir))}", 
                    image_dir]
    print(args)
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    config_ = config.get_config(
        args.config, overrides=args.override, show=False)
    print(config_)
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    engine = Engine(config_, mode="infer")
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    return engine

def infer_paddleClass(engine, dataTrackings, labels_allow_uniform):
    '''
        
    '''
    # result_engine = []
    # for dataTracking in dataTrackings:
    #     dataTracking_engine = engine.infer_custom(dataTracking)
    #     result_engine.append(dataTracking_engine)
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    result_engine = engine.infer_custom_multi_frame(dataTrackings)
    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
    result_engine_check, dataTracking_uniform_false = check_uniform(result_engine, labels_allow_uniform)
    return result_engine_check, dataTracking_uniform_false

if __name__ == "__main__":
    # paddleClas()
    # config_dir = "/home/minhssd/AI_hoabinh/yoloHubLoad/config/PPLCNet_x1_0.yaml"
    # weight_dir= '/home/minhssd/AI_hoabinh/weight/vehicle_attribute_infer/inference'
    # image_dir = 'Infer.infer_imgs=/home/minhssd/AI_hoabinh/yoloHubLoad/PaddleClas/data/image_vehical'
    # engine = paddleClas(config_dir, weight_dir, image_dir)
    # engine.infer()

    config_dir = "../config/MobileNetV1_multilabel.yaml"
    weight_dir= '/home/evnadmin/Documents/AI_hoabinh/weight/weight_paddle_class_v23/best_model'
    image_dir = 'Infer.infer_imgs=/home/evnadmin/Documents/AI_hoabinh/yoloHubLoad/PaddleClas/data/image/'
    engine = paddleClas(config_dir, weight_dir, image_dir)
    engine.infer()


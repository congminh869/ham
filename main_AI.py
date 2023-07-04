from AI_context import AIContext,  toPipe_string, fromPipe_string, putPipe_image, getPipe_image
import time
import redis
import json
import pycurl
from urllib.parse import urlencode
from io import BytesIO
import base64
import os
from datetime import datetime, timedelta
from multiprocessing import Process, Pipe, Manager, Condition, Event, Queue
from threading import Thread
from redis_client import RedisClient
import torch
from ServerMJPEG import StreamingHandler, StreamingServer
from multiprocessing.managers import BaseManager
from shm.SHM import DataImage, Channel
import requests


# import paddle
# paddle.utils.run_check()
# paddle.device.cuda.device_count()

# import paddle
# import signal
# import os

# paddle.disable_signal_handler()
# # os.kill(os.getpid(), signal.SIGSEGV)



FILE_CFG = "./config.txt"

def process_Event(pipe):  
    http_host = "https://172.16.89.17"
    curl = pycurl.Curl()
    while True:
        time.sleep(1)
        json_data = fromPipe_string(pipe)
        if json_data != "":  
            print(json_data)
            json_data_a = json.loads(json_data)
            if json_data_a["type"] == "ND_KhuonMat":
                url_init = http_host + "/api/events/access-user-event"  
                print(url_init)

                # headers = {"Content-Type": "multipart/form-data; boundary=l3iPy71otz"}
                data = {"deviceId" : json_data_a["deviceId"],
                        "time" : json_data_a["time"],
                        "type" : json_data_a["type"],
                        "left_eye" : json_data_a["data"]["left_eye"],
                        "right_eye" : json_data_a["data"]["right_eye"],
                        "nose" : json_data_a["data"]["nose"],
                        "mouth_left" : json_data_a["data"]["mouth_left"],
                        "mouth_right" : json_data_a["data"]["mouth_right"]}

                print(json_data_a["imagePaths"][0])
                file_img = open(json_data_a["imagePaths"][0] ,'rb')
                file = {'faceImage': file_img}

                x = requests.post(url_init, data= data, files = file, verify = False)

            else:
                url_init = http_host + "/api/events"  
                print(url_init)
                curl.setopt(pycurl.POST, 1)
                curl.setopt(pycurl.URL, url_init)
                curl.setopt(pycurl.CONNECTTIMEOUT, 30)
                curl.setopt(pycurl.SSL_VERIFYPEER, 0)   
                curl.setopt(pycurl.SSL_VERIFYHOST, 0)  
                buffer = BytesIO()
                curl.setopt(pycurl.HTTPHEADER, ['Accept: application/json',
                                'Content-Type: application/json'])
                curl.setopt(pycurl.WRITEFUNCTION, buffer.write)
                curl.setopt(pycurl.POSTFIELDS, json_data)    
                curl.perform()

                status_code = curl.getinfo(pycurl.RESPONSE_CODE)
                if status_code == 200 or status_code == 201:
                    resp_json = buffer.getvalue().decode('utf8')
                    print("Send Event succeed : ", resp_json)
                else: 
                    print("Send Event error {0}\nResponse {1} ".format(status_code, buffer.getvalue().decode('utf8')))
                

def process_MJPEGServer(pipe, input_d):
    StreamingHandler.s_pipe = pipe
    StreamingHandler.s_input = input_d
    address = ('', 8089)
    server = StreamingServer(address, StreamingHandler)
    while True:
        server.handle_request()

if __name__ == '__main__':  
    torch.multiprocessing.set_start_method('spawn')

    parent_conn, child_conn = Pipe()

    manager = Manager()
    # input_demo = manager.dict({"id": "", "type": ""})

    channel_mjpeg = Channel(1)
    channel_mjpeg.size_shm = 3
    channel_mjpeg.width_img = 1280
    channel_mjpeg.height_img = 720
    channel_mjpeg.depth_img = 3
    
    redis_c = RedisClient()
    redis_c_AI = RedisClient()
    redis_c_demo = RedisClient()

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"


    #Config 
    with open(FILE_CFG,'r+') as f:
        cfg_str = f.read()
        cfg_json = json.loads(cfg_str)
        
        redis_c.Init(cfg_json["redis_client"]["host"], cfg_json["redis_client"]["port"], cfg_json["redis_client"]["db"], cfg_json["redis_client"]["pwd"])
        redis_c_AI.Init(cfg_json["redis_client_ai"]["host"], cfg_json["redis_client_ai"]["port"], cfg_json["redis_client_ai"]["db"], cfg_json["redis_client_ai"]["pwd"])
        redis_c_demo.Init(cfg_json["redis_client_ai"]["host"], cfg_json["redis_client_ai"]["port"], 8, cfg_json["redis_client_ai"]["pwd"])

    eprocess = Process(target=process_Event, args=(child_conn, )) 
    eprocess.daemon = False
    eprocess.start()

    input_config = FILE_CFG
    mjpeg_process = Process(target=process_MJPEGServer, args=(channel_mjpeg, input_config)) 
    mjpeg_process.daemon = False
    mjpeg_process.start()

    redis_c_demo.Set("demo", "{}")
    redis_c_demo.Set("input_demo", '{"id": "", "type": ""}') 
    
    #List AI
    LIST_AI_CTX = []
    
    count_process = 0
    while True:
        if mjpeg_process.is_alive() is False:
            # input_demo.update({"id": "", "type": ""})
            mjpeg_process = Process(target=process_MJPEGServer, args=(channel_mjpeg, FILE_CFG)) 
            mjpeg_process.daemon = False
            mjpeg_process.start()
            redis_c_demo.Set("demo", "{}")
            redis_c_demo.Set("input_demo", '{"id": "", "type": ""}')  
            print("-------------------------------------> Restart MJPGSERVER ")


        new_list_ais_redis = []
        remove_list_ais_redis = []
        list_ais_str = ""
        list_ais_str = redis_c_AI.Get("ais")
        list_ais = None
        try:
            list_ais = json.loads(list_ais_str)
        except json.decoder.JSONDecodeError:
            print("Error parse redis data AI")
        if list_ais is not None:
            print(list_ais)
            #Check new element AI context
            for ai_redis in list_ais:
                f_existed = False 
                ipos_existed = -1 
                #print("========================> AI Process ", ai_redis["type_ai"], " | keyshm ", ai_redis["key_shm"])              
                for ip, ai in enumerate(LIST_AI_CTX):
                    if ai.type_ai == ai_redis["type_ai"] and ai.key_shm == ai_redis["key_shm"]:
                        #print("Existed AI Process ", ai.type_ai, " | keyshm ", ai.key_shm)
                        ipos_existed = ip
                        f_existed = True
                        break
                if f_existed is False:
                    new_list_ais_redis.append(ai_redis)
                else:
                    LIST_AI_CTX[ipos_existed].LoadListCam(redis_c,ai_redis) 
                    
            #Check element AI context removed
            for ai in LIST_AI_CTX:
                f_existed = False
                for ai_redis in list_ais:
                    #print("======> AI ", ai.type_ai , " | ", ai_redis["type_ai"], " | ", ai.key_shm,  " | ", ai_redis["key_shm"] )
                    if ai.type_ai == ai_redis["type_ai"] and ai.key_shm == ai_redis["key_shm"]:
                        f_existed = True
                        break
                if f_existed is False:
                    remove_list_ais_redis.append(ai_redis)

        else:
            time.sleep(5)
            continue

        input_demo = redis_c_demo.Get3("input_demo") 
        if input_demo is not None:
            print(input_demo)
            if "id" in input_demo and "type" in input_demo: 
                if input_demo["id"] != "":
                    if "type" in input_demo:
                        f_existed = False 
                        ipos_existed = -1 
                        for ip, ai in enumerate(LIST_AI_CTX):
                            if ai.type_ai == input_demo["type"]:
                                ipos_existed = ip
                                f_existed = True
                                break
                        if f_existed is True:
                            if "id" in input_demo:
                                LIST_AI_CTX[ipos_existed].LoadCamDemo(redis_c, int(input_demo["id"]), redis_c_demo) 
                else:
                    for ip in range(len(LIST_AI_CTX)):
                        LIST_AI_CTX[ip].ClearCamDemo(redis_c_demo) 



        #Run new AI process
        print("Len new = ", len(new_list_ais_redis))
        for i in range(len(new_list_ais_redis)):
            ai_ctx = AIContext()
            ai_ctx.Load(redis_c,new_list_ais_redis[i], count_process)
            count_process+=1
            LIST_AI_CTX.append(ai_ctx)
            # if LIST_AI_CTX[len(LIST_AI_CTX) - 1].type_ai == "GS_KhuVuc_ThiCong":
            LIST_AI_CTX[len(LIST_AI_CTX) - 1].Run(parent_conn, channel_mjpeg)
            
            
        #Stop AI prcoess
        print("Len remove = ", len(remove_list_ais_redis))
        for i in range(len(remove_list_ais_redis)):
            for j in range(len(LIST_AI_CTX)):
                if LIST_AI_CTX[j].type_ai == remove_list_ais_redis[i]["type_ai"] and  LIST_AI_CTX[j].key_shm == remove_list_ais_redis[i]["key_shm"]:
                    #print(">>>>>>>>>>>>",remove_list_ais_redis[i]["type_ai"])
                    LIST_AI_CTX[j].Stop()
                    LIST_AI_CTX.pop(j)
                    break           

        #Check status
        print("============================================")
        print(datetime.now())
        list_worker_stop_id = []
        for i in range(len(LIST_AI_CTX)):
            # if LIST_AI_CTX[i].type_ai == "GS_KhuVuc_ThiCong":
            if LIST_AI_CTX[i].isRunning() is True:
                print("\tprocess ", LIST_AI_CTX[i].name, " running")
            else:
                print("\tprocess ", LIST_AI_CTX[i].name, " stopped")
                list_worker_stop_id.append(i)

        #Restart Worker stop
        for i in range(len(list_worker_stop_id)):
            # if LIST_AI_CTX[i].type_ai == "GS_KhuVuc_ThiCong":
            LIST_AI_CTX[i].Stop()            
            LIST_AI_CTX[i].Restart(parent_conn, channel_mjpeg)
    
        time.sleep(5)
        

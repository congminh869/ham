import sysv_ipc
import cv2
import numpy as np
import hashlib
import json
import time
import shutil
import subprocess
import threading

class Channel:
    list_cid = []
    key_shm = -1
    size_shm = 5
    width_img = 1920
    height_img = 1080
    depth_img = 3
    
    def __init__(self, _key):
        self.key_shm = _key

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

class BDataImage:
    image = None     
    
    def __init__(self):
        self.image = None        



class SHMReader:   
    buf_sz = 0   
    
    f_first = True
    g_config_trigger = False
    
    info_channel_shm = None
    memory = None
    count_write = -1

    time_begin_reset = 0 
    
    def __init__(self, _channel):
        self.info_channel_shm = _channel        
  
    def _funcSort(self, e):
        return e.count    
        
    def _ParserData(self,memory_value):
        array_frame = []
        array_ret = []
        t_get = 0
        #Get image             
        first_pos = 0
        t1_1 = time.time()
        #Check header 
        if memory_value[first_pos:first_pos+4] != b'\x88\x88\x88\x88':
            return []
            
        #Get camera ID
        cid = int.from_bytes(memory_value[first_pos + 4 :first_pos + 8], byteorder='big')
        f_existed_cid = False
        count_frame_prev = -1
        i_pos_existed = -1

        if time.time() - self.time_begin_reset > 5:
            self.time_begin_reset = time.time()
            self.info_channel_shm.list_cid = []

        for i in range(len(self.info_channel_shm.list_cid)):
            if self.info_channel_shm.list_cid[i]["cid"] == cid:
                i_pos_existed = i
                f_existed_cid = True
                count_frame_prev = self.info_channel_shm.list_cid[i]["count"]
                break

        #Get type process
        type_id = int.from_bytes(memory_value[8 + first_pos : 12 + first_pos], byteorder='big')
        
        #Get frame count
        frame_count = np.frombuffer(memory_value[12 + first_pos : 20 + first_pos], dtype='>u8')[0]
        #if cid == 0:
        # print(frame_count, " | " ,count_frame_prev)
        if frame_count <= count_frame_prev:
            return []
        # print("[", cid, "] -- FRAME COUNT = ", frame_count)
            
        if f_existed_cid is False:
            new_camera = {"cid" : cid, "count" : -1}
            self.info_channel_shm.list_cid.append(new_camera)
        else:
            self.info_channel_shm.list_cid[i_pos_existed]["count"] = frame_count            

        #Calculate MD5 checksum and compare
        # bytes_src = memory_value[4 + first_pos : 20 + first_pos]; 
        # bytes_md5 = hashlib.md5(bytes_src)
        # if memory_value[-16 + first_pos + self.buf_sz : first_pos + self.buf_sz].hex() == bytes_md5.hexdigest():
            #Convert to numpy image
        image_np = np.fromstring(memory_value[19+ first_pos:-16 + first_pos + self.buf_sz], np.uint8)
        img = image_np[1:].reshape((self.info_channel_shm.height_img , self.info_channel_shm.width_img, self.info_channel_shm.depth_img))
        data = DataImage(cid,type_id,frame_count,img)
        array_frame.append(data)
        t2_1 = time.time()
        # print("time parse = ", t2_1 -t1_1)

        return [array_frame]

        
    def Read(self):
        ret = -1
        arr = []
        count_frame_per_second = 0
        if self.g_config_trigger == True or self.f_first ==True:
            try:
                self.memory = sysv_ipc.SharedMemory(self.info_channel_shm.key_shm)
                self.f_first = False
                self.time_begin_reset = time.time()
            except:
                pass
        if self.memory is not None:
            #Read position package new
            memory_pos = self.memory.read(4)
            pos = int.from_bytes(memory_pos[:4], byteorder='big')
            # print(pos)
            if pos < self.info_channel_shm.size_shm:
                self.buf_sz = self.info_channel_shm.width_img * self.info_channel_shm.height_img * self.info_channel_shm.depth_img + 36
                memory_value = self.memory.read(self.buf_sz, offset = self.buf_sz * pos + 4)
                arr = self._ParserData(memory_value)
            if len(arr) > 0:
                ret = 1
        return ret, arr               
                
    def Config(self):
        if self.g_config_trigger == True:
            return
                    
        
        

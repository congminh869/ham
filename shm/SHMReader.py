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


class SHMReader:   
    buf_sz = 0   
    
    f_first = True
    g_config_trigger = False
    
    info_channel_shm = None
    memory = None

    def __init__(self, _channel):
        self.info_channel_shm = _channel        
  
    def _funcSort(self, e):
        return e.count    
    
    def _ParserData(self,memory_value):
        array_frame = []
        array_ret = []
        t_get = 0
        #Get image   
        for id_p in range(self.info_channel_shm.size_shm):
            t1_1 = time.time() 
            img = None
            count_frame_prev = -1
            pos_camera = 0
            f_existed_cid = False
            
            first_pos = self.buf_sz*id_p
            #Check header 
            if memory_value[first_pos:first_pos+4] != b'\x88\x88\x88\x88':
                print("continue")
                continue
                
            #Get camera ID
            cid = int.from_bytes(memory_value[first_pos + 4 :first_pos + 8], byteorder='big')

            for i in range(len(self.info_channel_shm.list_cid)):
                if self.info_channel_shm.list_cid[i]["cid"] == cid:
                    f_existed_cid = True
                    pos_camera = i
                    #count_frame_prev = self.info_channel_shm.list_cid[i]["count"]
                    break

            #Get type process
            type_id = int.from_bytes(memory_value[8 + first_pos : 12 + first_pos], byteorder='big')
            
            #Get frame count
            frame_count = np.frombuffer(memory_value[12 + first_pos : 20 + first_pos], dtype='>u8')[0]
            #if cid == 0:
            #    print("[", cid, "] -- FRAME COUNT = ", frame_count, "----- pos = ", id_p)
                
            if f_existed_cid is False:
                new_camera = {"cid" : cid, "count" : -1}
                self.info_channel_shm.list_cid.append(new_camera)
            #else:
                #print("count_frame_prev = ", count_frame_prev)    
                #if frame_count > count_frame_prev:        
                #    dict_current_count[pos_camera]["count"] = frame_count
                #else:
                    #print("continue")
                    #continue
                    
            #Calculate MD5 checksum and compare
            bytes_src = memory_value[4 + first_pos : 20 + first_pos]; 
            bytes_md5 = hashlib.md5(bytes_src)
            if memory_value[-16 + first_pos + self.buf_sz : first_pos + self.buf_sz].hex() == bytes_md5.hexdigest():
                #Convert to numpy image
                image_np = np.fromstring(memory_value[19+ first_pos:-16 + first_pos + self.buf_sz], np.uint8)
                img = image_np[1:].reshape((self.info_channel_shm.height_img , self.info_channel_shm.width_img, self.info_channel_shm.depth_img))
                data = DataImage(cid,type_id,frame_count,img)
                array_frame.append(data)
            t2_1 = time.time()
            t_get += t2_1 - t1_1
            #print("         get[",  id_p, "] = ", t2_1 - t1_1)   
        #print("         ------>Total get ", t_get)        
        #Sort frame array
        t1_s = time.time()
        for i in range(len(self.info_channel_shm.list_cid)):
            array_channel_frame = []
            for j in range(len(array_frame)):
                if array_frame[j].cid == self.info_channel_shm.list_cid[i]["cid"]:
                    array_channel_frame.append(array_frame[j])
            #if len(array_channel_frame) > 0:
            #    print("_ParserData() channel ", array_channel_frame[0].cid, " size = ", len(array_channel_frame))
            #sort
            array_channel_frame.sort(key = self._funcSort)
            
            #filter frame old
            array_channel_frame_filter = [] 
            for j in range(len(array_channel_frame)):
                if array_channel_frame[j].count > self.info_channel_shm.list_cid[i]["count"]:
                    self.info_channel_shm.list_cid[i]["count"] = array_channel_frame[j].count
                    array_channel_frame_filter.append(array_channel_frame[j])
            
            array_ret.append(array_channel_frame_filter)
        t2_s = time.time()
        t_get += t2_s - t1_s
        #print("         sort = ", t2_s - t1_s)   
        #print("         ------>Total sort ", t_get)  
        return array_ret
        
    def Read(self):
        ret = -1
        arr = []
        count_frame_per_second = 0
        if self.g_config_trigger == True or self.f_first ==True:
            try:
                self.buf_sz = self.info_channel_shm.width_img * self.info_channel_shm.height_img * self.info_channel_shm.depth_img + 36
                self.memory = sysv_ipc.SharedMemory(self.info_channel_shm.key_shm)
                self.f_first = False

            except:
                pass
        if self.memory is not None:
            t1 = time.time()
            #self.buf_sz = self.info_channel_shm.width_img * self.info_channel_shm.height_img * self.info_channel_shm.depth_img + 36
            memory_value = self.memory.read(self.info_channel_shm.size_shm * self.buf_sz)
            #print("Size = ", len(memory_value), "----", self.buf_sz * 7)
            if len(memory_value) < self.info_channel_shm.size_shm * self.buf_sz:
                #print("Frame failed")
                return -1, []
            t2 = time.time()
            #print("Time read memory = ",  t2 - t1)
            t1_p = time.time()
            arr = self._ParserData(memory_value)
            t2_p = time.time()
            #print("=>>>>>>>>>Time read parser = ",  t2_p - t1_p)
            if len(arr) > 0:
                ret = 1
        return ret, arr               
                
    def Config(self):
        if self.g_config_trigger == True:
            return
                    
        
        

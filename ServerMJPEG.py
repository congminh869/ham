import logging
import socketserver
from threading import Condition, Thread
import cv2
import traceback
import io
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from shm.SHM import Channel, DataImage, SHMReader
from urllib.parse import urlparse
from urllib.parse import parse_qs
import struct
import numpy
from redis_client import RedisClient
import json

def putPipe_image(pipe , frame):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    h, w =  frame.image.shape[:2]
    shape = struct.pack('>IIQII',frame.cid, frame.type_id, frame.count, h, w)
    encoded = shape + frame.image.tobytes()
    pipe.send_bytes(encoded)
    return

def getPipe_image(pipe):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = None
    if pipe.poll(0.1):
        encoded = pipe.recv_bytes()
    if encoded is None:
        return None
    cid, type_id, count, h, w = struct.unpack('>IIQII',encoded[0:24])
    if w != 1280 or h != 720:
        return None
    image = numpy.frombuffer(encoded, dtype=numpy.uint8, offset=24).reshape(h,w,3)
    frame = DataImage(cid, type_id, count, image.copy())
    return frame

class StreamingHandler(BaseHTTPRequestHandler):
    s_pipe = None
    s_input = ""

    def do_GET(self):
        parsed_url = urlparse(self.path)
        id_cam = parse_qs(parsed_url.query)['id'][0]
        ai_type = parse_qs(parsed_url.query)['ai'][0]
        print("ID camera : ", id_cam)
        print("AI type: ", ai_type)
        print("self.s_input = ", self.s_input)
        if parsed_url.path == '/stream':
            redis_c_demo = RedisClient()
            with open(self.s_input,'r+') as f:
                cfg_str = f.read()
                cfg_json = json.loads(cfg_str)        
                redis_c_demo.Init(cfg_json["redis_client_ai"]["host"], cfg_json["redis_client_ai"]["port"], 8, cfg_json["redis_client_ai"]["pwd"])
                data_s = {"id" : id_cam, "type" : ai_type}
                data_s_json= json.dumps(data_s)
                redis_c_demo.Set("input_demo", data_s_json) 
            # self.s_input.update({"id" : id_cam, "type" : ai_type})
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            # try:
            shm_reader = SHMReader(self.s_pipe)
            while True:
                # print(self.s_pipe.qsize())
                ret , arr_data_img = shm_reader.Read()
                if ret > 0:
                    for arr_cid in arr_data_img:
                        for img in arr_cid:
                            if img is None:
                                continue
                            # scale = cv2.resize(frame_data.image, (1280, 720))
                            retval, buffer_i = cv2.imencode('.jpg', img.image)
                            if retval >= 0:
                                self.wfile.write(b'--FRAME\r\n')
                                self.send_header('Content-Type', 'image/jpeg')
                                self.send_header('Content-Length', len(buffer_i))
                                self.end_headers()
                                self.wfile.write(buffer_i)
                                self.wfile.write(b'\r\n')

            # except Exception as e:
            #     traceback.print_exc()
            #     logging.warning(
            #         'Removed streaming client %s: %s',
            #         self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


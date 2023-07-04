import redis
import time
import json

class RedisClient:
    g_host = "localhost"
    g_port = 6379
    g_db = 1
    g_pwd = ""
    
    r = None
    
    def __init__(self):
        self.r = None
    
    def Init(self, _host, _port, _db, _pwd):
        self.g_host = _host
        self.g_port = _port
        self.g_db = _db
        self.g_pwd = _pwd
        
        self.r = redis.Redis(host=self.g_host, port=self.g_port, db = self.g_db, password=self.g_pwd)
        
    def Set(self, _key, _value):
        self.r.set(_key, _value)
        
    def GetKeys(self):
        ret = []
        try:
            for key in self.r.keys():
                ret.append(key.decode("utf-8"))
        except:
            print("ERROR REDIS GetKeys")
        return ret
        
    def Get(self, _key):
        ret = ""
        try:
            for key in self.r.keys():
                if key.decode() == _key:
                    ret =  self.r.get(key).decode("utf-8")
                    break;
        except:
            print("ERROR REDIS Get")
        return ret

    def Get2(self, _key):
        ret = ""
        j = {}
        try:
            for key in self.r.keys():
                if key.decode() == _key:
                    ret =  self.r.get(key).decode("utf-8")
                    _json = json.loads(ret)
                    if type(_json) is dict:
                        _json["key_redis"] = key.decode()
                    break;
            return _json
        except:
            print("ERROR REDIS Get")
        return j

    def Get3(self, _key):
        ret = ""
        _json = {}

        for key in self.r.keys():
            if key.decode() == _key:
                ret =  self.r.get(key).decode("utf-8")
                _json = json.loads(ret)
                break;
        return _json
        
    def GetAll(self):
        ret = []
        try:
            for key in self.r.keys():
                ret.append(self.r.get(key).decode("utf-8"))
        except:
            print("ERROR REDIS GetAll")
        return ret   
        
    def GetAll2(self):
        ret = []
        try:
            for key in self.r.keys():
                str_json = self.r.get(key).decode("utf-8")
                _json = json.loads(str_json)
                if type(_json) is dict:
                    _json["key_redis"] = key
                    ret.append(_json)
        except:
            print("ERROR REDIS GetAll2")
        return ret             


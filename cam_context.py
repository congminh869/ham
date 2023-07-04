import time
import json
from redis_client import RedisClient
from datetime import datetime, timedelta, date
import copy
# from onvif import ONVIFCamera
import requests


NDTTEST = True

LOCAL_TIMEZONE = 7 #Viet Nam timezone
VALUE_ROUND_PTZ = 3

TIME_RETURN_PRESET = 10
TIME_RETURN_PRESET_2 = 300

class TimeTask:
    time_begin = -1
    time_end = -1 
    days = [] #Day in week [0, 1, 2, ...]
    day_time_begin = [0,0] #[hour, minute]
    day_time_end = [0,0] #[hour, minute]
    
    def __init__(self):
        self.time_begin = -1
        self.time_end = -1
        self.days = [] 
        self.day_time_begin = [0,0] 
        self.day_time_end = [0,0] 
    
    def isOn(self):

        if self.time_begin < 0 and self.time_end < 0:
            # print("---------->timeout start :", self.time_begin , " | stop : ", self.time_end )
            return False
        if self.time_begin == 0 and self.time_end == 0:
            # print("---------->now always: | start :", self.time_begin , " | stop : ", self.time_end )
            return True
        dt = datetime.now()
        now = dt.timestamp()
        if now >= self.time_begin and now <= self.time_end:
            return True            
        return False
        
    def SetLoopTime(self, _day, _day_time_begin, _day_time_end):
        week = [0,1,2,3,4,5,6]        
        for i in range(len(week)):
            for j in range(len(_day)):
                if week[i] == _day[j]:
                    self.days.append(week[i])
                    break
        
        if _day_time_begin[0] >= 0 and _day_time_begin[0] <= 23:
            self.day_time_begin[0] = _day_time_begin[0]
        else:
            self.day_time_begin[0] = 0
            
        if _day_time_begin[1] >= 0 and _day_time_begin[1] <= 59:
            self.day_time_begin[1] = _day_time_begin[1]
        else:
            self.day_time_begin[1] = 0
            
        if _day_time_end[0] >= 0 and _day_time_end[0] <= 23:
            self.day_time_end[0] = _day_time_end[0]
        else:
            self.day_time_end[0] = 0
            
        if _day_time_end[1] >= 0 and _day_time_end[1] <= 59:
            self.day_time_end[1] = _day_time_end[1]
        else:
            self.day_time_end[1] = 0
            
    def SetTimesLocal(self, t_begin, t_end):
        if t_begin <= 0 and t_end <= 0:
            self.time_begin = t_begin
            self.time_end = t_end
        else:
            # new_time_begin = t_begin
            # new_time_end = t_end 
            # if self.time_begin > 0 and self.time_begin > new_time_begin:       
            #     self.time_begin = t_begin + LOCAL_TIMEZONE*3600
            # elif self.time_begin <= 0:
            #     self.time_begin = t_begin + LOCAL_TIMEZONE*3600
            # if self.time_end > 0 and self.time_end < new_time_end:  
            #     self.time_end = t_end + LOCAL_TIMEZONE*3600
            # elif self.time_end <= 0:
            #     self.time_end = t_end + LOCAL_TIMEZONE*3600

            self.time_begin = t_begin
            self.time_end = t_end
           
    def Update(self):
        today = date.today() 
        if len(self.days) > 0:
            dt = datetime.now()
            now = dt.timestamp()
            dt_local = datetime.fromtimestamp(now)
            x = dt_local.weekday()
            for i in self.days:
                if x == i:
                    d_begin = datetime(today.year, today.month, today.day, self.day_time_begin[0], self.day_time_begin[1], 0)
                    d_end = datetime(today.year, today.month, today.day, self.day_time_end[0], self.day_time_end[1], 0)
                    self.time_begin = d_begin.timestamp()
                    self.time_end = d_end.timestamp()    

class PresetAIContext:
    id_preset = -1
    name = ""
    ptz = [0,0,0] #x,y,z
    rules = []
    time_task = None
    isDefault = False
    
    def __init__(self):
        self.id_preset = -1
        self.name = ""
        self.ptz = [0,0,0]
        self.rules = []
        self.time_task = TimeTask()
        self.isDefault = False

class CameraAIProfileContext:
    ai_id = -1
    ai_type = ""
    profile_id = -1
    presets = []
    key = -1  
    enable_ai = False

    tour_name = ""
    tour_start_time = -1
    tour_end_time = -1
    tour_times = []
    tour_preset_ids = []

    resolution_img = []

   
    def __init__(self):
        self.ai_id = -1
        self.ai_type = ""
        self.profile_id = -1
        self.presets = []
        self.key = -1
        self.enable_ai = False 

    def LoadTour(self, _json):
        if ("name" in _json) and ("startTime" in _json) and ("endTime" in _json) and ("data" in _json) and ("tourPresets" in _json):
            self.tour_name = _json["name"]
            datetime_begin = datetime.strptime(_json[i]["timeBegin"], '%Y-%m-%dT%H:%M:%S.%fZ')
            self.tour_start_time = int(datetime_begin.timestamp())
            datetime_end = datetime.strptime(_json[i]["timeEnd"], '%Y-%m-%dT%H:%M:%S.%fZ')
            self.tour_end_time = int(datetime_end.timestamp())
            if ("times" in _json["data"]):
                for i in range(len(times)):
                    datetime_t = datetime.strptime(_json["data"]["times"][i], '%Y-%m-%dT%H:%M:%S.%fZ')
                    t = int(datetime_t.timestamp())
                    self.tour_times.append(t)
            for preset_json in _json["tourPresets"]:
                if ("presetId" in preset_json) and ("standbyTime" in preset_json):
                    ele = [preset_json["presetId"], preset_json["standbyTime"]] 
                    self.tour_preset_ids.append(ele)
        
    def LoadRules(self, _json, ai_type_check = ""):
        if self.enable_ai is True: 
            time_default = [0,0]  
            if self.ai_type == "GS_KhuVuc" or self.ai_type == "GS_KhuVuc_ThiCong" or self.ai_type == "GS_NguoiMangVatThe"  or self.ai_type == "GS_DoBaoHo_DaiAnToan":
                time_default = [-1, -1]      
            if len(_json) == 0: #Fix value -> TODO
                # if self.enable_ai is True:
                pr = PresetAIContext()
                pr.time_task.SetTimesLocal(-1, -1)
                rule = {
                    "id" : 1,
                    "name" : "__default__",
                    "type" : "rectangle",
                    "x" : 0,
                    "y" : 0,
                    "width" : -1,
                    "height" : -1,
                    "time" : time_default
                }
                pr.isDefault = True
                pr.rules.append(rule)
                if time_default[0] == 0:
                    pr.time_task.SetTimesLocal(0, 0)
                self.presets.append(pr)

            for i in range(len(_json)):
                pr = PresetAIContext()
                pr.time_task.SetTimesLocal(-1, -1)
                pr.rules = []

                if "id" in _json[i]: 
                    pr.id_preset = _json[i]["id"]
                if "name" in _json[i]: 
                    pr.name = _json[i]["name"]
                if "isDefault" in _json[i]: 
                    pr.isDefault = _json[i]["isDefault"]
                if "data" in _json[i]:
                    if "ptz" in _json[i]["data"]:    
                        pr.ptz = [float(_json[i]["data"]["ptz"]["x"]), float(_json[i]["data"]["ptz"]["y"]), float(_json[i]["data"]["ptz"]["z"])]
                     
                if "rules" in _json[i]:
                    #print("===========================>RULE ",  _json[i]["rules"])
                    for j in range(len(_json[i]["rules"])):
                        t_begin = -1
                        t_end = -1
                        if "data" in _json[i]["rules"][j]:
                            #print("===========================>RULE ",  _json[i]["rules"][j]["data"])
                            if "area" in _json[i]["rules"][j]["data"]:
                                ai_ty = _json[i]["rules"][j]["data"]["area"]
                                if ai_type_check != "":
                                    if ai_ty == ai_type_check:
                                        self.ai_type = ai_type_check
                                    else:
                                        continue
                                if self.ai_type == "GS_KhuVuc_HanChe": 
                                        pr.time_task.SetTimesLocal(0, 0)
                                else:
                                    date_o = ""
                                    date_o2 = ""
                                    if "dates" in _json[i]["rules"][j]["data"]:  
                                        date_o = _json[i]["rules"][j]["data"]["dates"][1].split("T")[0]
                                        date_o2 = _json[i]["rules"][j]["data"]["dates"][0].split("T")[0]
                                    elif "date" in _json[i]["rules"][j]["data"]:  
                                        date_o = _json[i]["rules"][j]["data"]["date"]

                                    if "times" in _json[i]["rules"][j]["data"]:
                                        if len(_json[i]["rules"][j]["data"]["times"]) == 2 and date_o != "":
                                            str_dt =  date_o + "T" + str(_json[i]["rules"][j]["data"]["times"][0])
                                            # print(self.ai_type, "  ", str_dt)
                                            datetime_begin = datetime.strptime(str_dt, '%Y-%m-%dT%H:%M:%S')
                                            t_begin = int(datetime_begin.timestamp())
                                            if date_o2 != "":
                                                str_dt =  date_o2 + "T" + str(_json[i]["rules"][j]["data"]["times"][1])
                                            else:
                                                str_dt =  date_o + "T" + str(_json[i]["rules"][j]["data"]["times"][1])

                                            datetime_end = datetime.strptime(str_dt, '%Y-%m-%dT%H:%M:%S')
                                            t_end = int(datetime_end.timestamp())
                                            pr.time_task.SetTimesLocal(t_begin, t_end)
                                        else:
                                            if self.ai_type != "GS_KhuVuc" and self.ai_type != "GS_KhuVuc_ThiCong" and self.ai_type != "GS_NguoiMangVatThe"  and self.ai_type != "GS_DoBaoHo_DaiAnToan":
                                                pr.time_task.SetTimesLocal(0, 0)
                            else:
                                if self.ai_type != "GS_KhuVuc" and self.ai_type != "GS_KhuVuc_ThiCong" and self.ai_type != "GS_NguoiMangVatThe"  and self.ai_type != "GS_DoBaoHo_DaiAnToan":
                                    pr.time_task.SetTimesLocal(0, 0) 

                        if "id" in _json[i]["rules"][j] and "name" in _json[i]["rules"][j] and "data" in _json[i]["rules"][j]:
                             if "shapes" in  _json[i]["rules"][j]["data"]:
                                if len(_json[i]["rules"][j]["data"]["shapes"]) > 0:
                                    #Polygon  
                                    if _json[i]["rules"][j]["data"]["shapes"][0]["type"] == "polygon":                              
                                        if "points" in  _json[i]["rules"][j]["data"]["shapes"][0]:
                                            rule = {
                                                "id" : _json[i]["rules"][j]["id"],
                                                "name" : _json[i]["rules"][j]["name"],
                                                "type" : _json[i]["rules"][j]["data"]["shapes"][0]["type"],
                                                "points" : _json[i]["rules"][j]["data"]["shapes"][0]["points"],
                                                "time" : [t_begin + LOCAL_TIMEZONE*3600, t_end + LOCAL_TIMEZONE*3600]
                                            }
                                            if t_begin < 0 and t_end < 0:
                                                rule["time"] = time_default
                                            pr.rules.append(rule)
                                    #Rectangle
                                    if _json[i]["rules"][j]["data"]["shapes"][0]["type"] == "rectangle":  
                                        rule = {
                                            "id" : _json[i]["rules"][j]["id"],
                                            "name" : _json[i]["rules"][j]["name"],
                                            "type" : _json[i]["rules"][j]["data"]["shapes"][0]["type"],
                                            "x" : _json[i]["rules"][j]["data"]["shapes"][0]["x"],
                                            "y" : _json[i]["rules"][j]["data"]["shapes"][0]["y"],
                                            "width" : _json[i]["rules"][j]["data"]["shapes"][0]["width"],
                                            "height" : _json[i]["rules"][j]["data"]["shapes"][0]["height"],
                                            "time" : [t_begin + LOCAL_TIMEZONE*3600, t_end + LOCAL_TIMEZONE*3600]
                                        }
                                        if t_begin < 0 and t_end < 0:
                                            rule["time"] = time_default
                                        pr.rules.append(rule) 
                                    #Circle
                                    if _json[i]["rules"][j]["data"]["shapes"][0]["type"] == "circle":  
                                        rule = {
                                            "id" : _json[i]["rules"][j]["id"],
                                            "name" : _json[i]["rules"][j]["name"],
                                            "type" : _json[i]["rules"][j]["data"]["shapes"][0]["type"],
                                            "x" : _json[i]["rules"][j]["data"]["shapes"][0]["x"],
                                            "y" : _json[i]["rules"][j]["data"]["shapes"][0]["y"],
                                            "radius" : _json[i]["rules"][j]["data"]["shapes"][0]["radius"],
                                            "time" : [t_begin +LOCAL_TIMEZONE*3600, t_end + LOCAL_TIMEZONE*3600]
                                        }
                                        if t_begin < 0 and t_end < 0:
                                            rule["time"] = time_default
                                        pr.rules.append(rule)

                    if len(_json[i]["rules"]) == 0:
                        if self.enable_ai is True:
                            rule = {
                                "id" : 1,
                                "name" : "__default__",
                                "type" : "rectangle",
                                "x" : 0,
                                "y" : 0,
                                "width" : -1,
                                "height" : -1,
                                "time" : time_default
                            }
                            if time_default[0] == 0:
                                pr.time_task.SetTimesLocal(0, 0)
                            pr.rules.append(rule) 

                else:
                    rule = {
                        "id" : 1,
                        "name" : "__default__",
                        "type" : "rectangle",
                        "x" : 0,
                        "y" : 0,
                        "width" : -1,
                        "height" : -1,
                        "time" : time_default
                    }
                    if time_default[0] == 0:
                        pr.time_task.SetTimesLocal(0, 0)
                    pr.rules.append(rule) 
                    
                if len(pr.rules) > 0:                  
                    self.presets.append(pr)

    def isOnAI(self):
        if self.enable_ai is True:
            if len(self.tour_preset_ids) == 0:
                for preset in self.presets:
                    print("[", self.ai_type ,"]Preset ", preset.name, "  ", preset.time_task.isOn())
                    if preset.time_task.isOn():
                        return True
                return False
            else:
                dt = datetime.now()
                now = dt.timestamp()
                if now >= self.tour_start_time and now <= self.tour_end_time:
                    for t in tour_times:
                        if now >= t - 2*TIME_UPDATE_REDIS:
                            return True
                return False
        return False
                
    def PrintRules(self):
        for i in range(len(self.presets)):
            print("-----------------------------------")
            print("PRESET :ID ", self.presets[i].id_preset, " | Name ", self.presets[i].name, " | ", self.ai_type)
            print("        Preset ", self.presets[i].ptz)
            for j in range(len(self.presets[i].rules)):
                print("        Rule ", j , ":", self.presets[i].rules[j])
                print("        Time begin rule", j , ":", datetime.fromtimestamp(self.presets[i].rules[j]["time"][0]))
                print("        Time end rule", j , ":", datetime.fromtimestamp(self.presets[i].rules[j]["time"][1]))
            print("Time begin :", datetime.fromtimestamp(self.presets[i].time_task.time_begin))
            print("Time end :", datetime.fromtimestamp(self.presets[i].time_task.time_end))
                

               
        

class CameraProfileContext:
    profile_id = -1
    name = ""
    channel = -1
    isMain = False
    isThermal = False
    rtsp_url = ""
    resolution = []

    def __init__(self):
        self.profile_id = -1
        self.name = ""
        self.channel = -1
        self.isMain = False
        self.isThermal = False
        self.rtsp_url = ""

class CameraContext:
    id_cam = -1
    name = ""
    host = ""
    acc = ""
    pwd = ""
    profiles = []
    ais = []
    key_redis = ""
    isPTZ = False
    current_ptz_id = -1
    current_ptz = []
    isThremalCamera = False 
    current_ais_profile = -1

    is_Load = False
    is_Connect =False
	
    def __init__(self):
        self.id_cam = -1
        self.name = ""
        self.host = ""
        self.acc = ""
        self.pwd = ""
        self.key_redis = ""
        self.profiles = []
        self.ais = []
        self.isThremalCamera = False 
        self.isFaceDoor = False
	    
    def isLoad(self):
        return self.is_Load
    
    def isConnect(self):
        return self.is_Connect
        
    def GetRTSPBest(self):
        for i in range(len(self.profiles)):
            if self.profiles[i].isMain is True:
                return self.profiles[i].rtsp_url
        if len(self.profiles) > 0:
            return self.profiles[0].rtsp_url
        return ""

    def __GetBestProfile(self):
        if self.current_ais_profile == -1:    
            for i in range(len(self.profiles)):
                if self.profiles[i].resolution[1] == 1080:
                    return self.profiles[i].profile_id
            for i in range(len(self.profiles)):
                if self.profiles[i].isMain is True:
                    return self.profiles[i].profile_id
        else:
            for i in range(len(self.profiles)):
                if self.profiles[i].resolution[1] > self.current_ais_profile:
                    return self.profiles[i].profile_id
        return self.profiles[0].profile_id

        
    def CopyAIS(self, ais):
        self.ais = []
        self.ais = ais
        
    def UpdateAIS(self, _json):
        self.ais = []
        if "ais" in _json:
            for i in range(len(_json["ais"])):
                element = _json["ais"][i]
                count_type_khuvuc = 0
                pre_type = ""
                if "aiType" in element:
                    if element["aiType"] == "GS_KhuVuc":
                        if "presets" in element:
                            for pr in element["presets"]:
                                if count_type_khuvuc == 2:
                                    break
                                if "rules" in pr:
                                    for rule_i in pr["rules"]:
                                        if count_type_khuvuc == 2:
                                            break
                                        if "data" in rule_i:
                                            if "area" in rule_i["data"]:                                               
                                                if rule_i["data"]["area"] == "GS_KhuVuc_ThiCong":
                                                    if pre_type != "GS_KhuVuc_ThiCong":
                                                        count_type_khuvuc+=1
                                                        pre_type =  "GS_KhuVuc_ThiCong"
                                                elif rule_i["data"]["area"] == "GS_KhuVuc_HanChe":
                                                    if pre_type != "GS_KhuVuc_HanChe":
                                                        count_type_khuvuc+=1
                                                        pre_type =  "GS_KhuVuc_HanChe"

                print("count_type_khuvuc = ", count_type_khuvuc)
                if count_type_khuvuc == 0:
                    element = _json["ais"][i]
                    ai_ctx = CameraAIProfileContext()
                    if ("id" in element) and ("aiType" in element) and ("settings" in element) and ("presets" in element) :
                        ai_ctx.ai_id = element["id"]
                        ai_ctx.ai_type = element["aiType"]
                        ai_ctx.enable_ai = element["enabled"]
                        if  element["settings"] is not None:
                            if "profileId" in element["settings"] :
                                ai_ctx.profile_id = element["settings"]["profileId"]
                            else:
                                ai_ctx.profile_id =  self.__GetBestProfile()
                        self.current_ais_profile = ai_ctx.profile_id
                        for j in range(len(self.profiles)):
                            if self.profiles[j].profile_id == ai_ctx.profile_id: 
                                ai_ctx.resolution_img = self.profiles[j].resolution
                        ai_ctx.LoadRules(element["presets"])
                        if "tours" in element:   
                            ai_ctx.LoadTour(element["tours"])     
                        #if self.id_cam == 86:                 
                            #ai_ctx.PrintRules()            
                        self.ais.append(ai_ctx)
                elif count_type_khuvuc > 0:
                    pr_type = ""
                    for l in range(count_type_khuvuc):                  
                        element = _json["ais"][i]
                        ai_ctx = CameraAIProfileContext()
                        if ("id" in element) and ("aiType" in element) and ("settings" in element) and ("presets" in element) :
                            ai_ctx.ai_id = element["id"]
                            ai_ctx.ai_type = element["aiType"]
                            ai_ctx.enable_ai = element["enabled"]
                            if  element["settings"] is not None:
                                if "profileId" in element["settings"]:
                                    ai_ctx.profile_id = element["settings"]["profileId"]
                                else:
                                    ai_ctx.profile_id =  self.__GetBestProfile()
                            self.current_ais_profile = ai_ctx.profile_id
                            for j in range(len(self.profiles)):
                                if self.profiles[j].profile_id == ai_ctx.profile_id: 
                                    ai_ctx.resolution_img = self.profiles[j].resolution
                            if pr_type != "GS_KhuVuc_ThiCong":                             
                                ai_ctx.LoadRules(element["presets"], "GS_KhuVuc_ThiCong")
                                if ai_ctx.ai_type == "GS_KhuVuc_ThiCong":                                
                                    pr_type = "GS_KhuVuc_ThiCong"
                            if pr_type != "GS_KhuVuc_HanChe" and pr_type != "GS_KhuVuc_ThiCong": 
                                ai_ctx.LoadRules(element["presets"], "GS_KhuVuc_HanChe")
                                if ai_ctx.ai_type == "GS_KhuVuc_HanChe":                                
                                    pr_type = "GS_KhuVuc_HanChe"
                            if "tours" in element:   
                                ai_ctx.LoadTour(element["tours"])                      
                            #ai_ctx.PrintRules()            
                            self.ais.append(ai_ctx)
                            if ai_ctx.ai_type == "GS_KhuVuc_ThiCong":
                                ai_ctx2 = CameraAIProfileContext()
                                ai_ctx2 = copy.copy(ai_ctx)
                                ai_ctx2.ai_type = "GS_NguoiMangVatThe"
                                self.ais.append(ai_ctx2)  
                                ai_ctx3 = CameraAIProfileContext()
                                ai_ctx3 = copy.copy(ai_ctx)
                                ai_ctx3.ai_type = "GS_DoBaoHo_DaiAnToan"
                                self.ais.append(ai_ctx3)
        
    def Load(self, _json_str):
        #_json = json.loads(_json_str)
        if _json_str is None:
            return -1
        _json = _json_str
        if type(_json) is dict:
            if "id" in _json:
                self.id_cam = _json["id"]
            if "name" in _json:
                self.name = _json["name"]
            if "host" in _json:
                self.host = _json["host"]
            if "username" in _json:
                self.acc = _json["username"]
            if "password" in _json:
                self.pwd = _json["password"]
            if "key_redis" in _json:
                self.key_redis = _json["key_redis"]
            if "currentPreset" in _json:
                if _json["currentPreset"] is not None:
                    ptz = _json["currentPreset"]["data"]["ptz"]
                    self.current_ptz = [float(ptz["x"]), float(ptz["y"]), float(ptz["z"])]
                    self.current_ptz_id = _json["currentPreset"]["id"]
                else:
                    self.current_ptz = None
            if "caps" in _json:
                if _json["caps"] is not None:
                    if "ptz" in _json["caps"]:
                        self.isPTZ = _json["caps"]["ptz"]
                    if "thermal" in _json["caps"]:
                        self.isThremalCamera = _json["caps"]["thermal"]
                    if "door" in  _json["caps"]:
                        self.isFaceDoor = _json["caps"]["door"]
                        if self.isFaceDoor is True:
                            return -1
        else:
            print("Load data json failed ")
            return -1
        
        for i in range(len(_json["profiles"])):
            element = _json["profiles"][i]
            ctx = CameraProfileContext()
            ctx.profile_id = element["id"]
            if ("name" in element) and ("channel" in element) and ("streamUrl" in element):
                ctx.name = element["name"]
                ctx.channel = element["channel"]
                arrstr = element["resolution"].split("x")
                ctx.resolution = [int(arrstr[0]), int(arrstr[1])]                
                if ctx.name.find("main") >= 0  or ctx.name.find("Main") >= 0 or	ctx.name.find("MAIN") >= 0:
                    if self.isThremalCamera is True and ctx.channel == 2: 
                        ctx.isMain = False
                    else:
                        ctx.isMain = True
			      
                ctx.rtsp_url = self._genRTSPUrlwithAuth(element["streamUrl"])
                self.profiles.append(ctx);
         
        if "ais" in _json: 
            for i in range(len(_json["ais"])):
                element = _json["ais"][i]
                # ai_ctx = CameraAIProfileContext()
                # #print("Cam context load ========================================>",self.name)
                # if "enabled" in element:
                #     if element["enabled"] is False:
                #         continue
                # if ("id" in element) and ("aiType" in element) and ("settings" in element) and ("presets" in element)  and ("profileId" in element["settings"]) :
                #     ai_ctx.ai_id = element["id"]
                #     ai_ctx.ai_type = element["aiType"]
                #     ai_ctx.profile_id = element["settings"]["profileId"]
                #     for j in range(len(self.profiles)):
                #         if self.profiles[j].profile_id == ai_ctx.profile_id: 
                #             ai_ctx.resolution_img = self.profiles[j].resolution
                #     ai_ctx.LoadRules(element["presets"])
                #     if "tours" in element: 
                #         ai_ctx.LoadTour(element["tours"])
                #     #ai_ctx.PrintRules()            
                #     self.ais.append(ai_ctx)

                pre_type = ""
                count_type_khuvuc = 0
                if "aiType" in element:
                    if element["aiType"] == "GS_KhuVuc":
                        if "presets" in element:
                            for pr in element["presets"]:
                                if count_type_khuvuc == 2:
                                    break
                                if "rules" in pr:
                                    for rule_i in pr["rules"]:
                                        if count_type_khuvuc == 2:
                                            break
                                        if "data" in rule_i:
                                            if "area" in rule_i["data"]:                                               
                                                if rule_i["data"]["area"] == "GS_KhuVuc_ThiCong":
                                                    if pre_type != "GS_KhuVuc_ThiCong":
                                                        count_type_khuvuc+=1
                                                        pre_type =  "GS_KhuVuc_ThiCong"
                                                elif rule_i["data"]["area"] == "GS_KhuVuc_HanChe":
                                                    if pre_type != "GS_KhuVuc_HanChe":
                                                        count_type_khuvuc+=1
                                                        pre_type =  "GS_KhuVuc_HanChe"

                if count_type_khuvuc== 0:
                    element = _json["ais"][i]
                    ai_ctx = CameraAIProfileContext()
                    if ("id" in element) and ("aiType" in element) and ("settings" in element) and ("presets" in element) :
                        ai_ctx.ai_id = element["id"]
                        ai_ctx.ai_type = element["aiType"]
                        ai_ctx.enable_ai = element["enabled"]
                        if  element["settings"] is not None:
                            if "profileId" in element["settings"] :
                                ai_ctx.profile_id = element["settings"]["profileId"]
                            else:
                                ai_ctx.profile_id =  self.__GetBestProfile()
                        self.current_ais_profile = ai_ctx.profile_id
                        for j in range(len(self.profiles)):
                            if self.profiles[j].profile_id == ai_ctx.profile_id: 
                                ai_ctx.resolution_img = self.profiles[j].resolution
                        ai_ctx.LoadRules(element["presets"])
                        if "tours" in element:   
                            ai_ctx.LoadTour(element["tours"])     
                        #if self.id_cam == 86:                 
                            #ai_ctx.PrintRules()            
                        self.ais.append(ai_ctx)
                elif count_type_khuvuc > 0:
                    pr_type = ""
                    for l in range(count_type_khuvuc):                  
                        element = _json["ais"][i]
                        ai_ctx = CameraAIProfileContext()
                        if ("id" in element) and ("aiType" in element) and ("settings" in element) and ("presets" in element) :
                            ai_ctx.ai_id = element["id"]
                            ai_ctx.ai_type = element["aiType"]
                            ai_ctx.enable_ai = element["enabled"]
                            if  element["settings"] is not None:
                                if "profileId" in element["settings"]:
                                    ai_ctx.profile_id = element["settings"]["profileId"]
                                else:
                                    ai_ctx.profile_id =  self.__GetBestProfile()
                            self.current_ais_profile = ai_ctx.profile_id
                            for j in range(len(self.profiles)):
                                if self.profiles[j].profile_id == ai_ctx.profile_id: 
                                    ai_ctx.resolution_img = self.profiles[j].resolution
                            if pr_type != "GS_KhuVuc_ThiCong":                             
                                ai_ctx.LoadRules(element["presets"], "GS_KhuVuc_ThiCong")
                                if ai_ctx.ai_type == "GS_KhuVuc_ThiCong":                                
                                    pr_type = "GS_KhuVuc_ThiCong"
                            if pr_type != "GS_KhuVuc_HanChe" and pr_type != "GS_KhuVuc_ThiCong":  
                                ai_ctx.LoadRules(element["presets"], "GS_KhuVuc_HanChe")
                                if ai_ctx.ai_type == "GS_KhuVuc_HanChe":                                
                                    pr_type = "GS_KhuVuc_HanChe"
                            if "tours" in element:   
                                ai_ctx.LoadTour(element["tours"])
                            # if self.id_cam == 141:                      
                            #     ai_ctx.PrintRules()            
                            self.ais.append(ai_ctx)
                            if ai_ctx.ai_type == "GS_KhuVuc_ThiCong":
                                ai_ctx2 = CameraAIProfileContext()
                                ai_ctx2 = copy.copy(ai_ctx)
                                ai_ctx2.ai_type = "GS_NguoiMangVatThe"
                                self.ais.append(ai_ctx2)  
                                ai_ctx3 = CameraAIProfileContext()
                                ai_ctx3 = copy.copy(ai_ctx)
                                ai_ctx3.ai_type = "GS_DoBaoHo_DaiAnToan"
                                self.ais.append(ai_ctx3)

        return 1                     
    
    def check_ais_in_timework(ctx_ai):
        if len(ctx_ai.tour_preset_ids) == 0:
            for preset in ctx_ai.presets:
                print("Preset ", preset.name, "  ", preset.time_task.isOn())
                if preset.time_task.isOn():
                    return True
            return False
        else:
            dt = datetime.now()
            now = dt.timestamp()
            if now >= ctx_ai.tour_start_time and now <= ctx_ai.tour_end_time:
                for t in tour_times:
                    if now >= t - 2*TIME_UPDATE_REDIS:
                        return True
            return False

    # def ptz_move(self, ptz_value):
    #     mycam = ONVIFCamera(self.host, 80, self.acc, self.pwd, './wsdl')
    #     print('Connected to ONVIF camera', self.host, self.acc, self.pwd)
    #     media = mycam.create_media_service()
    #     media_profile = media.GetProfiles()[0]
    #     ptz = mycam.create_ptz_service()
    #     requesta = ptz.create_type('AbsoluteMove')
    #     requesta.ProfileToken = media_profile.token
    #     if requesta.Position is None:
    #         requesta.Position = ptz.GetStatus({'ProfileToken': media_profile.token}).Position
    #     requesta.Position.PanTilt.x = ptz_value[0]
    #     requesta.Position.PanTilt.y = ptz_value[1]
    #     requesta.Position.Zoom.x = ptz_value[2]
    #     ptz.AbsoluteMove(requesta)

    def ptz_api(self, preset_id):
        url_api = "https://127.0.0.1/api/devices/{0}/presets/{1}/preview".format(self.id_cam, preset_id)
        response = requests.get(url_api, verify=False)
        print("[", self.id_cam, "] Move to preset ", preset_id, response)

    def getRuleTimeoutPTZ(self, list_timeout, ai_type):
        # return list_timeout
        #Check ai type
        f_existed = False
        i_pos = -1
        for i, ai_ctx in enumerate(self.ais):
            if ai_type == ai_ctx.ai_type:
                f_existed = True
                i_pos = i
                break

        if f_existed is False:
            return list_timeout

        for i, pr in enumerate(self.ais[i_pos].presets):
            if pr.time_task.isOn() is False:
                continue
            if self.current_ptz is not None and len(self.current_ptz) == 3 and len(pr.ptz) == 3:
                # print(self.id_cam, self.current_ptz_id, pr.id_preset)
                if self.current_ptz_id == pr.id_preset:
                # if (round(self.current_ptz[0], VALUE_ROUND_PTZ) == round(pr.ptz[0], VALUE_ROUND_PTZ)) and (round(self.current_ptz[1], VALUE_ROUND_PTZ) == round(pr.ptz[1], VALUE_ROUND_PTZ))  and (round(self.current_ptz[2], VALUE_ROUND_PTZ) == round(pr.ptz[2], VALUE_ROUND_PTZ)):
                    #Check camera exitsed in list
                    # print("------------------------------------------------------>", self.id_cam)
                    pos_cam = -1
                    pos_ais = -1
                    for j, cam in enumerate(list_timeout):
                        if int(cam["id_cam"]) == self.id_cam:
                            pos_cam = j
                            for k, ais_c in enumerate(cam["ais"]):
                                if ais_c["type"] == ai_type:
                                    pos_ais = k
                                    break
                            if pos_ais >= 0:
                                break
                    if pos_cam < 0:
                        list_timeout.append({"id_cam": self.id_cam, "ais" : [{"type" : ai_type, "time" : -1}]})
                        return list_timeout
                    if pos_ais < 0:
                        list_timeout[pos_cam]["ais"].append({"type" : ai_type, "time" : -1})
                        return list_timeout
                    else:
                        list_timeout[pos_cam]["ais"][pos_ais]["time"] = -1
                        return list_timeout
                else:
                    #Check camera exitsed in list
                    pos_cam = -1
                    pos_ais = -1
                    for j, cam in enumerate(list_timeout):
                        if int(cam["id_cam"]) == self.id_cam:
                            pos_cam = j
                            for k, ais_c in enumerate(cam["ais"]):
                                if ais_c["type"] == ai_type:
                                    pos_ais = k
                                    break
                            if pos_ais >= 0:
                                break
                    if pos_cam < 0:
                        list_timeout.append({"id_cam": self.id_cam, "ais" : [{"type" : ai_type, "time" : -1}]})
                        return list_timeout
                    if pos_ais < 0:
                        list_timeout[pos_cam]["ais"].append({"type" : ai_type, "time" : -1})
                        return list_timeout

                    if list_timeout[pos_cam]["ais"][pos_ais]["time"] < 0:
                        list_timeout[pos_cam]["ais"][pos_ais]["time"] = time.time()
                    else:
                        if ai_type == "GS_KhuVuc_ThiCong":
                            print(self.id_cam, "Time = ", time.time() - list_timeout[pos_cam]["ais"][pos_ais]["time"])
                        if time.time() - list_timeout[pos_cam]["ais"][pos_ais]["time"] >= TIME_RETURN_PRESET:
                            print("-------------------------------------> PTZ MOVE ", pr.ptz)
                            self.ptz_api(pr.id_preset)
                            list_timeout[pos_cam]["ais"][pos_ais]["time"] = -1
                        return list_timeout
            elif self.current_ptz  is None:
                pos_cam = -1
                pos_ais = -1
                for j, cam in enumerate(list_timeout):
                    if int(cam["id_cam"]) == self.id_cam:
                        pos_cam = j
                        for k, ais_c in enumerate(cam["ais"]):
                            if ais_c["type"] == ai_type:
                                pos_ais = k
                                break
                        if pos_ais >= 0:
                            break
                if pos_cam < 0:
                    list_timeout.append({"id_cam": self.id_cam, "ais" : [{"type" : ai_type, "time" : -1}]})
                    return list_timeout
                if pos_ais < 0:
                    list_timeout[pos_cam]["ais"].append({"type" : ai_type, "time" : -1})
                    return list_timeout

                if list_timeout[pos_cam]["ais"][pos_ais]["time"]  < 0:
                    list_timeout[pos_cam]["ais"][pos_ais]["time"] = time.time()
                else:
                    if time.time() - list_timeout[pos_cam]["ais"][pos_ais]["time"] >= TIME_RETURN_PRESET_2:
                        print("-------------------------------------> PTZ MOVE 2", pr.ptz)
                        self.ptz_api(pr.id_preset)
                        list_timeout[pos_cam]["ais"][pos_ais]["time"]  = -1
                    return list_timeout
        return list_timeout


            
    def getRuleCurrent(self, ai_type):
        resolution_ret = []

        #Check ai type
        f_existed = False
        i_pos = -1
        for i, ai_ctx in enumerate(self.ais):
            if ai_type == ai_ctx.ai_type:
                f_existed = True
                i_pos = i
                break

        if f_existed is False:
            return [], resolution_ret

        resolution_ret = self.ais[i_pos].resolution_img
        # print(self.ais[i_pos].ai_type)
        #Check current preset
        for i, pr in enumerate(self.ais[i_pos].presets):
            if pr.time_task.isOn() is False:
                continue
            # print(ai_type, pr.name, pr.time_task.isOn())
            if self.current_ptz is not None and len(self.current_ptz) == 3 and len(pr.ptz) == 3:
                if (round(self.current_ptz[0], VALUE_ROUND_PTZ) == round(pr.ptz[0], VALUE_ROUND_PTZ)) and (round(self.current_ptz[1], VALUE_ROUND_PTZ) == round(pr.ptz[1], VALUE_ROUND_PTZ))  and (round(self.current_ptz[2], VALUE_ROUND_PTZ) == round(pr.ptz[2], VALUE_ROUND_PTZ)):
                    # if ai_type == "GS_KhuVuc_ThiCong":
                    #     print("-------------->Compare   :   ", self.current_ptz,  " | ", pr.ptz)
                    return pr.rules, resolution_ret                

        if len(self.ais[i_pos].presets) == 1 and self.ais[i_pos].presets[0].isDefault == True:
            return self.ais[i_pos].presets[0].rules, resolution_ret
        return [], resolution_ret

    
    def _genRTSPUrlwithAuth(self, url):
        if self.acc != "" and self.pwd != "":
            pos = -1
            if url.find("rtsp://") == 0:
                pos = 7
            else:
                print("[CameraContext::_genRTSPUrlwithAuth] Error rtsp url")
                return url
            tail_url = url[pos:]
            new_url = "rtsp://" + self.acc + ":" + self.pwd + "@" + tail_url
            return new_url
        else:
            return url

def remove_camera_from_list(list_input, cam):
    for i in range(len(list_input)):
        if list_input[i].host == cam.host :
            list_input.pop(i)
            break
    return list_input

def filter_similar_cameras(list_input):
    list_output = []
    loop = True
    ipos = -1
    
    while loop:
        list_similar = []
        loop = False
        if ipos == len(list_input) - 1:
            break
            
        for i in range(ipos + 1, len(list_input)):
            for j in range(i + 1, len(list_input)):
                if list_input[i].host == list_input[j].host :
                    list_similar.append(j)
                    
            f_erase = False
            num_er = 0
            for j in range(len(list_similar)):
                list_input.pop(list_similar[j] - num_er)
                num_er+=1
                f_erase = True

            list_output.append(list_input[i])
            ipos = i
            if f_erase is True:
                list_similar.clear()
                loop = True
                break
    return list_output
     
            
def compare_list_cameras(list_1, list_2):
    if len(list_1) != len(list_2) :
        return False
        
    count_similar = 0
    for i in range(len(list_1)):
        for j in range(len(list_2)):
            if list_1[i].host == list_2[j].host :
                 count_similar+=1
                 break
    
    if count_similar < len(list_1) :
        return False
    return True
    
def get_change_list_cameras(list_old, list_current):
    list_new = []
    list_remove = []
    list_same= []
    
    for i in range(len(list_old)):
        f_existed = False
        for j in range(len(list_current)):
            if list_old[i].host == list_current[j].host :
                f_existed = True
                break
        
        if f_existed is not True:
            list_remove.append(list_old[i])
        else:
            list_same.append(list_old[i])
            
    for i in range(len(list_current)):
        f_existed = False
        for j in range(len(list_same)):
            if list_current[i].host == list_same[j].host :
                f_existed = True
                break        
        if f_existed is not True:
            list_new.append(list_current[i])
            
    return list_new, list_remove
    


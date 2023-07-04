import time
import cv2
#debug log
from inspect import currentframe, getframeinfo
import datetime
import numpy as np

debug_log = False 
# Debug_log(currentframe(), getframeinfo(currentframe()).filename)
def Debug_log(cf, filename, name = None):
    if debug_log:
        ct = datetime.datetime.now()
        print(f'================ [{ct}] file {filename} , line : {cf.f_lineno} {name}')

class ConvertEvent():
    def __init__(self, time_kill_id_seconds=600, time_check_seconds=180):
        '''
            self.data_event_time_cid = {cid :  {"person": {id: time}, 
                                        "vehicle": {id: time}, 
                                        "holdThingDet": {id: time}, 
                                        "hat": {id: time}, 
                                        "uniform": {id: time}}}
        '''
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        self.type_warnings = ["person", "vehicle", "holdThingDet", "hat", "uniform"]
        self.txt_warnings = {
                        "person": "Nguoi ra vao", 
                        "vehicle": "Xe ra vao", 
                        "holdThingDet": "Nguoi mang vat the", 
                        "hat": "Mu", 
                        "uniform": "Dong phuc", 
                        }

        self.data_event_time_cid = {}
        self.data_event_time = {"person": {}, 
                                "vehicle": {}, 
                                "holdThingDet": {}, 
                                "hat": {}, 
                                "uniform": {}}
        self.time_kill_id = time_kill_id_seconds #600
        self.time_check = time_check_seconds #180
        self.type_obj = "typeObj"
        self.warning = "warning"
        self.crop_img = "cropImg"
        self.img = "img"
        self.data = "data"

    def check_data_event_time_cid(self):
        # Debug_log(currentframe(), getframeinfo(currentframe()).filename, '*****************check_data_event_time_cid*****************')
        for cid in self.data_event_time_cid:
            for data_event_time in self.data_event_time_cid[cid]:
                ids = []
                for id in self.data_event_time_cid[cid][data_event_time]:
                    time_current = time.time()
                    time_check = time_current - self.data_event_time_cid[cid][data_event_time][id] 
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'time_check : {str(time_check)}, data_event_time: {data_event_time} , id: {str(id)}')
                    if time_check >=self.time_kill_id:
                        #remove id
                        ids.append(id)
                        # del self.data_event_time_cid[cid][data_event_time][id]
                for id_del in ids:
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'*****************remove data_event_time_cid[{str(cid)}][{str(data_event_time)}][{str(id_del)}]')
                    del self.data_event_time_cid[cid][data_event_time][id_del]
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'self.data_event_time_cid : {self.data_event_time_cid}')

    def convert(self, dict_tracking, coordinate_rois):
        dict_data = {}
        cids = []
        if debug_log:
            print('data_event_time_cid start : ', self.data_event_time_cid)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
        for idx_tw, type_warning in enumerate(self.type_warnings):
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'type_warning : '+type_warning)
            if type_warning in dict_tracking: #get key in dict_tracking
                Debug_log(currentframe(), getframeinfo(currentframe()).filename, type_warning)
                # print()
                dataTrackings = dict_tracking[type_warning]
                Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'len(dataTrackings) : '+str(len(dataTrackings)))
                for idx, dataTracking in enumerate(dataTrackings):
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'idx : '+str(idx))
                    datas = []
                    cid = dataTrackings[idx].cid

                    if cid not in self.data_event_time_cid:
                        Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'cid not in data_event_time_cid')
                        #if cid not exit
                        self.data_event_time_cid[cid] = self.data_event_time
                        Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'construction self.data_event_time_cid : {self.data_event_time_cid}')
                    
                    dtectBoxs = dataTrackings[idx].dtectBoxs
                    
                    image = dataTrackings[idx].frame

                    coordinate_roi = coordinate_rois[cid]
                    pts = np.array(coordinate_roi,np.int32)
                    # pts = pts.reshape((-1, 1, 2))                 
                    image = cv2.polylines(image, [pts],True, (0, 0, 255), 3)


                    # if cid not in dict_data:
                    #     Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'cid not int dict_data')
                    #     image = dataTrackings[idx].frame
                    # else:
                    #     Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'cid int dict_data')
                    #     Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                    #     image = dict_data[cid][self.img]
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'len(dtectBoxs) : {str(len(dtectBoxs))}')
                    
                    for dtectBox in dtectBoxs:
                        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                        x1, y1, x2, y2 = int(dtectBox.bbox[0]), int(dtectBox.bbox[1]), int(dtectBox.bbox[2]), int(dtectBox.bbox[3])
                        x = int(x1)
                        y = int(y1)
                        w = int(x2-x1)
                        h = int(y2 - y1)
                        crop_img = image[y:y+h, x:x+w]
                        # cv2.imshow('imvrop', crop_img)
                        # cv2.waitKey(0)
                        id = dtectBox.id_tracking
                        Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'id : ' + str(id))
                        #check spam warning
                        Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'self.data_event_time_cid : {self.data_event_time_cid}')
                        # if cid in self.data_event_time_cid:
                        #     Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'cid in data_event_time_cid')
                        #     #if cid exited

                        #send all event 
                        if type_warning== "holdThingDet":
                            # cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 1)
                            cv2.putText(image, self.txt_warnings[type_warning], (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
                        else:
                            cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 1)
                            cv2.putText(image, self.txt_warnings[type_warning], (x1, y1-30*idx_tw), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
                        ######################
                            
                        if id in self.data_event_time_cid[cid][type_warning]:
                            Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'id_tracking in data_event_time_cid[{str(cid)}][{type_warning}]')
                            # if id existed
                            time_current = time.time()
                            time_check = time_current - self.data_event_time_cid[cid][type_warning][id] 

                            Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'self.time_check : {self.time_check}')
                            Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'time_check : {time_check}')

                            if time_check > self.time_check:
                                Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'type_warning : {type_warning}')
                                Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'self.txt_warnings[type_warning] : {self.txt_warnings[type_warning]}')
                                # if enough time , program send event
                                data = {self.type_obj:  type_warning, self.warning: self.txt_warnings[type_warning], self.crop_img: crop_img}
                                datas.append(data)
                                #draw image put txt and draw reg
                                # cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 1)

                                # cv2.putText(image, type_warning, (x1, y1-30*idx_tw), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
                                #update time for id
                                self.data_event_time_cid[cid][type_warning][id] = time.time()
                                Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'self.data_event_time_cid : {self.data_event_time_cid}')

                        else:
                            # id do not have in 
                            Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'id_tracking not in data_event_time_cid[{str(cid)}][{type_warning}]')
                            Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'type_warning : {type_warning}')
                            Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'self.txt_warnings[type_warning] : {self.txt_warnings[type_warning]}')
                            # if enough time , program send event
                            data = {self.type_obj:  type_warning, self.warning: self.txt_warnings[type_warning], self.crop_img: crop_img}
                            datas.append(data)
                            #draw image put txt and draw reg
                            # cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 1)

                            # cv2.putText(image, type_warning, (x1, y1-30*idx_tw), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
                            self.data_event_time_cid[cid][type_warning][id] = time.time()
                            Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'self.data_event_time_cid : {self.data_event_time_cid}')
                        
                        # else:
                        #     Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'cid not in data_event_time_cid')
                        #     #if cid not exit
                        #     self.data_event_time_cid[cid] = self.data_event_time
                        #     Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'construction self.data_event_time_cid : {self.data_event_time_cid}')

                        ###################
                    if len(dtectBoxs)>0:#len(datas)!=0:
                        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                        if cid not in dict_data:
                            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                            dict_data[cid] = {self.img: image, self.data: datas}
                        else:
                            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                            dict_data[cid][self.data] = dict_data[cid][self.data]+datas
                    if debug_log:
                        name_folder = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")
                        cv2.imwrite(f'./event/image_event{name_folder}.jpg', image)
                    # cv2.imshow('image_event', cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4))))
                    # cv2.waitKey(0)
        if debug_log:
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'check dict_data')
            for cid in dict_data:
                for idx, data in enumerate(dict_data[cid][self.data]):
                    print(f'cid : {str(cid)}, idx : {str(idx)}, data[{self.type_obj}] : {data[self.type_obj]}')
                    print('./event/'+str(idx)+data[self.type_obj]+'.jpg')
                    # cv2.imshow(str(idx)+data[self.type_obj], data[self.crop_img])
                    cv2.imwrite('./event/'+str(idx)+data[self.type_obj]+'.jpg', data[self.crop_img])
                # cv2.waitKey(0)
                        # print()
        # if debug_log:
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, 'check dict_data')
        if debug_log:
            for cid in dict_data:
                cv2.imwrite(f'./event/image_event_{str(cid)}.jpg',  dict_data[cid][self.img])
        self.check_data_event_time_cid()
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'self.data_event_time_cid : {self.data_event_time_cid}')
        if debug_log:
            print('************************************************')
            print(dict_data)
            print('************************************************')
        return dict_data


class ConvertEventFence():
    def __init__(self, time_reset=60):
        self.list_num_persons = {}
        self.list_num_FSs = {}
        self.reset_list_num_persons = {}
        self.time_reset = time_reset

    def check_time_reset(self):
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, self.reset_list_num_persons)
        for cid in self.reset_list_num_persons:
            time_current = time.time()
            time_check = time_current - self.reset_list_num_persons[cid]
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'time_check : {str(time_check)}')
            if time_check > self.time_reset:
                self.list_num_persons[cid] = []
                self.reset_list_num_persons[cid] = time.time()
                    

    def convert(self, dataTrackings_person, dataTrackings_FS, coordinate_rois):
        dict_data = {}
        check_person = False
        check_FS = False
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, self.list_num_persons)
        Debug_log(currentframe(), getframeinfo(currentframe()).filename, self.reset_list_num_persons)
        for idx in range(len(dataTrackings_person)):
            Debug_log(currentframe(), getframeinfo(currentframe()).filename)
            num_persons = len(dataTrackings_person[idx].dtectBoxs)
            # print('num_persons : ', num_persons)
            num_FSs = len(dataTrackings_FS[idx].dtectBoxs)
            cid = dataTrackings_person[idx].cid
            coordinate_roi =coordinate_rois[cid]
            frame = dataTrackings_person[idx].frame

            pts = np.array(coordinate_roi,np.int32)
            # pts = pts.reshape((-1, 1, 2))                 
            frame = cv2.polylines(frame, [pts],True, (0, 0, 255), 3)


            if cid not in self.reset_list_num_persons:
                self.reset_list_num_persons[cid] = time.time()

            Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'num_persons : {str(num_persons)}')
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'num_FSs : {str(num_FSs)}')
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'cid : {str(cid)}')

            if num_persons>0:
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                if cid in self.list_num_persons:
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename, f'str(len(self.list_num_persons[cid])) : {str(len(self.list_num_persons[cid]))}')
                    if len(self.list_num_persons[cid])==0:
                        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                        self.list_num_persons[cid] = [num_persons]
                        check_person = True
                        Debug_log(currentframe(), getframeinfo(currentframe()).filename, self.list_num_persons[cid])
                    else:
                        Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                        self.list_num_persons[cid].append(num_persons)
                        Debug_log(currentframe(), getframeinfo(currentframe()).filename, self.list_num_persons[cid])
                        if len(self.list_num_persons[cid])>=3:
                            if self.list_num_persons[cid][0] != self.list_num_persons[cid][1] and self.list_num_persons[cid][0] != self.list_num_persons[cid][2]:
                                check_person = True
                                self.list_num_persons[cid] = [self.list_num_persons[cid][2]]
                                Debug_log(currentframe(), getframeinfo(currentframe()).filename, self.list_num_persons[cid])
                            else:
                                self.list_num_persons[cid] = [self.list_num_persons[cid][0], self.list_num_persons[cid][2]]
                                Debug_log(currentframe(), getframeinfo(currentframe()).filename, self.list_num_persons[cid])
                    
                else:
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                    self.list_num_persons[cid] = [num_persons]
                    check_person = True
                    Debug_log(currentframe(), getframeinfo(currentframe()).filename, self.list_num_persons[cid])
            for num_person in range(num_persons):
                x1,y1,x2,y2 = dataTrackings_person[idx].dtectBoxs[num_person].bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
                cv2.putText(frame, 'Nguoi ra vao', (x1, y1-30), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)

            for num_FSs in range(num_FSs):
                x1,y1,x2,y2 = dataTrackings_FS[idx].dtectBoxs[num_FSs].bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
                cv2.putText(frame, 'Khoi/Lua', (x1, y1-30), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)

            if check_person or num_FSs>0:
                Debug_log(currentframe(), getframeinfo(currentframe()).filename)
                if cid not in dict_data:
                    dict_data[cid] = {"img": frame, "data": [{"typeObj": "person", "num": num_persons},
                                                             {"typeObj": "fireSmoke", "num": num_FSs},]} #dataTrackings[idx].frame
                check_person = False
                name_folder = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")
                if debug_log:
                    cv2.imwrite(f'./event/holdthing_{name_folder}.jpg', frame)
                # cv2.imshow(f'event_{str(cid)}', frame)
                # cv2.imshow(f'event_{str(cid)}_{str(time.time())}', frame)
            Debug_log(currentframe(), getframeinfo(currentframe()).filename, '***************************************')

        self.check_time_reset()
        # cv2.waitKey(1)        
        return dict_data






# if __name__ == '__main__':
    # dict_data = {1: {"img": "image", "data": data}}
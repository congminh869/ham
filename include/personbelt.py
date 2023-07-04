
import time
# from include.include_main import calculate_IOU
import cv2

class MappingOneFrame:
    def __init__(self, dataTrackings_sort, dataTrackings_belt):
        self.cid = dataTrackings_sort.cid
        self.id = []
        self.time = None
        self.status = []
        self.mapping_person_belt = []
        self.dataTrackings_sort = dataTrackings_sort
        self.dataTrackings_belt = dataTrackings_belt
    def map(self):
        for person in self.dataTrackings_sort.dtectBoxs:
            for belt in self.dataTrackings_belt.dtectBoxs:
                iou = self.calculate_IOU(person.bbox, belt.bbox)
                if iou>0.7 and belt.name_class=='belt':
                    self.mapping_person_belt.append(1)
                else:
                    self.mapping_person_belt.append(0)
            if sum(self.mapping_person_belt) > 0:
                self.id.append(person.id_tracking)
                self.status.append([1])
                # check_person_belt[person.id_tracking] = [1]
            else:
                self.id.append(person.id_tracking)
                self.status.append([0])
                # check_person_belt[person.id_tracking] = [0]
            self.mapping_person_belt.clear()
    def calculate_IOU(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        # iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value

        #IOU in here is different from original, the formula:
        if boxAArea < boxBArea:
            iou = interArea / boxAArea
        else:
            iou = interArea / boxBArea

        return iou

class PersonBelt:
    def __init__(self, start_time):
        self.person_belt = []
        self.check_person_belt_nframe = {}
        self.check_person_time_nframe = {}
        self.time_check = 50
        self.start_time = start_time
        self.total_time = 0
        self.person_belt_for_draw = {}
        self.time_delete = 900
        self.dict_data = {}
        self.dict_data_tempt = None
        # self.belt_appear_count = 0
        # self.time_warning = time.time()
        # self.n_frame = n_frame//n_frame_cap #number of frame processed for warning 
        # self.n_frame_cap = n_frame_cap
        # self.n_current_frame = 27000//n_frame_cap   #number of frame processed after 15s

    def check(self, dataTrackings_sort, dataTrackings_belt):
        cids_sort = [x.cid for x in dataTrackings_sort]
        cids_belt = [x.cid for x in dataTrackings_belt]
        # print("-----------cids_sort: ", cids_sort)
        # print("-----------cids_belt: ", cids_belt)
        if len(cids_sort) != len(cids_belt):
            intersection = set(cids_sort) & set(cids_belt)  #value matching in two lists
            cids_sort = [x if x in intersection else -1 for x in cids_sort] #set value not in intersection = -1
            cids_belt = [x if x in intersection else -1 for x in cids_belt] #set value not in intersection = -1
            cids_sort_position = sorted(range(len(cids_sort)), key=lambda k: cids_sort[k])  
            cids_belt_position = sorted(range(len(cids_belt)), key=lambda k: cids_belt[k])
            cids_sort_position = cids_sort_position[len(cids_sort)-len(intersection):]  #just keep position has value in intersection
            cids_belt_position = cids_belt_position[len(cids_belt)-len(intersection):]  #just keep position has value in intersection
        else:
            cids_sort_position = sorted(range(len(cids_sort)), key=lambda k: cids_sort[k])
            cids_belt_position = sorted(range(len(cids_belt)), key=lambda k: cids_belt[k])
            # print("----------cids_sort_position: ", cids_sort_position)
            # print("----------cids_belt_position: ", cids_belt_position)
        for i in range(len(cids_sort_position)):
            mof = MappingOneFrame(dataTrackings_sort[cids_sort_position[i]], dataTrackings_belt[cids_belt_position[i]])
            mof.map()
            for i,id in enumerate(mof.id):
                if id not in self.check_person_belt_nframe:
                    self.check_person_belt_nframe[id] = [mof.status[i], 0]
                    self.check_person_time_nframe[id] = time.time()
                else:
                    self.check_person_belt_nframe[id][0] += mof.status[i]
                    self.check_person_belt_nframe[id][1] = time.time() - self.check_person_time_nframe[id]
                    if self.check_person_belt_nframe[id][1] > self.time_check:
                        if sum(self.check_person_belt_nframe[id][0]) > 0:
                            self.person_belt_for_draw[id] = "belt"
                            # self.person_belt.append({"id": id, "noBelt": False})
                            # self.belt_appear_count += 1
                        else:
                            self.person_belt_for_draw[id] = "no belt"
                            self.person_belt.append({"id": id, "noBelt": True})
                        self.check_person_belt_nframe.pop(id)
            image_labeled = self.draw_box_with_label(mof.dataTrackings_sort)
            # image_labeled = cv2.resize(image_labeled, (1280, 640), interpolation=cv2.INTER_AREA)
            # print("----------check_person_belt_nframe: ", self.check_person_belt_nframe)
            # cv2.imshow("test", image_labeled)
            # cv2.waitKey(1)
            # print("********")
            if len(self.person_belt) > 0 :
                self.dict_data[mof.cid] = {"img": image_labeled, "data": self.person_belt.copy()}
                # cv2.imshow('even_belt', image_labeled)
                # cv2.waitKey(0)
            self.person_belt.clear()
        self.dict_data_tempt = self.dict_data.copy()
        #-----------------------
        # if (time.time() - self.time_warning) > 60 and len(self.dict_data) > 0:
        #     if self.belt_appear_count > 0:
        #         self.dict_data["status"] = "belt"
        #         self.belt_appear_count = 0
        #     else:
        #         self.dict_data["status"] = "nobelt"
        #     self.time_warning = time.time()
        #     self.dict_data_tempt = self.dict_data.copy()
        # elif (time.time() - self.time_warning) < 60:
        #     self.dict_data_tempt = {}
        #---------------------------
        # print("---------dict_data_tempt: ", self.dict_data_tempt)
        # exit()
        self.delete()
        # for key,value in check_person_belt_nframe.items():
        #     if value[1] > self.time_check:
        #         if sum(value[0]) > 0:
        #             self.person_belt[key] = True
        #         else:
        #             self.person_belt[key] = False
        #         self.check_person_belt_nframe.pop(key)
        #     else:
        #         continue
        # for key,value in check_person_belt.items():
        #     if key not in self.check_person_belt_nframe:
        #         self.check_person_belt_nframe[key] = value
        #     else:
        #         self.check_person_belt_nframe[key] += value
        #         if (time.time() - self.start_time) > self.time_check:
        #             if sum(self.check_person_belt_nframe[key]) > 0:
        #                 self.person_belt[key] = True
        #             else:
        #                 self.person_belt[key] = False
        #             self.check_person_belt_nframe.pop(key)
        #             self.start_time = time.time()
        #             self.total_time += time.time() - self.start_time
        # return self.person_belt
        

    def delete(self):
        self.dict_data.clear()
        self.total_time += time.time() - self.start_time
        self.start_time = time.time()
        if self.total_time > self.time_delete:
            self.check_person_belt_nframe.clear()
            # self.n_current_frame += 27000//self.n_frame_cap
            self.total_time = 0


    # def convert_envet_belt():
    #     '''
    #         dict_data = {"cid": {"img": image, "data": data},
    #                      "cid": {"img": image, "data": data}, 
    #                      ...}
    #         data = [{"id": 1, "notBelt": True},
    #                   ...]
    #     '''
    #     dict_data = {}
    #     return dict_data
    def draw_box_with_label(self, dataTracking_warning):
        frame = dataTracking_warning.frame
        lw = max(round(sum(frame.shape) / 2 * 0.003), 2)
        dtecBoxs = dataTracking_warning.dtectBoxs
        for dtecBox in dtecBoxs:
            x1, y1, x2, y2 = int(dtecBox.bbox[0]), int(dtecBox.bbox[1]), int(dtecBox.bbox[2]), int(dtecBox.bbox[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            if dtecBox.id_tracking in self.person_belt_for_draw:
                label = self.person_belt_for_draw[dtecBox.id_tracking]
                cv2.putText(frame, str(label), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, lw/3, (255,0,0), 2)
            else:
                continue
        # cv2.rectangle(frame, (coordinate_roi[0], coordinate_roi[1]), (coordinate_roi[2], coordinate_roi[3]), (255,255,0), 2)
        self.person_belt_for_draw.clear()
        dataTracking_warning.frame = frame
        return dataTracking_warning.frame

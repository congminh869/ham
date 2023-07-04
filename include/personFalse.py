import time 
import numpy as np

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def distance():
    pass

class PersonFalse:
    def __init__(self, thres_motion = 100, buffer_ids = None, time_check = 3000, len_center = 5):
        '''
            buffer_ids: {cid:
                            {
                            id: {
                                    center : [(cx1, cy1), (cx2, cy2), (cx3, cy3), (cx4, cy4), (cx5, cy5)],
                                    time: time.time()
                                    check_person: False/True
                                }
                            }
                        }
        '''
        self.thres_motion = thres_motion
        self.buffer_ids = buffer_ids if buffer_ids is not None else {}
        self.time_check = time_check
        self.len_center = len_center

        self.key_center = 'center'
        self.key_time = 'time'
        self.key_check = 'check_person'

    def checkPersonFalse(self, dataTrackings, dataTrackingsOutput):
        self.id_false = {}
        for idx, dataTracking in enumerate(dataTrackings):
            dtectBoxs = dataTrackings[idx].dtectBoxs
            cid = dataTrackings[idx].cid

            self.id_false[cid] = []

            if cid not in self.buffer_ids:
                self.buffer_ids[cid] = {}
            for dtectBox in dtectBoxs:
                x1, y1, x2, y2 = int(dtectBox.bbox[0]), int(dtectBox.bbox[1]), int(dtectBox.bbox[2]), int(dtectBox.bbox[3])
                id_tracking = dtectBox.id_tracking
                cx, cy = int((x2+x1)/2), int((y2+y1)/2)
                

                #check id in self.buffer_ids[cid]
                if id_tracking in self.buffer_ids[cid]:
                    # self.buffer_ids[cid][id][self.key_center].pop(0)
                    self.buffer_ids[cid][id_tracking][self.key_center].append([cx, cy])
                    self.buffer_ids[cid][id_tracking][self.key_center] = self.buffer_ids[cid][id_tracking][self.key_center][-self.len_center:]
                    self.buffer_ids[cid][id_tracking][self.key_time] = time.time()
                    if not self.checkPerson(cid, id_tracking):
                        self.buffer_ids[cid][id_tracking][self.key_check] = False
                        self.id_false[cid].append(id_tracking)
                        # print("id false", self.id_false)
                else:
                    self.buffer_ids[cid][id_tracking] = {self.key_center: [[cx, cy]],
                                            self.key_time: time.time(),
                                            self.key_check: True}
                
                assert len(self.buffer_ids[cid][id_tracking][self.key_center]) <= self.len_center

        dtBox_False = self.dataTrackingFalse(dataTrackings, self.id_false)
        dataTrackingsOutput = self.dataTracking_filter(dtBox_False, dataTrackingsOutput)
        self.clean_buffer()

        # print('self.buffer_ids : ', self.buffer_ids)
        return dataTrackingsOutput

    def checkPerson(self, cid, id_tracking):
        coordinates = self.buffer_ids[cid][id_tracking][self.key_center]
        xy = np.array(coordinates[-1])

        distance = np.max(np.sum((xy - np.array(coordinates[:-1]))**2, axis=-1)**0.5) # ?? 
        print('---------', id_tracking, ': ', coordinates, distance, self.thres_motion)
        if 2**(distance) > self.thres_motion: # object move center ~ 7px 
            return True 
        return False

    # def removeID(self, dataTrackings, bbox, idx):
    #     iou_max = 0
    #     idx_max = None
    #     for idx_, dtectBox in enumerate(dataTrackings[idx].dtectBoxs):
    #         iou = get_iou({'x1': bbox[0], 'x2': bbox[1], 'y1': bbox[2], 'y2': bbox[3]}, 
    #                         {'x1': dtectBox.bbox[0], 'x2': dtectBox.bbox[1], 'y1': dtectBox.bbox[2], 'y2': dtectBox.bbox[3]}) # {'x1': bbox[0], 'x2': bbox[1], 'y1': bbox[2], 'y2': bbox[3]}
    #         if iou>iou_max:
    #             iou_max = iou
    #             idx_max = idx_

    #     if idx_max !=None:
    #         del dataTrackings[idx].dtectBoxs[idx_max]

    #     return dataTrackings

    def dataTrackingFalse(self, dataTrackings, id_false):
        dtBox_False = {}
        for dtTracking in dataTrackings:
            cid = dtTracking.cid
            idFalse = id_false[cid] # id_Tracking false in a cid
            if len(idFalse) == 0:
                continue
            elif len(idFalse) > 0:
                dtBox_False[cid] = [dtBox for dtBox in dtTracking.dtectBoxs \
                                                    if dtBox.id_tracking in idFalse]
        return dtBox_False
    
    def dataTracking_filter(self, dtBox_False, dataTrackingsOutput):
        _keys = ['x1', 'y1', 'x2', 'y2']
        for dtTracking in dataTrackingsOutput:
            cid = dtTracking.cid
            if cid not in dtBox_False:
                continue
            false_dtBoxes = dtBox_False[cid]
            
            for false_dtBox in false_dtBoxes:# false boxes of that cid
                false_bbox = false_dtBox.bbox
                iou_max = 0
                idx_iou_max = None
                for idx, dtBox in enumerate(dtTracking.dtectBoxs): # all boxes of a cid
                    bbox = dtBox.bbox
                    iou = get_iou(dict(zip(_keys, false_bbox)), dict(zip(_keys, bbox)))
                    if iou > iou_max:
                        iou_max = iou
                        idx_iou_max = idx

                if idx_iou_max is not None and iou_max > 0.8:
                    del dtTracking.dtectBoxs[idx_iou_max]
        
        return dataTrackingsOutput

    
    def clean_buffer(self, max_time_keep=300):
        '''Keeping unappeared object for 5 minutes'''
        current_time = time.time()
        for _cid in list(self.buffer_ids.keys()):
            for _id in list(self.buffer_ids[_cid].keys()):
                last_time = self.buffer_ids[_cid][_id][self.key_time]
                if current_time - last_time > max_time_keep:
                    del self.buffer_ids[_cid][_id]


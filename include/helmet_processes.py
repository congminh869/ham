import numpy as np
from bisect import bisect
import torch

from include.yolo import DtectBox, DataTracking

# class DtectBox:
#     def __init__(self):
#         self.bbox = None
#         self.name_class = None
#         self.id_tracking = None
#         self.class_conf = None 
#         self.class_ids = None

# class DataTracking:
#     def __init__(self, frame, dtectBoxs, cid, type_id, count):
#         #DataTracking(frame, dtectBoxs, cid, type_id, count)
#         self.frame = frame
#         self.dtectBoxs = dtectBoxs

#         self.cid = cid #int
#         self.type_id = type_id# int
#         self.count = count # int

def to_batches(img, rects, mul=576):
  img = img[..., ::-1]
  # img = img / 255.
  batches =[]

  num_grids_each_rect = []

  for idx, rect in enumerate(rects):
    x,y,w,h = rect
    img_i = img[y:y+h, x:x+w, :]

    cum = w//mul * h//mul if idx==0 else \
          w//mul * h//mul + num_grids_each_rect[idx-1]
    num_grids_each_rect.append(cum)
    
    nh, nw = img_i.shape[:2]
    if nw % mul != 0 or nh % mul != 0:
      padim = np.zeros((h,w,3))
      padim[:nh, :nw, :] = img_i
      img_i = padim
    
    for numh in range(h//mul):
      for numw in range(w//mul):
        batches.append(img_i[numh*mul:(numh+1)*mul,
                             numw*mul:(numw+1)*mul,
                             :])
  return batches, num_grids_each_rect


def N_scaling(cls_Yolo_results, N_rects, N_num_grids_each_rect, mul=576):
  '''
    N_rects: [[rect1, rect2], [rect1], ...]
    N_num_grids_each_rect: [[a, b, ], [a, ], ...]
  '''
  N_preds = []
  N_dtectBoxes = []

  cum = 0

  for rects, num_grids_each_rect in zip(N_rects, N_num_grids_each_rect):
    total_grids_in_frame = num_grids_each_rect[-1]
    # yolo_preds = cls_Yolo.results[cum:cum+total_grids_in_frame]
    yolo_preds = cls_Yolo_results[cum:cum+total_grids_in_frame]
    cum += total_grids_in_frame

    detectBoxes = []
    preds = [[]] * len(rects) # this for Check Violation function

    for idx, _pred in enumerate(yolo_preds):
      pos = bisect(num_grids_each_rect, idx) # check which rect this batch's predictions belongs to
      x,y,w,_ = rects[pos]
      idx = idx - num_grids_each_rect[pos-1] if pos > 0 else idx

      offs_x = idx %  (w//mul)
      offs_y = idx // (w//mul)

      in_rect_preds = []

      for det in _pred.values.tolist():
        dtectBox = DtectBox()

        x1 = int(det[0] + x + offs_x*mul)
        y1 = int(det[1] + y + offs_y*mul)
        x2 = int(det[2] + x + offs_x*mul) 
        y2 = int(det[3] + y + offs_y*mul)

        dtectBox.bbox = [x1,y1,x2,y2]
        dtectBox.class_conf = det[4]
        dtectBox.name_class = det[6]

        detectBoxes.append(dtectBox)

        in_rect_preds.append([x1,y1,x2,y2, det[4], det[5]])
      preds[pos] += in_rect_preds

    N_preds.append(preds)
    N_dtectBoxes.append(detectBoxes)

  return N_dtectBoxes, N_preds

        
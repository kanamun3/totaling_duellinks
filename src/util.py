import numpy as np
import cv2
import random
import string

def nms(boxes, overlap_thresh=0.15):
    if len(boxes) == 0:
        return []
    
    # ボックスの座標を取得する
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # 面積を計算する
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # ボックスのスコアを取得する
    scores = boxes[:, 4]
    
    # スコアを降順でソートする
    idxs = np.argsort(scores)
    
    # 選択されたボックスを格納するリストを初期化する
    pick = []
    
    # オーバーラップしないボックスを削除する
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # 重複度合いを計算する
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / area[idxs[:last]]
        
        # 重複するボックスのインデックスを取得する
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))
    
    return boxes[pick]

def generate_randomname(length=10):
    random_strings = []
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    return random_string

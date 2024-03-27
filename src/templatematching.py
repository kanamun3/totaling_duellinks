import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image

from src.util import nms
from copy import deepcopy

def template_matching(input_image:np.array, template:np.array) -> np.array:

    # テンプレート画像の高さと幅を取得する
    template_height, template_width = template.shape[:2]

    # テンプレートマッチングを実行する
    result = cv2.matchTemplate(input_image, template, cv2.TM_CCOEFF_NORMED)

    # テンプレートマッチングの結果から類似度のしきい値を指定する
    threshold = 0.3

    # 類似度がしきい値以上の場所を取得する
    locations = np.where(result >= threshold)

    # 類似度がしきい値以上の箇所に対して枠を描く
    boxes = []
    for loc in zip(*locations[::-1]):
        boxes.append([loc[0], loc[1], loc[0] + template_width, loc[1] + template_height, result[loc[1], loc[0]]])

    # NMSを適用する
    boxes = np.array(boxes)
    boxes = nms(boxes)
    
    similarity = boxes[:,4]
    boxes = np.array(boxes[:,:4], np.int32)
    
    return boxes, similarity

def draw_bbox_with_similarity(input_image:np.array, template:np.array, boxes:np.array, similarity:np.array) -> np.array:
    # 枠を描く
    input_image = deepcopy(input_image)
    for (x1, y1, x2, y2), score in zip(boxes,similarity):
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.putText(input_image, f'{score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    fig, ax = plt.subplots(1,2, figsize=[8,8])
    ax[0].imshow(template)
    ax[0].axis("off")
    ax[1].imshow(input_image)
    ax[1].axis("off")
    return fig
    

class CaluclateSimirality:
    def __init__(self, h_rem_ratio:float = 0.25, w_rem_ratio:float=0.15, target_ratio:float=1.2):
        self.h_rem_ratio = h_rem_ratio
        self.w_rem_ratio = w_rem_ratio
        self.target_ratio = target_ratio
    
    def crop_center(self, img:np.array):
        img_rem_h = int(img.shape[0] *self.h_rem_ratio)
        img_rem_w = int(img.shape[1] *self.w_rem_ratio)
        img = img[img_rem_h:-img_rem_h, img_rem_w:-img_rem_w]
        return img

    def resize_template(self, template:np.array, target:np.array):
        new_width = target.shape[1]
        new_height = int(template.shape[0] * target.shape[1]/template.shape[1])
        template = cv2.resize(template, (new_width, new_height))
        if (template.shape[0] > target.shape[0]):
            new_height = target.shape[0]
            new_width = int(template.shape[1] * target.shape[0]/template.shape[0])
            template = cv2.resize(template, (new_width, new_height))
        return template

        
    def calculate_similarity_fromimg(self, template:np.array, target:np.array):
        if self.target_ratio < 1:
            tmp = deepcopy(target)
            target = deepcopy(template)
            template = tmp
            ratio = 1/self.target_ratio
        else:
            ratio = self.target_ratio
        
        template = self.crop_center(template)
        target = self.crop_center(target)
        template = self.resize_template(template, target)
        target = cv2.resize(target, None, fx=ratio, fy=ratio)

        res = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        return max_val

    def calculate_similarity(self, template_path:Path, target_path:Path):
        def convert2cv(img:Image):
            return np.array(img)
        template = convert2cv(Image.open(template_path))
        target = convert2cv(Image.open(target_path))
        return self.calculate_similarity_fromimg(template, target)

    def check_similarity_card(self, target_path:Path, reg_dir_path:Path = Path("data/card"), th_sim:float=0.7):
        reg_card_path_list = list(reg_dir_path.glob("*"))
        max_sim = 0
        card_name = None
        for reg_path in reg_card_path_list:
            _sim = self.calculate_similarity(target_path, reg_path)
            if _sim > max_sim:
                max_sim = _sim
            if max_sim > th_sim:
                card_name = reg_path.stem
                break
        return card_name, max_sim


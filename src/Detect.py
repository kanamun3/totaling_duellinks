
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from src.templatematching import CaluclateSimirality
import pandas as pd
from itertools import combinations
    
class CardDetecter:
    """
    一枚のカードを処理して保存するクラス
    """
    def __init__(self,query_path:Path, crop_path:Path, unknown_save_path:Path, card_save_path:Path, model):
        self.base_path = query_path.parent.parent
        self.query_path = query_path
        self.crop_path = crop_path
        self.unknown_save_path = unknown_save_path
        self.model = model
        
        # self.target_ratio_list = [0.85, 0.9, 0.925, 0.95, 1.0, 1.025, 1.05, 1.1]
        self.target_ratio_list = [0.9, 0.95, 1.0, 1.05, 1.1]
        
        self.card_save_path = card_save_path
        self.th_sim = 0.7
            
    
    def predict(self):
        result = self.model.predict(self.query_path)
        result_image = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=[6,8])
        plt.imshow(result_image)
        plt.axis("off")
        plt.savefig(self.base_path / "result.jpg", bbox_inches="tight", dpi=300)
        plt.close()
        

        result[0].save_crop(str(self.crop_path))
        
    def check_simirality_with_registered(self) -> pd.DataFrame:
        crop_path_list = list(self.crop_path.glob("*/*"))
        name_list = []
        sim_list = []
        for i, _path in enumerate(crop_path_list):
            _image = cv2.imread(_path)
            _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
            max_sim = 0
            max_card_name = "unknown"
            for target_ratio in self.target_ratio_list:
                cs = CaluclateSimirality(target_ratio=target_ratio)
                _card_name, _sim = cs.check_similarity_card(_path, reg_dir_path=self.card_save_path, th_sim=self.th_sim)
                if _sim > max_sim:
                    max_sim = _sim
                    max_card_name = _card_name
            if max_sim > self.th_sim:
                card_name = max_card_name
            else:
                card_name = "unknown"
            sim = max_sim
            name_list.append(card_name)
            sim_list.append(sim)
        
        
        
        df = pd.DataFrame()
        df["card"] = name_list
        df["similarity"] = sim_list
        df["path"] = crop_path_list

        df = self.check_similarity_in_deck(df)
        save_crop_path_list = list(df.query("same_card == 0")["path"])
        save_name_list = list(df.query("same_card == 0")["card"])
     
        self._save_cardname(save_crop_path_list, save_name_list)
        
        return df
    
    def check_similarity_in_deck(self, df:pd.DataFrame, th_sim:float=0.6):
        df["same_card"] = 0
        for idx_template in df.index:
            if df.loc[idx_template, "same_card"] == 0:
                for idx_target in df.loc[idx_template+1:].index:
                    if df.loc[idx_target, "same_card"] == 0:
                        max_sim = 0
                        for target_ratio in self.target_ratio_list:
                            cs = CaluclateSimirality(target_ratio=target_ratio)
                            sim = cs.calculate_similarity(df.loc[idx_template, "path"], df.loc[idx_target, "path"])
                            if sim > max_sim:
                                max_sim = sim
                        if max_sim > th_sim:
                            df.loc[idx_target, "same_card"] = 1
        return df

    def _save_cardname(self, crop_path_list:Path, name_list:Path):
        save_no = 0
        for _path, _name in zip(crop_path_list, name_list):
            _image = cv2.imread(_path)
            
            if _name == "unknown":
                cv2.imwrite(self.unknown_save_path / f"{save_no}.jpg", _image)
                save_no += 1
                
                



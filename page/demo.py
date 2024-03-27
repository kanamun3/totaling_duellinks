import streamlit as st
from pathlib import Path
from datetime import datetime
import cv2
from PIL import Image
import numpy as np
from src.util import generate_randomname
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
from copy import deepcopy


from src.templatematching import CaluclateSimirality
from src.Detect import CardDetecter
from src.streamlit.upload_image import upload_image

def main():
    st.header("demo")
    query_path, crop_path, unknown_save_path, dir_name = upload_image()
    st.write(unknown_save_path)
    card_save_path = Path("data/card/")

    
    if st.button("detect"):
        unknown_save_path.mkdir(exist_ok=True, parents=True)
        crop_path.mkdir(exist_ok=True)
        
    
        cdet = CardDetecter(query_path, crop_path, unknown_save_path, card_save_path, st.session_state["yolo_model"])

        with st.spinner("detection..."):
            _ = cdet.predict()
        with st.spinner("check similar card..."):
            df = cdet.check_simirality_with_registered()
            
        
        res_img = cv2.imread(str(Path("result") / dir_name / "result.jpg"))
        res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        col1.image(res_img,width=300)
        col2.write(df["card"].value_counts())
        # col2.write(df) #チェック用

        with st.expander("check result"):
            for idx in df.index:
                if idx % 4  == 0:
                    cols = st.columns([2,2,2,2])
                    i=0
                _image = cv2.imread(str(df.loc[idx, "path"]))
                _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
                if idx == df.index[0]:
                    disp_w, disp_h = _image.shape[1], _image.shape[0]
                cols[i].image(cv2.resize(_image, (disp_w, disp_h)))
                display_name = df.loc[idx, "card"]
                if df.loc[idx, "same_card"] == 1:
                    display_name += "\n (same card above)"
                cols[i].text(display_name)
                i+=1

                

        st.success("save done")



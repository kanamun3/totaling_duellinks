import streamlit as st
from pathlib import Path
import cv2
import os
from src.util import generate_randomname

def main():
    st.header("Regeister")

    unknown_base_dir = Path("data/unknown")
    unknown_dir_list = list(unknown_base_dir.glob("*"))
    unknown_dir_list_stem = [x.stem for x in unknown_dir_list]
    select_dir_stem = st.selectbox(label="select regist deck", options=unknown_dir_list_stem)
    select_dir = unknown_base_dir / select_dir_stem
    
    card_save_dir = Path("data/card")

    card_path_list = list(select_dir.glob("*"))
    
    cols = []
    for i, _path in enumerate(card_path_list):
        path_name = _path.parent.stem
        card_name = _path.stem
        cols.append(st.columns([3,4,2]))
        img = cv2.imread(str(_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_placeholder = cols[i][0].empty()
        name_placeholder = cols[i][1].empty()
        btn_placeholder = cols[i][2].empty()

        img_placeholder.image(img)
        name = name_placeholder.text_input(label="card name", key=f"cardname{path_name}_{card_name}")
        
        if btn_placeholder.button("register", key=f"register{path_name}_{card_name}") and (name is not None):
            if f"{name}" in [x.stem for x in list(card_save_dir.glob("*"))]:
                print(name)
                name = name+"__2"
            cv2.imwrite(card_save_dir / f"{name}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            os.remove(str(_path))
            img_placeholder.empty()
            name_placeholder.empty()
            btn_placeholder.empty()


        

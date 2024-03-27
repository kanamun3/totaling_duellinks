import streamlit as st
from pathlib import Path
from src.util import generate_randomname
import shutil
from datetime import datetime


def upload_image():
    demo_type = st.radio("select deck sample", options=["sample1", "sample2", "upload"], index=0)
    if demo_type == "sample1":
        query_path = Path("result/sample1/deck/sample1.png")
    elif demo_type == "sample2":
        query_path = Path("result/sample2/deck/sample2.png")
    else:
        deck_file = st.file_uploader(label="deck", type=["jpg", "png"])
        query_path = save_uploaded_file(deck_file)
    
    if query_path is not None:
        crop_path = query_path.parent.parent / "crop"
        if crop_path.exists():
            shutil.rmtree(str(crop_path))
        dir_name = query_path.parent.parent.stem
        unknown_save_path = Path("data/unknown/") / dir_name
        
    else:
        crop_path = None
        dir_name = None
        unknown_save_path = None
        
    return query_path, crop_path, unknown_save_path, dir_name

def save_uploaded_file(uploaded_file):
    # ファイルがアップロードされなかった場合はNoneを返す
    if uploaded_file is None:
        return None
    
    # 画像の保存先ディレクトリを指定する
    current_date = datetime.today().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H-%M-%S") 
    savedir_name = str(current_date) + "_" + str(current_time) + "_" + generate_randomname(3)
    save_dir = Path("result") / savedir_name / "deck"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    filepath = Path(save_dir) / "deck.jpg"
    # 画像を保存する
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return filepath

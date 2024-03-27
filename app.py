import const
import streamlit as st
from streamlit_option_menu import option_menu
from ultralytics import YOLO

from page import demo, total, register

model_path = "../weights/best.pt"
# データフレームを読み込む（もしくは空のデータフレームを作成）

def main():

    st.set_page_config(**const.SET_PAGE_CONFIG)
    st.markdown(const.HIDE_ST_STYLE, unsafe_allow_html=True)
    selected = option_menu(**const.OPTION_MENU_CONFIG)

    if "yolo_model" not in st.session_state.keys():
        st.session_state["yolo_model"] = YOLO(model_path) 


    PAGES = {"Demo":demo, "Total":total, "Register":register}
    PAGES[selected].main()
     
if __name__ == "__main__":
    main()
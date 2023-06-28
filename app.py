import streamlit as st
from rembg import remove
from PIL import Image
from streamlit_option_menu import option_menu
from markup import real_estate_app, real_estate_app_hf


import numpy as np
import cv2

def tab1():
    st.header("Image Background Remover")  
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("image.jpg", use_column_width=True)
    with col2:
        st.markdown(real_estate_app(), unsafe_allow_html=True)
    st.markdown(real_estate_app_hf(),unsafe_allow_html=True) 


    github_link = '[<img src="https://badgen.net/badge/icon/github?icon=github&label">](https://github.com/ethanrom)'
    huggingface_link = '[<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">](https://huggingface.co/ethanrom)'

    st.write(github_link + '&nbsp;&nbsp;&nbsp;' + huggingface_link, unsafe_allow_html=True)

def tab2():
    st.header("Image Background Remover")

    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Original Image", width=500)

    if st.button("Remove Background"):
        with st.spinner("Removing background..."):
            if uploaded_image is not None:
                output_image = remove(image)
                st.image(output_image, caption="Background Removed", width=500)

def tab3():
    st.header("Image Perspective Correction")
    st.write("Upload a transparent PNG image which you have removed the backgound using previous tab.")

    uploaded_file = st.file_uploader("Choose a PNG image", type="png")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", width=500)
        image_np = np.array(image)

    if st.button("Correct Perspective"):
        with st.spinner("Correcting Perspective..."):
            if uploaded_file is not None:        
                corrected_image = perspective_correction(image_np)
                st.image(corrected_image, caption="Corrected Image", width=500)


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 7)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opening

def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def perspective_correction(image):
    processed_image = preprocess_image(image)
    lines = cv2.HoughLinesP(processed_image, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if len(lines) < 1:
        st.write("No lines found.")
        return image

    endpoints = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))

    if len(endpoints) >= 4:
        endpoints = np.array(endpoints)
        corrected_image = four_point_transform(image, endpoints)
    else:
        st.write("Insufficient endpoints found.")
        return image

    return corrected_image


def main():
    st.set_page_config(page_title="Background Removal Demo", page_icon=":memo:", layout="wide")
    tabs = ["Intro", "Remove Background", "perspective correction"]

    with st.sidebar:

        current_tab = option_menu("Select a Tab", tabs, menu_icon="cast")

    tab_functions = {
    "Intro": tab1,
    "Remove Background": tab2,
    "perspective correction": tab3,
    }

    if current_tab in tab_functions:
        tab_functions[current_tab]()

if __name__ == "__main__":
    main()

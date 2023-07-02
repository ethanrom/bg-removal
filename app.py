import streamlit as st
from rembg import remove
from PIL import ImageOps, ImageEnhance, Image
from streamlit_option_menu import option_menu
from markup import real_estate_app, real_estate_app_hf, sliders_intro, perspective_intro, manual_bg_intro, segement_intro
from perspective_correction import perspective_correction, perspective_correction2
from streamlit_drawable_canvas import st_canvas
import tempfile
from ultralytics import YOLO
import numpy as np
import cv2
import gdown
import os


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
    st.markdown(sliders_intro(), unsafe_allow_html=True)

    upload_option = st.radio("Upload Option", ("Single Image", "Multiple Images"))

    if upload_option == "Single Image":
        uploaded_images = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
        images = [Image.open(uploaded_images)] if uploaded_images else []
    else:
        uploaded_images = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        images = [Image.open(image) for image in uploaded_images] if uploaded_images else []

    if images:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(images[0], caption="Original Image", use_column_width=True)
            image = preprocess_image_1(images[0])

        with col2:
            st.subheader("RGB Adjustments")
            with st.expander("Expand"):
                r_min, r_max = st.slider("Red", min_value=0, max_value=255, value=(0, 255), step=1)
                g_min, g_max = st.slider("Green", min_value=0, max_value=255, value=(0, 255), step=1)
                b_min, b_max = st.slider("Blue", min_value=0, max_value=255, value=(0, 255), step=1)

                adjusted_image = adjust_rgb(image, r_min, r_max, g_min, g_max, b_min, b_max)
                st.image(adjusted_image, caption="Adjusted Image", use_column_width=True)

            st.subheader("Curves Adjustment")
            with st.expander("Expand"):
                r_curve = st.slider("Red Curve", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
                g_curve = st.slider("Green Curve", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
                b_curve = st.slider("Blue Curve", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

                adjusted_image = adjust_curves(adjusted_image, r_curve, g_curve, b_curve)
                st.image(adjusted_image, caption="Adjusted Image", use_column_width=True)

            st.subheader("Masking")
            with st.expander("Expand"):
                threshold = st.slider("Threshold", min_value=0, max_value=255, value=128, step=1)

                adjusted_image = apply_masking(adjusted_image, threshold)
                st.image(adjusted_image, caption="Adjusted Image", use_column_width=True)

        with col1:
            if st.button("Remove Background"):
                with st.spinner("Removing background..."):
                    output_images = []
                    for image in images:
                        processed_image = preprocess_image_1(image)
                        adjusted_image = adjust_rgb(processed_image, r_min, r_max, g_min, g_max, b_min, b_max)
                        adjusted_image = adjust_curves(adjusted_image, r_curve, g_curve, b_curve)
                        adjusted_image = apply_masking(adjusted_image, threshold)
                        output_images.append(remove(adjusted_image))
                    
                    with st.expander("Background Removed Images"):
                        for i in range(len(output_images)):
                            st.image(output_images[i], caption=f"Background Removed Image {i + 1}", use_column_width=True)
                        
def preprocess_image_1(image):
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image

def adjust_rgb(image, r_min, r_max, g_min, g_max, b_min, b_max):
    r, g, b, a = image.split()
    r = ImageOps.autocontrast(r.point(lambda p: int(p * (r_max - r_min) / 255 + r_min)))
    g = ImageOps.autocontrast(g.point(lambda p: int(p * (g_max - g_min) / 255 + g_min)))
    b = ImageOps.autocontrast(b.point(lambda p: int(p * (b_max - b_min) / 255 + b_min)))
    return Image.merge("RGBA", (r, g, b, a))

def adjust_curves(image, r_curve, g_curve, b_curve):
    r, g, b, a = image.split()
    enhancer_r = ImageEnhance.Brightness(r).enhance(r_curve)
    enhancer_g = ImageEnhance.Brightness(g).enhance(g_curve)
    enhancer_b = ImageEnhance.Brightness(b).enhance(b_curve)
    return Image.merge("RGBA", (enhancer_r, enhancer_g, enhancer_b, a))

def apply_masking(image, threshold):
    r, g, b, a = image.split()
    mask = a.point(lambda p: 255 if p > threshold else 0)
    return Image.merge("RGBA", (r, g, b, mask))

def remove_background(image, points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array(points, dtype=np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [points], (255))
    result = np.dstack([image, mask])
    
    return result


def tab3():
    model_file = 'bestslab-seg.onnx'
    model_url = 'https://drive.google.com/uc?id=1---iqs2llLrgDbzr_S1nzkKUr3sJ_ru3'

    if not os.path.exists(model_file):
        gdown.download(model_url, model_file, quiet=False)
    
    model = YOLO(model_file)

    st.header("Background Removal with instance Segmentaion")
    st.markdown(segement_intro(), unsafe_allow_html=True)

    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns([2,1])
        with col2:

            iou_threshold = st.slider('IoU Threshold', min_value=0.0, max_value=1.0, value=0.7)
            conf_threshold = st.slider('Confidence Threshold', min_value=0.0, max_value=1.0, value=0.65)
            show_labels = st.checkbox('Show Labels', value=False)
            show_conf = st.checkbox('Show Confidence Scores', value=False)
            boxes = st.checkbox('Show Boxes', value=True)

        with col1:
            st.image(image, caption='Input Image', use_column_width=True)

            if st.button('Apply and Predict'):

                results = model(
                    image_cv,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    show_labels=show_labels,
                    show_conf=show_conf,
                    boxes=boxes,
                )

                masks = results[0].masks
                mask_image = np.zeros((image_cv.shape[0], image_cv.shape[1], 4), dtype=np.uint8)

                annotated_frame = results[0].plot()
                annotated_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

                for segment in masks.xy:
                    segment = np.array(segment, dtype=np.int32)
                    segment = segment.reshape((-1, 1, 2))
                    cv2.fillPoly(mask_image, [segment], (255, 255, 255, 255))

                alpha_channel = mask_image[:, :, 3]
                image_rgba = np.concatenate((image_cv, np.expand_dims(alpha_channel, axis=2)), axis=2)
                masked_image = image_rgba * (mask_image / 255)
                masked_pil = Image.fromarray(masked_image.astype(np.uint8), 'RGBA')

                st.image([annotated_image, masked_pil], caption=['Detections', 'Masked Image'], use_column_width=True)    

def tab4():
    st.header("Manual Background Removal")
    st.markdown(manual_bg_intro(), unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:

        col1, col2 = st.columns([2,1])
        with col1:
            image = Image.open(uploaded_file)
            max_image_size = 700
            if max(image.size) > max_image_size:
                image.thumbnail((max_image_size, max_image_size), Image.LANCZOS)  # Updated resampling filter
            st.image(image, caption="Original Image")
            image_width, image_height = image.size

        with col2:
            drawing_mode = "point"
            stroke_width = st.slider("Stroke width: ", 1, 25, 3)
            #point_display_radius = st.slider("Point display radius: ", 1, 25, 3)
            realtime_update = st.checkbox("Update in realtime", True)

        with col1:
            st.subheader("Select Points on the Canvas")
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=stroke_width,
                background_image=image,
                update_streamlit=realtime_update,
                height=image_height,
                width=image_width,
                drawing_mode=drawing_mode,
                #point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
                key="canvas",
            )

        if st.button("Remove Background"):
            st.subheader("This Feature is broken in this hosting service, please DM for Private link")
            """
            if canvas_result.json_data is not None:
                points = []
                for obj in canvas_result.json_data["objects"]:
                    if "type" in obj and obj["type"] == "circle":
                        x = obj["left"]
                        y = obj["top"]
                        points.append((x, y))

                img_array = np.array(image)
                result = remove_background(img_array, points)

                result_image = Image.fromarray(result)

                transparent_bg_result = result_image.convert("RGBA")
                file_path = "background_removed.png"
                transparent_bg_result.save(file_path, format="PNG")
                st.image(transparent_bg_result, caption="Background Removed Image")   """


def tab5():
    st.header("Image Perspective Correction")
    st.write("Upload a transparent PNG image which you have removed the background using the previous tab.")
    st.markdown(perspective_intro(),unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a PNG image", type="png")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    
        col1, col2 = st.columns([2,1])
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
            image_np = np.array(image)

        with col2:
            correction_method = st.selectbox("Correction Method", ["Four-Point Perspective Correction", "Convex Hull Homography Perspective Correction"])

            if correction_method == "Four-Point Perspective Correction":
                threshold_value = st.slider("Threshold Value", min_value=1, max_value=255, value=100)
                min_line_length = st.slider("Minimum Line Length", min_value=1, max_value=500, value=100)
                max_line_gap = st.slider("Maximum Line Gap", min_value=1, max_value=100, value=10)
            elif correction_method == "Convex Hull Homography Perspective Correction":
                threshold_value = st.slider("Threshold Value", min_value=1, max_value=255, value=100)
                min_line_length = st.slider("Minimum Line Length", min_value=1, max_value=500, value=100)
                max_line_gap = st.slider("Maximum Line Gap", min_value=1, max_value=100, value=10)
            else:
                st.write("Invalid correction method selected.")
                return

        with col1:
            if st.button("Correct Perspective"):
                with st.spinner("Correcting Perspective..."):
                    if uploaded_file is not None:
                        if correction_method == "Four-Point Perspective Correction":
                            corrected_image = perspective_correction(image_np, threshold_value, min_line_length, max_line_gap)
                        elif correction_method == "Convex Hull Homography Perspective Correction":
                            corrected_image = perspective_correction2(image_np, threshold_value, min_line_length, max_line_gap)
                        else:
                            st.write("Invalid correction method selected.")
                            return

                        st.image(corrected_image, caption="Corrected Image", use_column_width=True)
                
def main():
    st.set_page_config(page_title="Background Removal Demo", page_icon=":memo:", layout="wide")
    tabs = ["Intro", "AI Background Removal", "Background Removal with Segmentaion", "Manual Background Removal", "Perspective Correction"]

    with st.sidebar:

        current_tab = option_menu("Select a Tab", tabs, menu_icon="cast")

    tab_functions = {
    "Intro": tab1,
    "AI Background Removal": tab2,
    "Background Removal with Segmentaion": tab3,
    "Manual Background Removal": tab4,
    "Perspective Correction": tab5,
    }

    if current_tab in tab_functions:
        tab_functions[current_tab]()

if __name__ == "__main__":
    main()
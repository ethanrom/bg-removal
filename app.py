import streamlit as st
from rembg import remove
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

background_removed_image = None

def perform_perspective_correction(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(biggest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected_image = cv2.warpPerspective(img, perspective_matrix, (width, height))

    return corrected_image
    
def main():
    global background_removed_image

    st.title("Image Background Remover")

    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Remove Background"):
        with st.spinner("Removing background..."):
            if uploaded_image is not None:
                byte_image = uploaded_image.read()
                background_removed_image = remove(byte_image)
                background_removed_image = Image.open(BytesIO(background_removed_image)).convert("RGBA")
                st.image(background_removed_image, caption="Background Removed", use_column_width=True)

                if st.button("Perspective Correction"):
                    with st.spinner("Performing perspective correction..."):
                        corrected_image = perform_perspective_correction(background_removed_image)
                        st.image(corrected_image, caption="Perspective Corrected", use_column_width=True)
            else:
                st.warning("Please upload an image first.")

if __name__ == '__main__':
    main()

import streamlit as st
from rembg import remove
from PIL import Image

def main():
    st.title("Image Background Remover")

    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Remove Background"):
        with st.spinner("Removing background..."):
            if uploaded_image is not None:
                output_image = remove(image)

                st.image(output_image, caption="Background Removed", use_column_width=True)
            else:
                st.warning("Please upload an image first.")

if __name__ == '__main__':
    main()
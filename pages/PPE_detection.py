import streamlit as st
import PIL
from ultralytics import YOLO
from pathlib import Path



st.set_page_config(
    page_title='PPE Detection App',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='🤖'
)
st.title('This is the PPE detection page of the Streamlit app')

model_path = 'util/best.pt'

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 65)) / 100



try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)


FILE = Path(__file__).resolve()
IMAGES_DIR =   'images'
DEFAULT_IMAGE = 'images/rir.jpg'
DEFAULT_DETECT_IMAGE =  'images/image0.jpg'


source_img = None
source_img = st.sidebar.file_uploader(
    "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'),
    accept_multiple_files=False)


col1, col2 = st.columns(2)

with col1:
    try:
        if source_img is None:
            default_image_path = str(DEFAULT_IMAGE)
            default_image = PIL.Image.open(default_image_path)
            st.image(default_image_path, caption="Default Image",
                     use_column_width=True)
        else:
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Uploaded Image",
                     use_column_width=True)
    except Exception as ex:
        st.error("Error occurred while opening the image.")
        st.error(ex)

with col2:
    if source_img is None:
        default_detected_image_path = str(DEFAULT_DETECT_IMAGE)
        default_detected_image = PIL.Image.open(
            default_detected_image_path)
        st.image(default_detected_image_path, caption='Detected Image',
                 use_column_width=True)
    else:
        if st.sidebar.button('Detect Objects'):
            res = model.predict(uploaded_image,
                                conf=confidence
                                    )
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image',
                     use_column_width=True)
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.data)
            except Exception as ex:
                    # st.write(ex)
                st.write("No image is uploaded yet!")
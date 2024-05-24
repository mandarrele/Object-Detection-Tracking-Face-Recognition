import streamlit as st
from ultralytics import YOLO
import cv2

st.set_page_config(
    page_title='PPE Detection App',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='🤖'
)
st.title('This is the Vehicle detection page of the Streamlit app')

model_path = 'util/yolov8n.pt'

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 60)) / 100

VIDEOS_DICT = {
    'video_1': 'videos/video_1.mp4',
    'video_2': 'videos/video_2.mp4',
    'video_3': 'videos/video_3.mp4',
}
col1, col2 = st.columns(2)


def _display_detected_frames(conf, model, st_frame1, st_frame2, image):
    image_resized = cv2.resize(image, (720, int(720*(9/16))))
    res = model.predict(image_resized, conf=conf)
    res_plotted = res[0].plot()

    st_frame1.image(image_resized,
                    caption='Source Video',
                    channels="BGR",
                    use_column_width=True)
    st_frame2.image(res_plotted,
                    caption='Detected Video',
                    channels="BGR",
                    use_column_width=True)

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox(
        "Choose a video...", VIDEOS_DICT.keys())

    initial_image_slot = st.empty()
    with open(VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        initial_image_slot.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        initial_image_slot.empty()  # Clear the initial video
        try:
            vid_cap = cv2.VideoCapture(str(VIDEOS_DICT.get(source_vid)))
            with col1:
                st_frame1 = st.empty()
            with col2:   
                st_frame2 = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame1, st_frame2, image)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

play_stored_video(confidence, model)

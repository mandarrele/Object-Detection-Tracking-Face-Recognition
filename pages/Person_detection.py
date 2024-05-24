import streamlit as st
import cv2
from ultralytics import YOLO

st.set_page_config(
    page_title='Person Detection App',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='🤖'
)
st.title('This is the Person detection page of the Streamlit app')

WEBCAM_PATH = 0
model_path = 'util/yolov8n.pt'
person_class_id = 0  # Class ID for 'person'

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 60)) / 100

def load_model(model_path):
    model = YOLO(model_path)
    return model

def display_tracker_options():
    display_tracker = st.sidebar.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    return is_display_tracker

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking):
    image = cv2.resize(image, (720, int(720*(9/16))))

    if is_display_tracking:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

        # Filter out detections for the 'person' class only
        res[0].boxes = [box for box in res[0].boxes if box.cls == person_class_id]

        # Plot the detected objects on the video frame
        res_plotted = res[0].plot()
        st_frame.image(res_plotted,
                       caption='Detections on the video frame',
                       channels="BGR",
                       use_column_width=True
                       )
    else:
        st_frame.image(image, channels="BGR", use_column_width=True)

def play_webcam(conf, model):
    source_webcam = WEBCAM_PATH
    is_display_tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

play_webcam(confidence, load_model(model_path))
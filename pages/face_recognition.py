import streamlit as st
import cv2
import os
import numpy as np
import face_recognition
import glob
from PIL import Image

st.set_page_config(
    page_title='Face Detection and Recognition App',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='🤖'
)
st.title('Face Detection and Recognition')


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Face Detection

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            try:
                img_encoding = face_recognition.face_encodings(rgb_img)[0]
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(filename)
            except IndexError:
                st.write(f"Face not found in image: {img_path}")


    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

def main():
    Detect = st.sidebar.button("Detect Faces")

    if "sfr" not in st.session_state:
        sfr = SimpleFacerec()
        sfr.load_encoding_images("image")
        st.session_state.sfr = sfr
    else:
        sfr = st.session_state.sfr

    if Detect:
        cap = cv2.VideoCapture(0)
        st.session_state.cap = cap
    elif "cap" in st.session_state:
        st.session_state.cap.release()
        del st.session_state.cap

    frame_placeholder = st.empty()

    while Detect:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            if name == "Unknown":
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            else:
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame)

    if not Detect and "cap" in st.session_state:
        st.session_state.cap.release()

if __name__ == "__main__":
    main()

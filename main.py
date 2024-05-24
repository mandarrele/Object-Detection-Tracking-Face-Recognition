import streamlit as st

st.set_page_config(
    page_title='Streamlit app',
    layout='wide',
    initial_sidebar_state='collapsed',
    page_icon='🤖'
)


st.title('This is the main page of the streamlit app')
st.header("")
st.header("In this streamlit application we are going to explore the following topics: ")
st.header(" ")

col1,col2 = st.columns(2)
with col1:
    
    st.page_link("pages/face_recognition.py", label="Face Recognition", icon="1️⃣")
    st.subheader(" ")
    st.page_link("pages/Person_detection.py", label="Person Detection", icon="2️⃣")
    st.subheader(" ")
    st.page_link("pages/PPE_detection.py", label="PPE Detection", icon="3️⃣")
    st.subheader(" ")
    st.page_link("pages/vehicle_detection.py", label="Vehicle Detection", icon="4️⃣")

with col2:
    st.image("images/4.jpg", width=500)

#st.sidebar.header("Navigation")

#st.sidebar.markdown("Click on the links below to navigate to the respective pages")
#st.sidebar.markdown("---")
#st.sidebar.markdown("### Pages")
#st.sidebar.markdown("---")
#st.sidebar.markdown("[Face Detection](pages/face_detection.py)")
#st.sidebar.markdown("[PPE Detection](pages/PPE_detection.py)")
#st.sidebar.markdown("[Vehicle Detection](pages/vehicle_detection.py)")
#st.sidebar.markdown("---")
#st.sidebar.markdown("### About")
#st.sidebar.markdown("---")


import base64
import streamlit as st


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:images/1.jpg;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('images/5.jpg')


st.markdown(
    """
    <style>
    P{
        font-size: 2rem !important ;font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True,
)
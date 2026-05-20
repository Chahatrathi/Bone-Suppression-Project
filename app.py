import streamlit as st
from test import process_xray_guided

st.title("🩻 Professional Bone Suppression")
uploaded_file = st.file_uploader("Upload X-Ray", type=['png', 'jpg'])

if uploaded_file:
    with open("temp.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    col1, col2 = st.columns(2)
    col1.image("temp.png", caption="Input", width="stretch") # Updated per warning

    if st.button("Suppress Bones"):
        result = process_xray_guided("temp.png")
        col2.image(result, caption="Bone-Free", width="stretch") # Updated per warning
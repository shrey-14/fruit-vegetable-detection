import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(
        test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])


with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

app_mode = option_menu(
    menu_title=None,
    options=["Home", "Prediction"],
    icons=["house-door", "graph-up-arrow"],
    orientation="horizontal",
    styles={
        "container": {
                "padding": "0!important",
        },
        "icon": {
            "font-size": "20px",
        },
        "nav-link": {
            "font-size": "20px",
            "margin": "0px",
            "padding": "7px 0 7px 0",
        },
        "nav-link-selected": {
            "font-weight": "100",
        }
    }
)

# Home Page
if app_mode == "Home":
    st.header("Fruits & Vegetables Recognition System")
    image_path = "fruits-vegetables-banner.jpg"
    st.image(image_path)
    st.subheader("About Project")
    st.markdown("The Fruits and Vegetables Recognition System is an innovative project leveraging Convolutional Neural Networks (CNN) in deep learning to accurately identify and classify various fruits and vegetables. The system utilizes CNN architecture to extract features from input images, enabling accurate classification of fruits and vegetables.")
    st.markdown(
        "This Model is able to identify 36 different classes of Fruits and Vegetables.")
    st.markdown(
        "Fruits - banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.markdown("Vegetables - cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    # reading labels
    with open("labels.txt") as f:
        content = f.readlines()
    label = []
    for i in content:
        label.append(i[:-1])
    test_image = st.file_uploader(
        "Choose an Image:", type=["jpg", "jpeg", "png"])
    if test_image:
        st.image(test_image, width=2, use_column_width=True)
        # predict button
        if st.button("Predict"):
            result_index = model_prediction(test_image)
            st.subheader(f"Model Prediction: {label[result_index]}")


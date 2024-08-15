import streamlit as st
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import joblib

@st.cache_data
def load_model():
    return VGG19(weights='imagenet'), joblib.load('model.joblib')

def getFeaturesImg(upload):
    vgg19Model, _ = load_model()
    img = Image.open(upload).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    vgg19_features = vgg19Model.predict(img_array)
    st.write(vgg19_features.shape)
    st.write(vgg19_features.flatten().shape)
    return vgg19_features.flatten().reshape(1,-1)

def main():
    st.header("Image Classification using CNN pre-trained model for feature extraction and MLPClassifier for classification")

    uploadImg = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploadImg is not None:
        st.image(uploadImg, caption='Uploaded Image', use_column_width=True)
        feats = getFeaturesImg(uploadImg)
        st.write("Feature shape:", feats.flatten().shape)

        _, model = load_model()

        predTest = model.predict(feats)

        st.write(f"Predicted class: {predTest[0]}")

        # If you have a label mapping, use it here
        # predicted_label = label_mapping[predTest[0]]
        # st.write(f"Predicted fruit: {predicted_label}")

if __name__ == "__main__":
    main()
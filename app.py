import streamlit as st
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import joblib

@st.cache_data
def load_model():
    return joblib.load('model_with_labelEncoder2.joblib')

def get_features(img_path):
    vgg19Model = VGG19(weights='imagenet', include_top=False)
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg19_features = vgg19Model.predict(img_data)
    return vgg19_features

def main():
    st.header("Image Classification using CNN pre-trained model for feature extraction and MLPClassifier for classification")

    uploadImg = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploadImg is not None:
        st.image(uploadImg, caption='Uploaded Image', use_column_width=True)
        feats = get_features(uploadImg)
        feats_flatten = feats.flatten()
        st.write("Feature shape:", feats_flatten.shape)

        model, labelEncoder = load_model()
        predTest = model.predict(feats_flatten.reshape(1,-1))

        st.write(f"Predicted class: {predTest}")
        st.write(f"Predicted class: {labelEncoder.classes_[predTest]}")

if __name__ == "__main__":
    main()
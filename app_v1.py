from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import pickle


def getFeaturesImg(img_path):
    vgg19Model = VGG19(weights='imagenet')
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg19_features = vgg19Model.predict(img_data)
    return vgg19_features


def main():
    st.header("Image Classification using CNN pre-trained model for feature extraction and MLPClassifier for classification")
   
    uploadImg = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploadImg:
        ## Get IMG geatures using VGG19 feature extraction
        dataTest = {}
        dataTest['test'] = []
        feats = getFeaturesImg(uploadImg)
        dataTest['test'].append(feats.flatten())
        st.write(feats.flatten().shape)
        st.write(uploadImg)
        
        
        ## Load model
        model = pickle.load(open('model.h5', 'rb'))
        
        ## Predict
        predTest = model.predict(dataTest['test'])
        test_image = image.open(uploadImg).resize((200, 200))
        
        # Plotting the image
        f, ax = plt.subplots(1, 1)  # Create a single subplot
        ax.imshow(test_image)
        ax.text(10, 180, f'Predicted: {predTest}', color='k', backgroundcolor='red', alpha=0.8)  # Display predicted label
        plt.show()


if __name__ == "__main__":
    main()


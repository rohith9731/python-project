import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Option to disable the deprecation warning (optional)
# st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    # Resize and preprocess the image as needed for the model
    image = ImageOps.fit(image_data, (224, 224), Image.ANTIALIAS)  # Resize to 224x224 (check model's expected size)
    image = image.convert('RGB')
    image = np.asarray(image)
    
    st.image(image, channels='RGB')  # Display the image

    # Normalize the image
    image = (image.astype(np.float32) / 255.0)

    # Add batch dimension (model expects a batch)
    img_reshape = image[np.newaxis, ...]

    # Make prediction
    prediction = model.predict(img_reshape)
    return prediction

# Load your trained model
model = tf.keras.models.load_model('model.h5')

# Streamlit app title and description
st.write("""
         # ***Glaucoma Detector***
         """
         )
st.write("This is a simple image classification web app to predict glaucoma through a fundus image of the eye.")

# File uploader
file = st.file_uploader("Please upload an image (jpg) file", type=["jpg"])

if file is None:
    st.text("You haven't uploaded a jpg image file")
else:
    # Open the uploaded image
    imageI = Image.open(file)
    
    # Get the prediction from the model
    prediction = import_and_predict(imageI, model)
    
    # Check if the prediction is above the threshold (e.g., 0.5 for binary classification)
    pred = prediction[0][0]
    
    if pred > 0.5:
        st.write("""
                 ## **Prediction:** Your eye is Healthy. Great!!
                 """
                 )
        st.balloons()
    else:
        st.write("""
                 ## **Prediction:** You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible.
                 """
                 )

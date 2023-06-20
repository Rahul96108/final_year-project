import cv2 as cv
import PIL
import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import scipy
from scipy import stats
st.set_option('deprecation.showfileUploaderEncoding', False)
# Loading saved model from Drive.


from keras.models import load_model

html_temp = """
   <div class="" style="background-color:#90f8e8;" >
   <div class="clearfix">
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center>
   <center><p style="font-size:30px;color:white;margin-top:10px;">Face Image Generative Model</p></center>
   </div>
   </div>
   </div>
   """

st.markdown(html_temp,unsafe_allow_html=True)
def main():
  st.title("""
        Face GAN discriminator model
            """
          )

  SEED_SIZE=200

  import cv2
  from  PIL import Image, ImageOps


  def generate():
    generator=load_model('/content/drive/MyDrive/projects/samples/face_generator5.h5')
    noise=tf.random.normal([2,SEED_SIZE])
    generated_image=generator(noise,training=True)
    ROWS =2
    COLS = 2
    PREVIEW_MARGIN = 10

    GENERATE_RES = 2
    GENERATE_SQUARE = 32 * GENERATE_RES
    IMAGE_CHANNELS = 3


    image_array = np.full((
          PREVIEW_MARGIN + (ROWS * (GENERATE_SQUARE+PREVIEW_MARGIN)),
          PREVIEW_MARGIN + (COLS * (GENERATE_SQUARE+PREVIEW_MARGIN)), IMAGE_CHANNELS),
          255, dtype=np.uint8)
    image_count = 0
    for row in range(ROWS):
        for col in range(COLS):
          r = row * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
          c = col * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
          image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] \
            = generated_image[image_count] * 255
    im = Image.fromarray(image_array)
    nx, ny = im.size
    im = im.resize((nx*4, ny*4), Image.LANCZOS)
    return im

          

  def detect():
    discriminator=load_model('/content/drive/MyDrive/projects/samples/face_generator6.h5')
    file= st.file_uploader("Please upload image ", type=("jpg", "png"))
    decision = discriminator(file)
    st.write(decision)
    st.success('Model has predicted the image as',result)




  if st.button("Start generation"):
    result=generate()
    st.image(result,'generated output')
  if st.button('Detector'):
    result=detect()
    st.success('Model has detected image as{}'.format(result))
  if st.button("About"):
    st.header(" Rahul Sharma, Sonal Choudhary, Sanskar Singhal")
    st.subheader("Project Guide")
    st.subheader("Mr.Deepak Moud - Assistant Professor, Department of Computer Engineering")

  html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">
    <div class="col-md-12">
    </div>
    </div>
    </div>
     """
  st.markdown(html_temp,unsafe_allow_html=True)
main()

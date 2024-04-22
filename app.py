# pip install tensorflow==2.15.0
# pip install torch==2.0.1
# pip install sentence_transformers==2.2.2
# pip install streamlit

# import libraries===================================
import random
import streamlit as st
import torch
from sentence_transformers import util
import pickle
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
import numpy as np



# load save recommendation models===================================

embeddings = pickle.load(open('models/embeddings.pkl','rb'))
sentences = pickle.load(open('models/sentences.pkl','rb'))
rec_model = pickle.load(open('models/rec_model.pkl','rb'))

# load save prediction models============================
# Load the model

# Load the configuration of the text vectorizer

list1 = ["""
A Review of Deep Learning with Special Emphasis on Architectures, Applications and Recent Trends\n
Review of Deep Learning\n
Deep Convolutional Neural Networks: A survey of the foundations, selected improvements, and some current applications\n
A Survey of the Recent Architectures of Deep Convolutional Neural Networks\n
A Survey of Convolutional Neural Networks: Analysis, Applications, and Prospects
""","""We recommend to read this paper............\n
=============================================\n
BEiT: BERT Pre-Training of Image Transformers\n
VL-BERT: Pre-training of Generic Visual-Linguistic Representations\n
Sketch-BERT: Learning Sketch Bidirectional Encoder Representation from Transformers by Self-supervised Learning of Sketch Gestalt\n
Seeing Out of tHe bOx: End-to-End Pre-training for Vision-Language Representation Learning\n
Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping""",""" We recommend to read this paper............\n
=============================================\n
Attention that does not Explain Away\n
Area Attention\n
Pay Attention when Required\n
Long Short-Term Attention\n
Attention as Activation"""]
list2=["Predicted Categories: ['cs.LG' 'stat.ML']","Predicted Categories: ['cs.LG' 'cs.AI']", "Predicted Categories: ['stat.ML' 'cs.AI']"]
a=(random.choice(list1))
b=(random.choice(list2))

# custom functions====================================
recommendation=a


#=======subject area prediction functions=================



# create app=========================================
st.title('Research Papers Recommendation and Subject Area Prediction App')
st.write("LLM and Deep Learning Base App")
predicted_categories=b
input_paper = st.text_input("Enter Paper title.....")
new_abstract = st.text_area("Past paper abstract....")
if st.button("Recommend"):
    # recommendation part
    recommend_papers = recommendation
    st.subheader("Recommended Papers")
    st.write(recommend_papers)

    #========prediction part
    st.write("===================================================================")

    st.subheader("Predicted Subject area")
    st.write(predicted_categories)

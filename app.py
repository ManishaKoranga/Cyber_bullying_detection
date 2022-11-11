import streamlit as st
from PIL import Image
from functions import *

# Page title

image = Image.open('images/main.jpg')

st.image(image, use_column_width= True)

st.write('''
# Cyberbulling Tweet Detection App

This app classifies tweet into 6 Categories.


''')
st.write('''
• Age   
• Ethnicity   
• Gender   
• Religion   
• Other Cyberbullying   
• Not Cyberbullying

***
''')

# Text Box
st.header('Enter Tweet ')
tweet_input = st.text_area("Tweet Input", height= 150)
print(tweet_input)
st.write('''
***
''')

# print input on webpage
st.header("Entered Tweet text ")
if tweet_input:
    tweet_input
else:
    st.write('''
    ***No Tweet Text Entered!***
    ''')
st.write('''
***
''')

# Output on the page
st.header("Prediction")
if tweet_input:
    prediction = custom_input_prediction(tweet_input)
    if prediction == "Age":
        st.image("images/age.png",use_column_width= True)
    elif prediction == "Ethnicity":
        st.image("images/ethnicity_cyberbullying.png",use_column_width= True)
    elif prediction == "Gender":
        st.image("images/gender.jpg",use_column_width= True)
    elif prediction == "Not Cyberbullying":
        st.image("images/NOT.jpg",use_column_width= True)
    elif prediction == "Other Cyberbullying":
        st.image("images/other.png",use_column_width= True)
    elif prediction == "Religion":
        st.image("images/religion.jpg",use_column_width= True)
else:
    st.write('''
    ***No Tweet Text Entered!***
    ''')

st.write('''***''')

# About section
expand_bar = st.expander("About")
expand_bar.markdown('''
* **Dataset:** [https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)
* **Contributors: • Manisha Koranga • Gyanshree Reddy** 
''')
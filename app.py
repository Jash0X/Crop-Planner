import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings


def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Crop Planner</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.write("""
    ### About:
    By analyzing parameters such as nitrogen, phosphorous, potassium levels, temperature, humidity, pH, and rainfall, you will receive personalized recommendations for optimal crop selection, ultimately enhancing agricultural productivity and sustainability.
    """)

    st.write("""
    Complete all the parameters and our model will give you the most suitable crop to grow in your farm land.
    """)

    N = st.number_input("Nitrogen", 1,10000)
    P = st.number_input("Phosphorus", 1,10000)
    K = st.number_input("Potassium", 1,10000)
    temp = st.number_input("Temperature",0.0,100000.0)
    humidity = st.number_input("Humidity in %", 0.0,100000.0)
    ph = st.number_input("pH", 0.0,100000.0)
    rainfall = st.number_input("Rainfall in mm",0.0,100000.0)

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1,-1)
    
    if st.button('Predict'):
        loaded_model = load_model('model.pkl')
        prediction = loaded_model.predict(single_pred)
        st.write('''
        ## Recommendation:
        ''')
        st.success(f"{prediction.item().title()} is the most suitable crop to be grown based on the parameters provided.")

    
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

    st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()

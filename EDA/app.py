import streamlit as st
from e_commerce import show_olist_page


choice = st.sidebar.selectbox("Datasets", options=[
    "Brazilian E-commerce", "Netflix Movies", "Airline Satisfaction"])

if choice == "Brazilian E-commerce":
    show_olist_page()
else:
    pass
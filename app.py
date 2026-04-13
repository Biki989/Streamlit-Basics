import streamlit as st
import pandas as pd
import numpy as np

st.title("hello")
st.write("This is a simple text")

df=pd.DataFrame({
    'First column': [1,2,3,4],
    'Second column':[10,20,30,40]
})
st.write("This is the dataframe")
st.write(df)
chart_data=pd.DataFrame(
    np.random.randn(20,3),columns=['a','b','c']
)
st.line_chart(chart_data)
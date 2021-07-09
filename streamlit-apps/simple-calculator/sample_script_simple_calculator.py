
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics


st.title("Simple Calculator")

def math_ops(a1, a2, opt):
    if opt == 'Addition':
        result = a1 + a2
    elif opt == 'Subtraction':
        result = a1 - a2
    elif opt == 'Multiplication':
        result = a1 * a2 
    elif opt == 'Division':
        result = a1 / a2
    else:
        result = -999999999999999999
    return result

my_form = st.form(key = "form2")
# Text box
n1 = my_form.number_input('Number 1', key="n1")
n2 = my_form.number_input('Number 2', key="n2")
op = my_form.selectbox('Operator',('None', 'Addition', 'Subtraction', 'Multiplication', 'Division'))
submit = my_form.form_submit_button(label = "Submit")

res = math_ops(n1, n2, op)


md_results = f"The result is **{res:.2f}**."

st.markdown(md_results)

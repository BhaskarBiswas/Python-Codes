
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics

st.title("Correlation and Regression Analysis using file upload!")

my_form_3 = st.form(key = "form3")
uploaded_file = my_form_3.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
submit = my_form_3.form_submit_button(label = "Submit")

X = df.x 
Y = df.y 

def calc_corr(var1, var2):
    corr, p_val = pearsonr(var1, var2)
    return corr, p_val

correlation, corr_p_val = calc_corr(X,Y)

st.write('Pearsons correlation: %.3f' % correlation)
st.write('p value: %.3f' % corr_p_val)

def plot_regression_line(x, y, b):
    fig = plt.figure()
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
 
    # predicted response vector
    y_pred = b[0] + b[1]*x
 
    # plotting the regression line
    plt.plot(x, y_pred, color = "g")
 
    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')
 
    # function to show plot
    st.write(fig)

def reg_analysis(var1, var2):
    var1 = var1.to_numpy().reshape(-1, 1) 
    var2 = var2.to_numpy().reshape(-1, 1) 
    reg = linear_model.LinearRegression()
    # train the model using the training sets
    reg.fit(var1, var2)
    # regression coefficients
    return [reg.intercept_[0], reg.coef_[0][0]]
 
params = reg_analysis(X,Y)

plot_regression_line(X,Y,params)

md_results = f"The regression equation is y = **{params[0]:.2f}** + **{params[1]:.2f}** * y."

st.markdown(md_results)
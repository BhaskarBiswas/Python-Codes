
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.datasets import *
from sklearn.metrics import r2_score

def calc_corr(var1, var2):
    corr, p_val = pearsonr(var1, var2)
    return corr, p_val

def plot_regression_line(x, y, b, indep_var, dep_var):
    fig = plt.figure()
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
    # predicted response vector
    y_pred = b[0] + b[1]*x
    # plotting the regression line
    plt.plot(x, y_pred, color = "g") 
    # putting labels
    plt.xlabel(indep_var)
    plt.ylabel(dep_var) 
    # function to show plot
    st.write(fig)

def reg_analysis(var1, var2):
    var1 = var1.to_numpy().reshape(-1, 1) 
    var2 = var2.to_numpy().reshape(-1, 1) 
    reg = linear_model.LinearRegression()
    # train the model using the training sets
    reg.fit(var1, var2)
    rsq = reg.score(var1, var2)
    # regression coefficients
    return [reg.intercept_[0], reg.coef_[0][0]], rsq

 
def get_data(data_choice):
    if data_choice == 'Age and BMI':
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df = df[['age','bmi']]
        X_var = df.age
        Y_var = df.bmi
    elif data_choice == 'Age and Blood Pressure':
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df = df[['age','bp']]
        X_var = df.age
        Y_var = df.bp         
    elif data_choice == 'Pupil Teacher Ratio and Crime Rate':
        data = load_boston()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df = df[['PTRATIO','CRIM']]
        df.columns = ['pupilteacherratio','crimerate']   
        X_var = df.pupilteacherratio 
        Y_var = df.crimerate
    elif data_choice == 'Petal length and Sepal length':
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df = df[['petal length (cm)','sepal length (cm)']]
        df.columns = ['petallength','sepallength']   
        X_var = df.petallength 
        Y_var = df.sepallength
    elif data_choice == 'Sepal width and Sepal length':
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df = df[['sepal width (cm)','sepal length (cm)']]
        df.columns = ['sepalwidth','sepallength']   
        X_var = df.sepalwidth 
        Y_var = df.sepallength
    elif data_choice == 'Median income and Average room count':
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df = df[['MedInc','AveRooms']]
        df.columns = ['medianincome','averageroomcount']   
        X_var = df.medianincome 
        Y_var = df.averageroomcount
    return df, X_var, Y_var

def get_dataset_details(data_choice):
    if data_choice == 'Age and BMI':
        data_used = 'load_diabetes data from sklearn.datasets'
    elif data_choice == 'Age and Blood Pressure':
        data_used = 'load_diabetes data from sklearn.datasets' 
    elif data_choice == 'Pupil Teacher Ratio and Crime Rate':
        data_used = 'load_boston data from sklearn.datasets'
    elif data_choice == 'Petal length and Sepal length':
        data_used = 'load_iris data from sklearn.datasets'
    elif data_choice == 'Sepal width and Sepal length':
        data_used = 'load_iris data from sklearn.datasets'
    elif data_choice == 'Median income and Average room count':
        data_used = 'fetch_california_housing data from sklearn.datasets'
    return data_used

st.title("Correlation and Regression Analysis")

my_form_3 = st.form(key = "form3")
user_input = my_form_3.selectbox('Data',('None', 'Age and BMI', 'Age and Blood Pressure', 'Pupil Teacher Ratio and Crime Rate', 
                        'Petal length and Sepal length', 'Sepal width and Sepal length', 'Median income and Average room count'))
submit = my_form_3.form_submit_button(label = "Submit")

if user_input == 'None':
    st.write('No data selected')

if user_input != 'None':
    dataset_used = get_dataset_details(user_input)
    st.write('Dataset used: ', dataset_used)
    df, X, Y = get_data(user_input)
    indep_var = df.columns[0]
    dep_var = df.columns[1]
    correlation, corr_p_val = calc_corr(X,Y)
    st.write('Pearsons correlation: %.3f' % correlation)
    st.write('p value: %.3f' % corr_p_val)
    params, rsq = reg_analysis(X,Y)
    plot_regression_line(X,Y,params,indep_var, dep_var)
    md_results1 = f"The regression equation is y = **{params[0]:.2f}** + **{params[1]:.2f}** * x."
    st.markdown(md_results1)
    md_results2 = f"R-Square value = **{rsq:.2f}**."
    st.markdown(md_results2)

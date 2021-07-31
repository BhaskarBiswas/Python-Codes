import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_expn_val(rule):
    if rule == 'Linear':
        expn = 1
    elif rule == 'Square':
        expn = 2
    elif rule == 'Cube':
        expn = 3
    return expn


def get_reward(share, expn, n1=0, n2=0, n3=0, n4=0):
    names = ['Atanu', 'Bhaskar', 'Deb', 'Rajani']
    scores = [n1,n2,n3,n4]
    cnt = sum(x > 0 for x in scores)
    prize = cnt*share
    scores_expn = [s**expn for s in scores]
    scores_prop = [s/sum(scores_expn) for s in scores_expn]
    reward_prop = [s*prize for s in scores_prop] 
    data_tuples = list(zip(names,reward_prop,scores))
    df = pd.DataFrame(data_tuples, columns=['Name','Reward','Score'])
    df['Risk'] = np.where(df['Score']>0, share, 0)
    df['Profit/Loss'] = df['Reward'] - df['Risk']
    df = df[['Name','Risk', 'Reward', 'Profit/Loss']]
    df['Reward'] = df['Reward'].round(1)
    df['Profit/Loss'] = df['Profit/Loss'].round(1)
    return df 

def get_plot_bar_graph(df):
    n = df['Name']
    pl = df['Profit/Loss']
    col = []
    for val in pl:
        if val < 0.5:
            col.append('red')
        else:
            col.append('green')
    fig = plt.figure(figsize = (10, 5))
    plt.bar(n, pl, color =col, width = 0.4)   
    plt.xlabel("Players")
    plt.ylabel("Profit/Loss")
    plt.title("Profit/Loss graph")
    plt.show()
    st.write(fig)


st.title("Dream XI !")
my_form = st.sidebar
#(key = "form2")
# Text box
with st.sidebar.form(key ='Form1'):
    share = st.number_input('Player share', step=1, key="share")
    rule = st.selectbox('Rule',('None', 'Linear', 'Square', 'Cube'))
    Atanu = st.number_input('Atanu', step=0.1, key="n1")
    Bhaskar = st.number_input('Bhaskar', step=0.1, key="n2")
    Deb = st.number_input('Deb', step=0.1, key="n3")
    Rajani = st.number_input('Rajani', step=0.1, key="n4")
    op = st.selectbox('Operator',('None', 'Addition', 'Subtraction', 'Multiplication', 'Division'))
    submit = st.form_submit_button(label = "Submit")

expn = get_expn_val(rule)
reward_prop = get_reward(share, expn, Atanu,Bhaskar,Deb,Rajani)

st.header("Results")
#st.subheader("Selection of predicted retention times")
st.dataframe(reward_prop)
get_plot_bar_graph(reward_prop)



import numpy as np
import matplotlib.pyplot as plt
 
  
# creating the dataset
data = {'C':20, 'C++':15, 'Java':30,
        'Python':35}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot

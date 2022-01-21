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
    else:
        expn = 0
    return expn


def get_reward(share, expn, n1=0.01, n2=0.01, n3=0.01, n4=0.01):
    scores = [n1,n2,n3,n4]
    cnt = sum(x > 0 for x in scores)
    if sum(scores) == 0:
        df = pd.DataFrame(columns = ['Name', 'Score', 'Risk', 'Reward', 'Profit/Loss'])
    else:
        names = ['Atanu', 'Bhaskar', 'Deb', 'Rajani']

        prize = cnt*share
        scores_expn = [s**expn for s in scores]
        scores_prop = [s/sum(scores_expn) for s in scores_expn]
        reward_prop = [s*prize for s in scores_prop] 
        data_tuples = list(zip(names,reward_prop,scores))
        df = pd.DataFrame(data_tuples, columns=['Name','Reward','Score'])
        df['Risk'] = np.where(df['Score']>0, share, 0)
        df['Profit/Loss'] = df['Reward'] - df['Risk']
        df = df[['Name', 'Score', 'Risk', 'Reward', 'Profit/Loss']]
        df['Reward'] = df['Reward'].round(1)
        df['Profit/Loss'] = df['Profit/Loss'].round(1)
        if df['Profit/Loss'].sum() > 0:
          adj = df['Profit/Loss'].sum()
          idx = df['Profit/Loss'].idxmax()
          df.loc[idx, 'Profit/Loss'] -= adj
        elif df['Profit/Loss'].sum() < 0:
          adj = df['Profit/Loss'].sum()
          idx = df['Profit/Loss'].idxmin()
          df.loc[idx, 'Profit/Loss'] -= adj
        df.drop(['Risk'], axis=1)
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

def get_allocation(D):
    neg_D = {}
    pos_D = {}
    for key in D:
        if D[key] >= 0:
            pos_D[key] = D[key]
        else:
            neg_D[key] = D[key]
    pos_D = dict(sorted(pos_D.items(), key = lambda x:(x[1],x[0]), reverse=True))
    neg_D = dict(sorted(neg_D.items(), key = lambda x:(x[1],x[0])))
    transfer = []
    for p in pos_D:
        for n in neg_D:
            amt = round(min(pos_D[p],abs(neg_D[n])),1)
            pos_D[p] = pos_D[p] - amt
            neg_D[n] = neg_D[n] + amt
            transfer.append((n,p,amt))
    return transfer

st.title("Dream XI !")
my_form = st.sidebar
#(key = "form2")
# Text box
with st.sidebar.form(key ='Form1'):
    match_name = st.text_input('Match Description', key="match_name")
    share = st.number_input('Player share', step=1, key="share")
    rule = st.selectbox('Rule',('None', 'Linear', 'Square', 'Cube'))
    Atanu = st.number_input('Atanu', step=0.1, key="n1")
    Bhaskar = st.number_input('Bhaskar', step=0.1, key="n2")
    Deb = st.number_input('Deb', step=0.1, key="n3")
    Rajani = st.number_input('Rajani', step=0.1, key="n4")
    submit = st.form_submit_button(label = "Submit")

expn = get_expn_val(rule)
reward_prop = get_reward(share, expn, Atanu,Bhaskar,Deb,Rajani)

reward_dict = reward_prop.set_index('Name')['Profit/Loss'].to_dict()
share_txt = "Share: " + share
rule_txt = "Rule: " + rule
if reward_prop.shape[0] == 0:
    st.markdown("Enter values in the sidebar")
else:
    st.header(match_name)
    st.subheader(share_txt)
    st.subheader(rule_txt)
    st.dataframe(reward_prop)
    get_plot_bar_graph(reward_prop)


transfer_list = get_allocation(reward_dict)

print(transfer_list)

for trsr in transfer_list:
    if trsr[2]>0:
      st.subheader("{} pays Rs. {} to {}".format(trsr[0], trsr[2], trsr[1]))


#K Nearest Neighbor (knn) algorithm in python

#Today's post is on K Nearest neighbor and it's implementation in python.
#Let's take a hypothetical problem. There are two sections in a class. One section was provided a special coaching program in Mathematics, Physics and Chemistry (they were exposed to a particular treatment), and the next objective is to find the efficiency of the program, or how better the particular section performed.
#This is a common problem, and one of the highly used method to calculate is the test and control analysis. However, the main problem remains is how to find the idea control group for a particular test group. One solution to this problem can be given by KNN, or the k-nearest neighbor algorithm. Under this algorithm, for every test student, we can find k different control students based on some pre-determined criteria.
#There are a number of articles in the web on knn algorithm, and I would not waste your time here digressing on that. Rather, I would like to share the python code that may be used to implement the knn algorithm on your data.
#So, we decide to find the control students based on the marks obtained in last examination in Physics, Chemistry and Mathematics. Our aim is to quantify the increase in total marks obtained in the latest examination (Post_Total_Marks) compared to last examination (Pre_Total_Marks).
#The data of the section that was provided the coaching program is saved in Test_Data.csv and the data of the other section is saved in Control_Data.csv.

import csv
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

#reading in test and control files
control = pd.read_csv('/Users/Desktop/Control_Data.csv', sep = ',')
test = pd.read_csv('/Users/Desktop/Test_Data.csv', sep = ',')

test_rcnt = len(test.index)

#subsetting data for only variables to match on
knn_test = test[['Physics','Chemistry','Mathematics']]
knn_control = (control[['Physics','Chemistry','Mathematics']])

#number of nearest neighbors
k=30

#running KNN algorithm
nbrs = NearestNeighbors(n_neighbors = 30, algorithm = 'kd_tree').fit(knn_control)
neigh = nbrs.kneighbors(knn_test, return_distance = False)

#creating a blank dataframe to store output
output1=pd.DataFrame(columns=['roll_number','Physics','Chemistry','Mathematics','Pre_Total_Marks','Post_Total_Marks','nbr_avg_Pre_Total_Marks','nbr_avg_Pre_Total_Marks'])

#looping over the KNN result to summarise the results for all control students
for i in range (0, test_rcnt):
    roll_number = test.ix[i,'roll_number']
    Physics = test.ix[i,'Physics']
    Chemistry = test.ix[i,'Chemistry']
    Mathematics = test.ix[i,'Mathematics']
    Pre_Total_Marks = test.ix[i,'Pre_Total_Marks']
    Post_Total_Marks = test.ix[i,'Post_Total_Marks']

    nbr_Physics = 0
    nbr_Chemistry = 0
    nbr_Mathematics = 0
    nbr_Pre_Total_Marks = 0
    nbr_Post_Total_Marks = 0

    for j in range (0, k):
        nbr_Physics += float(control.ix[neigh[i,j],'Physics'])
        nbr_Chemistry += float(control.ix[neigh[i,j],'Chemistry'])
        nbr_Mathematics += float(control.ix[neigh[i,j],'Mathematics'])
        nbr_Pre_Total_Marks += float(control.ix[neigh[i,j],'Pre_Total_Marks'])
        nbr_Post_Total_Marks += float(control.ix[neigh[i,j],'Post_Total_Marks'])

    nbr_avg_Physics = float(nbr_Physics/k)
    nbr_avg_Chemistry = float(nbr_Chemistry/k)
    nbr_avg_Mathematics = float(nbr_Mathematics/k)
    nbr_avg_Pre_Total_Marks = float(nbr_Pre_Total_Marks/k)
    nbr_avg_Post_Total_Marks = float(nbr_Post_Total_Marks/k)

    output1.loc[len(output)]=[roll_number, Physics, Chemistry, Mathematics, Pre_Total_Marks, Post_Total_Marks, nbr_avg_Pre_Total_Marks, nbr_avg_Pre_Total_Marks]

#calculating the incremental value and sending the output to csv
output1['est_post_marks']=output1['Pre_Total_Marks'] * (output1['nbr_avg_Post_Total_Marks'] / output1['nbr_avg_Pre_Total_Marks'])
output1['marks_inc']=output1['Post_Total_Marks'] - output1['est_post_marks']
output1.to_csv('Output_File_1.csv')

#final summary table
output2=pd.DataFrame(columns=['student_count','tot_marks_inc','avg_marks_inc'])

#summary of the incremental value and sending the output to csv
student_count=output1.roll_number.nunique()
tot_marks_inc=sum(output1['marks_inc'])
avg_marks_inc=tot_marks_inc/student_count
output2.loc[0]=[student_count,tot_marks_inc,avg_marks_inc]

output2.to_csv('Output_File_2.csv')

#Note: The indentation may be broken when you paste this code. Please take additional care to look and correct the indentation before using this code.

Output_File_1.csv contains the data at a student level for the test section. If you notice, the marks in Physics would be close to the nbr_avg_Physics. Same holds for Chemistry and Mathematics as well. This is because the knn algorithm finds control students who have very similar marks in the three subjects compared to the test student.

The incremental value is calculated as follows:
Estimated Post Period Marks = Test Pre Marks * (Control Post Marks / Control Pre Marks)
Marks Increment = Test Post Marks - Estimated Post Period Marks

Output_File_2.csv contains the summarized data of the above calculation.

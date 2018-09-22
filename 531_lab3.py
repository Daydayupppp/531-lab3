
# coding: utf-8

# In[2]:


#lab3q1_Wei_Tang

import matplotlib.pyplot as plt
import csv
with open("36100293.csv", "r") as infile:
    year = []
    gdp = []
    dict1 = {}
    reader = csv.reader(infile)
    for row in reader:
        try:
            if (row[4] == "Gross domestic product (GDP) at market prices"):
                dict1[row[0]] = row[11]
                year.append(int(row[0]))
                gdp.append(float(row[11]))
        except:
            print("Error")
    numyear = input("Enter a year to look up GDP: ")
    base = dict1["1970"]
    numyeargdp = dict1[numyear]
    diff = (float(numyeargdp)-float(base))/float(base)*100
    print("GDP in {} was {} billion dollars, which is {} % different than in 1970.".format(numyear, numyeargdp, diff))  

    plt.plot(year, gdp)
    plt.xlabel("year")
    plt.ylabel("GDP")
    plt.show()


# In[99]:


#lab3q2_Wei_Tang

get_ipython().run_line_magic('matplotlib', 'inline')
import urllib.request
import zipfile
import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans, vq
"""url = 'https://www150.statcan.gc.ca/n1/tbl/csv/11100032-eng.zip'
urllib.request.urlretrieve(url, '11100032-eng.zip')
with zipfile.ZipFile("11100032-eng.zip","r") as zip_ref:
    zip_ref.extractall()"""
    
    
df = pd.read_table("11100032.csv", sep=',')
canada = df.loc[df['GEO'] == "Canada"]
#Second Way
#df[df.GEO == "Canada"]
print("Number of data rows for Canada: ",len(canada))
total_income = canada.loc[canada["Income concept"] == "Average total income"]
income_tax = canada.loc[canada["Income concept"] == "Average income tax"]
print("Number of data rows for set one: ", len(total_income))
print("Number of data rows for set one: ", len(income_tax))
print(income_tax[['Income concept','VALUE']].head(4))

income = total_income[["VALUE"]]
tax = income_tax[["VALUE"]]
model1 = LinearRegression()
model1.fit(tax, income)

bar_width = 0.35
opacity = 0.85

plt.scatter(tax, income, color='g')
plt.plot(tax, model1.predict(tax),color='r')
plt.show()

eco_family = total_income["Economic family type"].tolist()
income = total_income["VALUE"].tolist()
tax = income_tax["VALUE"].tolist()

index = np.arange(len(income_tax))
plt.bar(index, income, width = bar_width, alpha = opacity, label="income")
plt.bar(index + bar_width, tax, width = bar_width, alpha = opacity, label="Tax")
plt.legend()
plt.xticks(index + bar_width, eco_family, rotation=90)
plt.figure(figsize=(6,6))
plt.show()


data = []  
for i in range(0, len(income_tax)):
    data.append([tax[i], income[i]])
numclusters = 3
centroids,_ = kmeans(data, numclusters)
idx,_ = vq(data, centroids)

clusters = [] 
for i in range(0, numclusters): 
    clusters.append([[],[]]) 
for i in range(0,len(idx)): 
    clusterIdx = idx[i] 
    clusters[clusterIdx][0].append(data[i][0]) 
    clusters[clusterIdx][1].append(data[i][1]) 
plt.plot(clusters[0][0],clusters[0][1],'ob', clusters[1][0],clusters[1][1],'or', clusters[2][0], clusters[2][1], 'oy') 
plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8) 
plt.show()


# In[94]:





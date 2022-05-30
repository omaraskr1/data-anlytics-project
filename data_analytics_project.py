#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io 
from sklearn.linear_model import LinearRegression
from sklearn import tree


# In[2]:



df = pd.read_csv("employee_attrition_test2.csv")

df.head()

#romove from data all nulls values
df=df.dropna(axis=0,how='any')

df.describe()



# In[3]:



print('''employee_attrition_test2 description:
1-BusinessTravel:consist of Travel rarely,Travel_Frequently,Non-Travel  
2-EducationField:consist of Life Sciences,Medical,Technical Degree,Marketing,Human Resources
3-gender: Male/Female
4-Age: employee age in years
5-JobRole: Sales Executive,Research Scientist ,Laboratory Technician,Manufacturing Director 
,Healthcare Representative ,Manager, Research Director,Human Resources, 
Sales Representative 
6-MaritalStatus:Married ,Single,Divorced '''
)


# In[4]:



"""**Anomaly** **Detection** **algorithem**

"""

# Extract two column for the algorithem and convert them to numpy array
dbscanData = df[['DailyRate','MonthlyIncome']]
dbscanData = dbscanData.values.astype('float32',copy=False)
dbscanData

# potting data before detecting outliears
plt.scatter(dbscanData[:,0], dbscanData[:,1])
plt.show()

# determining the epsilon and min samples per cluster
dbscan = DBSCAN(eps = 150, min_samples = 3)

# fit data to the model
pred = dbscan.fit_predict(dbscanData)
# determine outliers 
outliers = np.where(pred == -1)
values = dbscanData[outliers]
# potting data after detecting outliears
plt.scatter(dbscanData[:,0], dbscanData[:,1])
plt.scatter(values[:,0], values[:,1], color='r')
plt.show()


# In[5]:


"""Visualization"""

#get for each Job Rule number of males that work over time and i used histogram to help me use many columns   

x=df.loc[(df["Gender"] == "Male")  &  (df["OverTime"]=='Yes')]
p=sns.histplot(y="JobRole",data=x)
p.set_title("Male Job Rule that work over time")


# In[6]:


"""Visualization"""
#get martial status for female that work more than 7 years in company and i used pie chart to clarify percentage of martial status and i have less than 5 columns
y=df.loc[(df["Gender"] == "Female") & (df["TotalWorkingYears"] > 7) ]
arr1 = np.array(y[y["MaritalStatus"]=="Divorced"].count())
arr2 =np.array(y[y["MaritalStatus"]=="Married"].count())
arr3=np.array(y[y["MaritalStatus"]=="Single"].count())
res = []
for i in arr1:
    if i not in res:
        res.append(i)
for i in arr2:
    if i not in res:
        res.append(i)
for i in arr3:
    if i not in res:
        res.append(i)
        
plt.pie(res,labels=["Divorced","Married","Single"],autopct='%1.1f%%')
plt.title("Martial Status for Female")
plt.show()


# In[7]:


"""Text **mining**"""

import nltk
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download("vader_lexicon")
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.sentiment import SentimentIntensityAnalyzer



#read data from file
with open('test.txt') as f:
    contents = f.readlines()
    sentences = list(contents)

    # Removing punctuation from data
    for sentence in sentences:
       sentence=sentence.lower()
       print("\nOriginal Sentence: " + sentence)
       result = sentence.translate(str.maketrans('', '', string.punctuation))
       print("After removing punctuation: " + result)

    # Tokenization
    tokenizedText = []
    for sentence in sentences:
       split_sentence_towords = word_tokenize(sentence)
       tokenizedText.append(split_sentence_towords)
    for sentence in tokenizedText:
       print(sentence)

    # Stop Words
    stopWords = set(stopwords.words('english'))
    for sentence in tokenizedText:
       print("\nSentence: ", sentence)
       for word in sentence:
           if word in stopWords:
               sentence.remove(word)
       print("After removing stop words: ", sentence)

    # Lemmatization
    word_lemm = WordNetLemmatizer()
    for sentence in tokenizedText:
       l_list = []
       print("\nWords: ", sentence)
       for word in sentence:
           l_list.append(word_lemm.lemmatize(word))
           print("After Lemmatization: ", l_list)

    # Stemming
    word_stemmer = PorterStemmer()
    for words in tokenizedText:
       s_list = []
       print("\n3Words: ", words)
       for word in words:
           s_list.append(word_stemmer.stem(word))
           print("After stemming: ", s_list)

    # Sentiment Analysis
    for sentence in sentences:
       s = SentimentIntensityAnalyzer()
       vs = s.polarity_scores(sentence)
       print(sentence, str(vs))


# In[8]:


"""Hosni **part**"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
sns.set(color_codes=True )
air = pd.read_csv ('employee_attrition_test2.csv')
air.hist(figsize=(7,7))
sns.pairplot(air)
air=air.dropna()


# In[9]:



#predict 1
df.corr()

x = df[['JobLevel']]
y = df['MonthlyIncome']
model = LinearRegression() 
clf = model.fit(x,y)
print( 'Co.fficitnt: ', clf.coef_)
predictions = model.predict(x)
for index in range(len(predictions)):
  print('Actual: ', y[index], '        ','Prtdicttd : ', predictions[index])


# In[10]:


#predict 2
a = df['Education'].values.reshape(324,1)
a.shape

x1 = df['Education']
y1 = df['DailyRate']
model = tree.DecisionTreeRegressor()
model.fit(a,y1)
predictions = model.predict(a)
print(model.feature_importances_)
for index in range(len(predictions)):
  print('Actual: ', y1[index], '        ','Prtdicttd : ', predictions[index])


# In[ ]:





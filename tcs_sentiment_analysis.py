import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt
from collections import Counter

plt.style.use("fivethirtyeight")

tag = []
tweets = []

with open("tweets.json") as posts:
    data=json.load(posts)

    # Enter twitter accounts from any of these [ damu , dinesh , naveen , harsha , raghav , seenu , gouthami ]

name = input("Enter the name of twitter account:")

for i in data["tweets"]:
    if i["name"]==name:
        tag.append(i["posts"])

for i in tag:

    for t in i:
      tweets.append(t)

   #data=pd.DataFrame(tweets)

df= pd.DataFrame([i for i in tweets],columns=["Tweets"])

   # show the first 5 rows of data

#print(df.head())

# cleaning up the text using a function

def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', "", text)  # Removed '@' symbols
    text = re.sub(r'#', '', text)  # Removed ' # ' symbols
    text = re.sub(r'RT[\s]+', '', text)  # Removed RT
    text = re.sub(r'https?:\/\/\s+', '', text)  # Removed "http" links
    return text

    # adding the cleaned data to a column in the dataframe with name "tweets"

df["Tweets"] = df["Tweets"].apply(cleanTxt)


# print(df)

# Creating functions to get polarity and subjectivity  and adding those as two separate columns

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def getPolarity(text):
    return TextBlob(text).sentiment.polarity


df["Subjectivity"] = df["Tweets"].apply(getSubjectivity)
df["Polarity"] = df["Tweets"].apply(getPolarity)

print(df)
print()
print()

# Plotting the Wordcloud

allwords = ' '.join([twts for twts in df["Tweets"]])
#print(allwords)

    # visualizing the positive words and their occurances
count={}
split = allwords.split(" ")
#print(split)
for i in split:
    count[i] = count.get(i, 0) + 1
sorted = sorted(count.items())
sorted = sorted[:len(sorted)]

polarity = {}
negativity = {}
for k,v in sorted:
    if(TextBlob(k).sentiment.polarity)>0:

       polarity[k]=polarity.get(k,v)

    if (TextBlob(k).sentiment.polarity) < 0:
        negativity[k] = negativity.get(k, v)


plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.bar([k for k,v in polarity.items()],[v for k,v in polarity.items()],align="center",color=["teal","skyblue","#005566","grey","#9988AA"])
plt.title("Sentiment Analysis from twitter data")
plt.xlabel("Words with positive  polarity")
plt.ylabel("Frequency of occurance")
plt.subplot(2,1,2)
plt.bar([k for k,v in negativity.items()],[v for k,v in negativity.items()],align="center",color=["teal","skyblue","#005566","grey","#9988AA"])
plt.xlabel("Words with negative  polarity")
plt.ylabel("Frequency of occurance")
plt.show()

wordcloud = WordCloud(width=500, height=300, random_state=21, max_font_size=119).generate(allwords)
plt.imshow(wordcloud , interpolation = "bilinear")
plt.axis("off")
plt.show()




# Creating a function to demonstrate the negative , neutral and postitive statements

def getAnalysis(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"

print("Polarity description of each tweet")
print()
print()
df["Analysis"] = df["Polarity"].apply(getAnalysis)

print(df)

# printing all the positive statements

j = 1
k=1
sortedDF = df.sort_values(by=["Polarity"])
print()
print()
for i in range(0, sortedDF.shape[0]):
    if (sortedDF["Analysis"][i] == "Positive"):
        while k==1:
            print("        The positive statements from this account are :    ")
            print()
            print()
            k+=1
        print(str(j) + ')' + sortedDF["Tweets"][i])
        print()
        j += 1

    # printing all the negative statements

j = 1
k=1
sortedDF = df.sort_values(by=["Polarity"], ascending="False")
print()
for i in range(0, sortedDF.shape[0]):
    if (sortedDF["Analysis"][i] == "Negative"):
        while k==1:
            print("       The negative statements from this account are :   ")
            print()
            k+=1
        print(str(j) + ')' + sortedDF["Tweets"][i])
        print()
        j += 1

    # plotting the polarity and subjectivity

plt.figure(figsize=(10,10))
for i in range(0, df.shape[0]):
    plt.scatter(df["Polarity"][i], df["Subjectivity"][i], color="Blue")

plt.title("Sentiment Analysis")
plt.xlabel("Polarity")
plt.ylabel("Range")
plt.show()

'''# Getting the percentages of positive and negative tweets

ptweets = df[df.Analysis == "Positive"]
ptweets = ptweets["Tweets"]
round((ptweets.shape[0] / df.shape[0]) * 100 , 1)

ntweets = df[df.Analysis == "Negative"]
ntweets = ntweets["Tweets"]
round((ntweets.shape[0] / df.shape[0]) * 100 , 1)'''

# plotting and visualizing the counts

plt.title("Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Counts")
df["Analysis"].value_counts().plot(kind="bar")
plt.show()









#simple sentiment analyzer done using vader of nltk 


'''
this is the basic formula used to calculate the compound score
compund_score=(positive score-negative score)/(positive score+negative score+neutral score)
neutral score= 1-(positive score + negative score)
'''

#need to install pandas as pip install pandas 
#need to install nltk as pip install nltk


import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


#sentences list for sentiment analysis
sentences=["I had a fantastic time at the party last night.",
    "The movie was incredibly boring and dull.",
    "She did an amazing job on her presentation.",
    "The weather today is absolutely dreadful.",
    "I love the new features in the software update.",
    "The service at the restaurant was terrible and slow.",
    "I had a wonderful vacation in the mountains.",
    "The book was quite disappointing and not engaging.",
    "The new smartphone is incredibly innovative.",
    "I found the lecture to be rather uninteresting.",
    "The concert was superb and exceeded all my expectations.",
    "My experience with the customer service was frustrating.",
    "The meal was delicious and well-prepared.",
    "The hotel room was small and very uncomfortable.",
    "I’m thrilled with the results of my exam.",
    "The project was poorly managed and a complete failure.",
    "The new café in town is cozy and charming.",
    "I was let down by the product quality.",
    "The team did an excellent job on the new campaign.",
    "The noise from the construction site is extremely irritating.",
    "The cake at the bakery was absolutely delightful.",
    "The traffic was terrible and caused a lot of delays.",
    "I enjoyed the holiday season with family and friends.",
    "The workshop was a waste of time and poorly organized.",
    "The view from the top of the hill is breathtaking.",
    "The customer support was unhelpful and rude.",
    "I am very pleased with the outcome of the project.",
    "The movie was a letdown and didn’t live up to the hype.",
    "The gym facilities are top-notch and well-maintained.",
    "The cleanliness of the apartment was quite poor.",
    "I appreciate the thoughtful gift you gave me.",
    "The flight experience was awful and stressful.",
    "The new restaurant has a fantastic ambiance.",
    "The instructions were confusing and misleading.",
    "The new book is incredibly captivating and well-written.",
    "The sales team was unresponsive and disorganized.",
    "The service at the spa was exceptional and relaxing.",
    "The software crashes frequently and is frustrating to use.",
    "I’m excited about the new opportunities ahead.",
    "The dinner was overpriced and not worth the money.",
    "The concert performance was electrifying and memorable.",
    "The product didn’t meet my expectations and was defective.",
    "I enjoyed every moment of the beautiful sunset.",
    "The presentation was poorly done and lacked substance.",
    "The new car has exceeded all my expectations.",
    "The hotel staff was rude and unaccommodating.",
    "The community event was a huge success.",
    "The noise from the neighbors is quite bothersome.",
    "I’m thrilled with the improvements in the new version.",
    "The service at the fast food place was subpar.",
    "The beach vacation was relaxing and enjoyable.",
    "The game was frustrating and full of glitches.",
    "I’m delighted with the outcome of the meeting.",
    "The customer service experience was disappointing.",
    "The garden is looking beautiful and vibrant.",
    "The restaurant had a terrible ambiance and service.",
    "I’m really happy with my new haircut.",
    "The lecture was monotonous and hard to follow.",
    "The new feature in the app is very user-friendly.",
    "The hotel room was dirty and not well-maintained.",
    "I had a blast at the amusement park yesterday.",
    "The product packaging was poor and flimsy.",
    "The new paint job on the house looks fantastic.",
    "The food at the event was bland and tasteless.",
    "I’m satisfied with the level of service provided.",
    "The experience at the dentist was very uncomfortable.",
    "The new dress looks stunning on you.",
    "The project did not go as planned and was disappointing.",
    "The museum visit was educational and enjoyable.",
    "The phone call with customer support was frustrating.",
    "The outdoor adventure was exhilarating and fun.",
    ]

#converting list to dataframe
df=pd.DataFrame({"sentence":sentences})


#initialize SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()


#function calculate the sentiment scores 
def analyze_sentiment(text):
    score=sia.polarity_scores(text)
    return score['compound']


#calculating the sentiment scores of each sentence in the dataframe and adding it to the dataframe as column sentiment_scores
df["sentiment_scores"]=df["sentence"].apply(analyze_sentiment)


#assign true value for sentences with sentiment_scores higher than 0.05 and false for values less than -0.05 and neutral for 0 values in the column positive
df["positive"]=df["sentiment_scores"].apply(lambda x: 'True' if x>0.05 else 'False' if x<-0.05 else 'neutral')

#assign true value for sentences with sentiment_scores higher than 0.05 and false for values less than -0.05 and neutral for 0 values in the column negative
df["negative"]=df["sentiment_scores"].apply(lambda x: 'True' if x<-0.05 else 'False' if x>0.05 else 'neutral')


#convert the data set into a csv file and export
df.to_csv("sentiments.csv",encoding="utf-8",index=False)

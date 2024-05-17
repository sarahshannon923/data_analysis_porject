import googleapiclient.discovery
import pandas as pd 
import requests
from sklearn.feature_extraction.text import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Start with data extraxction
#These contain all the necessary Keys and details to access youtube comments 
dev='AIzaSyA175sujR_6Llq-5f6hX9A2RiNs1QaNgbM'
api_service_name = "youtube"
api_version = "v3"
api_key = dev

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=api_key)

#create a defined function to extract the comments
def get_comments(video):
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video,
        maxResults=100
    )
#emepty list for the comments
    comments = []

#execute the request.
    response = request.execute()

#get the comments from the response using a loop.
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        public = item['snippet']['isPublic']
        comments.append([
            comment['likeCount'],
            comment['textOriginal'],
            comment['videoId']
        ])
#a loop to get to the next page once the comments from the previous one are exhausted
    while True:
        try:
            nextPageToken = response['nextPageToken']
        except KeyError:
            break
        nextPageToken = response['nextPageToken']
#create a new request object with the next page token.
        nextRequest = youtube.commentThreads().list(part="snippet", videoId=video, maxResults=100, pageToken=nextPageToken)
#execute the next request.
        response = nextRequest.execute()
#get the comments from the next response.
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            public = item['snippet']['isPublic']
            comments.append([
                comment['likeCount'],
                comment['textOriginal'],
                comment['videoId']
            ])

    df2 = pd.DataFrame(comments, columns=[ 'like_count', 'Comment','video_id'])
    return df2  


#get comments from four different videos 
dft1 = get_comments('vN6vAneCWAA')
print(dft1)
dft2 = get_comments('qeDIRKEl260')
print(dft2)
dft3 = get_comments('8qX-eKiK-Fw')
print(dft3)
dft4 = get_comments('p9aLKVBI8N0')
print(dft4)

#all of these have the topic horse training
#combine four objects into one data frame
df_training = pd.concat([dft1,dft2,dft3,dft4], ignore_index=True)[['like_count', 'Comment', 'video_id']]
print(df_training)

#this was the loop to pull from multiple videos but it stopped function and I couldn't fix it so I found another solution
# df_training = pd.DataFrame()
# for i in ['vN6vAneCWAA','qeDIRKEl260','8qX-eKiK-Fw','p9aLKVBI8N0']:
#     df2 = get_comments(i)
#     df_training = pd.concat([df_training, df2]) this isn't working here i will come back

#save dataframe
df_training.to_csv('webscrapingtraining.csv')

# df_competition = pd.DataFrame()
# for i in ['5eGmxbOj_LY','Py45OeysrmY','pq_pb88pBOQ','nA0LUwf0pT8']:
#     df4 = get_comments(i)
#     df_competition = pd.concat([df_competition,df4])

# df_competition.head()

#competion videos 
dfc1 = get_comments('5eGmxbOj_LY')
print(dfc1)
dfc2 = get_comments('Py45OeysrmY')
print(dfc2)
dfc3 = get_comments('pq_pb88pBOQ')
print(dfc3)
dfc4 = get_comments('nA0LUwf0pT8')
print(dfc4)

df_competition = pd.concat([dfc1,dfc2,dfc3,dfc4], ignore_index=True)[['like_count', 'Comment', 'video_id']]
print(df_competition)

df_competition.to_csv('webscrapingcompetition.csv')


#defined function for removing punctuation emojis and converting all letter to lowercase
def clean_punctuation(Comment):
    no_punc = re.sub(r'[^\w\s]', '',str(Comment))#re.sub is for substrings
    no_digits = ''.join([i for i in no_punc if not i.isdigit()]) #join i.isdigit
    return(no_digits)

def all_lowercase(Comment):
    text_lower = Comment.lower()
    return text_lower

df_competition['Comment'] = df_competition['Comment'].apply(clean_punctuation)
df_training['Comment'] = df_training['Comment'].apply(clean_punctuation)

df_competition['Comment'] = df_competition['Comment'].apply(all_lowercase)
df_training['Comment'] = df_training['Comment'].apply(all_lowercase)

#save again so there is a copy of raw data and cleaned data
df_competition.to_csv('cleaneddatacompetition.csv')
df_training.to_csv('cleaneddatatraining.csv')

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentiment_analyzer = SentimentIntensityAnalyzer()

def get_sentiment(Comment):
    sentiment_scores = sentiment_analyzer.polarity_scores(Comment)
    return sentiment_scores['compound']

df_training['Sentiment'] = df_training['Comment'].apply(get_sentiment)
df_competition['Sentiment'] = df_competition['Comment'].apply(get_sentiment)

print("DataFrame training with Sentiment Scores:")
print(df_training)
print("\nDataFrame competition with Sentiment Scores:")
print(df_competition)

average_sentinment_training = df_training['Sentiment'].mean()
average_sentinment_competition = df_competition['Sentiment'].mean()

print(average_sentinment_competition)
print(average_sentinment_training)

#visualisation
df_training['DataFrame'] = 'Training videos'
df_competition['DataFrame'] = 'Competition videos'
combined_df = pd.concat([df_training, df_competition])
custome_palette = {'Training videos':'#00FF00', 'Competition videos':'#008080'}

#box plot is the best chart for this comparison as the relationships and differences are visually clear
plt.figure(figsize=(10, 6))
sb.boxplot(x='DataFrame', y='Sentiment', data=combined_df, palette=custome_palette)
plt.title('Sentiment Comparison Between Two DataFrames')
plt.show()

#insights 

#both of these types of videos have a positive skew to their sentiment scores showing a support fanbase for both youtube channels
#the training videos have more negative outliers suggestion some negative comments which is to be expected since no one can agree on best training methods
#the training videos also have a higher median sentiment score than the competition videos 
#the competition videos have a longer third quartile showing more dispertion of sentiment scores which are closer to neutral
#the competition video have a bigger minimum range

# reccomendations
# it is reccomend that more research is conducted into why the sentiment scores of the competion videos have a lower range
# it is reccomend that an analysis of the outlier comments is conducted to see if any glaring issues can be decerned
# it is reccomended that that emphsis is put on training video going forward.

#references
#https://github.com/wjbmattingly/topic_modeling_textbook/blob/main/02_03_setting_up_tf_idf.ipynb
#https://github.com/analyticswithadam/Python/blob/main/YouTube_Comments_Advanced.ipynb
#https://www.youtube.com/@elphick.event.ponies
#https://www.youtube.com/@LifeontheLeftRein
#https://www.kaggle.com/code/ar5entum/comparing-methods-of-sentiment-analysis


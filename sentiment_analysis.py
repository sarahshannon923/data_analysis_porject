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

#tf-idf analysis begins here
#1 vectorize
# tf-idf algorithm
vectorize = TfidfVectorizer(
    lowercase=True,
    max_features=100,
    max_df=0.8,
    min_df=5,
    ngram_range=(1,3),
    stop_words="english"
)

#isolate comment text from the numerical data in the data frame
Comment_data = df_training['Comment'].tolist()

#pass in document
vectors = vectorize.fit_transform(Comment_data)
print(vectors[0])

#get text repersentation of keywords
feature_names = vectorize.get_feature_names_out()
print(feature_names[0]) #absolutely most common # results

#convert vector into a repersentation of each word and their td-idf score
dense = vectors.todense()
print(dense[0])

#convert into list form
denselist = dense.tolist()
print(denselist[0])

#covert the numbers into words
all_keywords = []
for description in denselist:
    x=0
    keywords = []
    for word in description:
        if word > 0:
            keywords.append(feature_names[x])
        x=x+1
    all_keywords.append(keywords)

print ("Only Keywords Text:")
print (all_keywords[0])

#2 K-means clustering
#find overlap in keywords
#5 clusters
true_k = 5
#create model
model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)

#fit vectors to model
model.fit(vectors)
#tf-idf as the basis of cluster
order_centroids = model.cluster_centers_.argsort()[:,::-1]
terms = vectorize.get_feature_names_out()
#nuemeric value of clusters
print(order_centroids)

print(terms[92])

#automate keyword conversion
i=0
for cluster in order_centroids:
    print(f"Cluster {i}")
    for keyword in cluster[0:10]:
        print(terms[keyword])
    print('')
    i=i+1

#visualisation
kmean_indicies = model.fit_predict(vectors)
pca = PCA(n_components=2) #prinicipal component analysis
scatter_plot_points = pca.fit_transform(vectors.toarray())

colors = ["r", "b", "m", "y", "c"]

x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]

fig, ax = plt.subplots(figsize=(50,50))
scatter = ax.scatter(x_axis,y_axis, c=[colors[d] for d in kmean_indicies])
plt.savefig("scatter_sentimentanalysis.png")
plt.show()


#conclusions about the data and further reccomendations
#incomplete I willupdate

#references
#https://github.com/wjbmattingly/topic_modeling_textbook/blob/main/02_03_setting_up_tf_idf.ipynb
#https://github.com/analyticswithadam/Python/blob/main/YouTube_Comments_Advanced.ipynb

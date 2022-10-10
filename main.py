import snscrape.modules.twitter as sntwitter
import pandas as pd
import emoji
import nltk
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
import seaborn as sb
import matplotlib.pyplot as plt


def emoji_replace(twt):
    return(emoji.replace_emoji(twt, ''))


def emo_label(text):
    return (emotion(text)[0]['label'])


topic=input("Enter the topic:")


tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
nltk.download('omw-1.4')


emoscore=[]


# Creating list to append tweet data to


attributes_container = []

# Using TwitterSearchScraper to scrape data and append tweets to list
for i, tweet in enumerate(
        sntwitter.TwitterSearchScraper(topic).get_items()):
    if i > 150:
        break
    attributes_container.append([tweet.user.username, tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])


# Creating a dataframe to load the list
tweets_df = pd.DataFrame(attributes_container, columns=["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"])
tweets_df['Tweet']=tweets_df['Tweet'][0:150].apply(emoji_replace)
tweets_df['emotion']=tweets_df['Tweet'][0:150].apply(emo_label)


sb.countplot(data = tweets_df, y = 'emotion').set_title("Emotion Distribution")
plt.show()







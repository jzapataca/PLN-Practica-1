import random
import os

import xml.etree.ElementTree as ET

def read_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root

file_path = "./data/raw/corpus/general-train-tagged-3l.xml"
root = read_xml(file_path)

def filter_tweets(root):
    positive_tweets = []
    negative_tweets = []
    
    for tweet in root.findall(".//tweet"):
        polarity = tweet.find("sentiments/polarity/value").text
        if polarity == "P":
            positive_tweets.append(tweet)
        elif polarity == "N":
            negative_tweets.append(tweet)
    
    num_to_select = min(len(positive_tweets), len(negative_tweets)) // 5
    selected_positive = random.sample(positive_tweets, num_to_select)
    selected_negative = random.sample(negative_tweets, num_to_select)
    
    return selected_positive, selected_negative

def save_tweets(tweets, folder):
    file_path = f"{folder}/tweets.txt"
    if os.path.exists(file_path):
        os.remove(file_path)
    for tweet in tweets:
        tweet_text = tweet.find("content").text
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(tweet_text + "\n")

selected_positive, selected_negative = filter_tweets(root)

save_tweets(selected_positive, "./data/interim/pos")
save_tweets(selected_negative, "./data/interim/neg")

print("Filtered and saved tweets successfully.")
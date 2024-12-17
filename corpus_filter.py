import random
import os
import xml.etree.ElementTree as ET

def read_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root

train_file = "./data/raw/corpus/general-train-tagged-3l.xml"
test_file = "./data/raw/corpus/general-test-tagged-3l.xml"

def filter_tweets(root):
    positive_tweets = []
    negative_tweets = []
    
    for tweet in root.findall(".//tweet"):
        polarity = tweet.find("sentiments/polarity/value").text
        if polarity == "P":
            positive_tweets.append(tweet)
        elif polarity == "N":
            negative_tweets.append(tweet)
    
    return positive_tweets, negative_tweets

# Read and filter both files
train_root = read_xml(train_file)
test_root = read_xml(test_file)

# Get positive and negative tweets from both files
train_pos, train_neg = filter_tweets(train_root)
test_pos, test_neg = filter_tweets(test_root)

# Combine tweets from both files
all_positive = train_pos + test_pos
all_negative = train_neg + test_neg

# Select random subset
num_to_select = min(len(all_positive), len(all_negative)) // 5
selected_positive = random.sample(all_positive, num_to_select)
selected_negative = random.sample(all_negative, num_to_select)

def save_tweets(tweets, folder):
    file_path = f"{folder}/tweets.txt"
    if os.path.exists(file_path):
        os.remove(file_path)
    for tweet in tweets:
        tweet_text = tweet.find("content").text
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(tweet_text + "\n")

save_tweets(selected_positive, "./data/interim/pos")
save_tweets(selected_negative, "./data/interim/neg")

print(f"Filtered and saved {num_to_select} tweets of each polarity successfully.")
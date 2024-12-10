from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import os
import csv

driver = webdriver.Chrome()
url = "https://steamcommunity.com/app/1874000/reviews/?browsefilter=toprated&snr=1_5_100010_"
driver.get(url)

reviews = driver.find_elements(By.CLASS_NAME, "apphub_UserReviewCardContent")

while len(reviews) < 200:
    driver.execute_script("arguments[0].scrollIntoView()", reviews[-1])
    reviews = driver.find_elements(By.CLASS_NAME, "apphub_CardTextContent")

reviews_text = [review.text for review in reviews]
df_reviews = pd.DataFrame(reviews_text, columns=['Reviews'])
df_reviews.to_csv("./data/reviews.csv", index=False, header=False)

driver.quit()
#!/usr/bin/env python
# encoding: utf-8

import tweepy #https://github.com/tweepy/tweepy
import csv

#Twitter API credentials
#consumer_key = "xSHOwWn5Xj1w4GrGEynJOHnQ7"
#consumer_secret = "TRBwUu1dTje8tinDEySVwnICOROE0jYZOcl4oUcQxjUcqLgfun"
#access_key = "3040773440-1OyU0AndGsSyAjVBUN2bfWp3UI3ekOEunfnzS6y"
#access_secret = "mb9jVNXKHTT1RPFmNJrVwwfCoT7cRrrg4TFbYxd6yX7qz"


consumer_key = "tNq5rXzQtpCfQN5SqhvYDjx84" 
consumer_secret = "3KTXCM4jVgizY1MCPHCm02oxV21OBMdX2jGAfY7mZ27aSbwveF" 
access_key = "1003684959419322371-2yvCMWHGhs7S32TIgEY9wV9SziKg11"
access_secret = "MyIrkq24K7R7WehRLeRkpVCePbmQB7L56He65gfeaLgAw"


def get_all_tweets(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method
	
	#authorize twitter, initialize tweepy
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	api = tweepy.API(auth)
	
	#initialize a list to hold all the tweepy Tweets
	alltweets = []	
	
	#make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(screen_name = screen_name,count=200)
	
	if len(new_tweets) < 1:
		print("dii")
		return []
	#save most recent tweets
	alltweets.extend(new_tweets)
	
	#save the id of the oldest tweet less one
	
	if len(new_tweets) > 0:
		print("do")
		oldest = alltweets[-1].id - 1
	
	#keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0:
		print "getting tweets before %s" % (oldest)
		
		#all subsiquent requests use the max_id param to prevent duplicates
		new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
		
		#save most recent tweets
		alltweets.extend(new_tweets)
		
		#update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1
		print("second")
		print(oldest)
		
		print "...%s tweets downloaded so far" % (len(alltweets))
	
	#transform the tweepy tweets into a 2D array that will populate the csv	
	outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
	
	#write the csv	
	with open('%s_tweets.csv' % screen_name, 'wb') as f:
		writer = csv.writer(f)
		print("writting")
		writer.writerow(["id","created_at","text"])
		writer.writerows(outtweets)

	return outtweets
  	
#	pass


if __name__ == '__main__':
	#pass in the username of the account you want to download
	#all_t=get_all_tweets("J_tsar")
	all_t =get_all_tweets("realDonaldTrump")
	print("number of tweets :" + str(len(all_t)))

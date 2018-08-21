This is the infrastructure for the TweetBot Podcast recommender that I build
for the Insight Data Project. It builds a tweetbot that responds with podcast recommendations
as a reply to tweets sent by users. 

The ML pipeline can be describe as follows:
(i) It takes existing (~120,000) podcast data in multiple languages (all.csv) and performs nlp preprocessing (word stopping, cleaning of e-mails urls, TF-IDF). 
Look at the data_processing.py file.
(ii) It performs cosine similarity (TF-IDF) between preprocessed podcast descriptions and the user's posting history 
to select 100 candidate podcasts (bot class in podfather.py).
From this pool of 100 podcast candidates it randomly selects 3 podcasts as suggestions ((bot class in podfather.py).
(iii) Then it uses buzz words from from the (100) podcast candidates to find additional 
recommended podcasts episodes from the Listen Notes data base ((bot class in podfather.py).
(iv) Also, it looks into who the user is following and uses the names of the celebrities followed to find 
podcasts episodes on the Listen Notes database ((bot class in podfather.py)
(v) Finally it sends the tweet with podcast recommendations to the user.

All of the files (all.csv, podfather.py,data_processing.py) needed to run the tweetbot can be found in the /notebooks/bot directory.

To run tweetbot just run in the command line:

python podfather.py

**NOTE**: the podfather module depends on the data_processing.py and all.csv files. Also the user needs to create an
account on Twitter and obtain credentials (i.e. API key, API secret, Access token and Access token secret) on the Twitter developer site to access the Twitter API 
(Go to http://socialmedia-class.org/twittertutorial.html for further guidance).

The Project results, analysis and source code can be found in the following sub-directories of the /notebooks directory:

**bot directory: contains the bulk of the tweetbot infrastructire to perform LDA, TF-IDF 
  		 contained in the data_processing.py and podfather.py script**


analysis -> counting the languages of the project

api -> example notebook to connect to Listen Notes API

lda -> LDA example notebook

nmf -> non-negative factorization Calculations notebook (too slow)

sql -> PosGres SQL transoformation notebook

lag_analysis -> time_analysis.ipynb looks at estimates of response of the tweetbot ( worst case scenario is 20 seconds)

url_cleaner ->  the url_cleaner.ipynb detects malfunctioning podcasts urls 

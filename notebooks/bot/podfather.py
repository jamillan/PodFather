
from __future__ import unicode_literals
import re
import numpy as np
import pandas as pd
from pprint import pprint
import random
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
import tweet_dumper
import time
import itunes

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import unirest
import tweepy
import random
#from our keys module (keys.py), import the keys dictionary

CONSUMER_KEY = "my CONSUMER_KEY"
CONSUMER_SECRET = "my CONSUMER_SECRET"
ACCESS_TOKEN = "my ACCESS TOKEN"
ACCESS_TOKEN_SECRET = "my ACCESS TOKEN SECRET" 

#CONSUMER_KEY = "tNq5rXzQtpCfQN5SqhvYDjx84"
#CONSUMER_SECRET = "3KTXCM4jVgizY1MCPHCm02oxV21OBMdX2jGAfY7mZ27aSbwveF"
#ACCESS_TOKEN = "1003684959419322371-2yvCMWHGhs7S32TIgEY9wV9SziKg11"
#ACCESS_TOKEN_SECRET = "MyIrkq24K7R7WehRLeRkpVCePbmQB7L56He65gfeaLgAw"


auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# Run in python console
import nltk; nltk.download('stopwords')
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words(['english','danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian', 'portuguese', 'russian', 'spanish', 'swedish', 'turkish'])
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

import string 
strings = string.ascii_lowercase[1:26]

df = pd.read_csv("all.csv",header=None)
dict_num_colnames = {'slug':0,'name':2,'image_url':3,'feed_url':4,'website_url':5,
                      'itunes_owner_name':6,'itunes_owner_email':7,
                        'description':11,'itunes_summary':12}

#backup file where we store the description, itunes_summary, name and website for each podcast
# some of this contes could be Nan, empty etc

new_df = pd.DataFrame({'content':df.iloc[:,dict_num_colnames['description']],    
                       'content2':df.iloc[:,dict_num_colnames['itunes_summary']],
                       'name': df.iloc[:,dict_num_colnames['name']],
                       'website_url': df.iloc[:,dict_num_colnames['website_url']]
                      })

# Store the ids of english files (good ids list)

bad_ids= []
good_ids = []

#
data = new_df.content[:].values.tolist()
data1 = new_df.content2[:].values.tolist()
pod_names = new_df.name[:].values.tolist()
pod_url = new_df.website_url[:].values.tolist()
#data = new_df.text[2:nbr_data].values.tolist()
new_data = []
for idx,_sentence in enumerate(data):

    if isinstance(_sentence,float):
        bad_ids.append(idx)
        continue

    tokens = _sentence.split()
    #tokens = tokens + data[idx].split()
    new_string = str()

    ADD_STRING = True
    for t in tokens:
        try:
            t.encode('utf-8')

            new_string = new_string +  t.encode('utf-8')+ ' '

        except:
            bad_ids.append(idx)
            ADD_STRING = False
            break
    if ADD_STRING:
        new_data.append(new_string)
        good_ids.append(idx);


# Now that we have the good files let's play with them

final_df =new_df.loc[:].drop(bad_ids)
data = list(new_data)
size =len(final_df);
new_index = [i for i in range(size)]
final_df.index = new_index
print("total number of podcast: " + str(len(new_data)))

#data = new_df.content[:50].values.tolist()


# Remove Emails
new_data = []
for idx,sentence in enumerate(data):
    new_sentence=str()
    try:
        
        new_sentence =  re.sub('\S*@\S*\s?', '', sentence)
        
    except:
         continue;
            
    try:
        
        new_sentence =  re.sub('\s+', ' ', new_sentence)
        
    except:
         continue;   
    
    try:
        
        new_sentence =  re.sub("\'", "", new_sentence)
        
    except:
         continue;

    new_data.append(new_sentence)


# Note: the data list now contains the description of the podcast!


data = new_data
final_df.content = data
#print len(data)
#print data[:10]

# This is where we use TF-IDF and KNN to recommend podcasts based on tweeter activity
# Why TF_IDF?  Because of the following reasons:

#- Is fast
#- we can use all podcast
#- it transform data into normalized vectors, perfect for cosine similarity



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)


y = [int(1) for i in range(len(data))]
tfidf_train = tfidf_vectorizer.fit_transform(data)

def get_from_tfidf(user ='kanyewest'):
    user_tweets = pd.read_csv(user + '_tweets.csv')
    user_tweets = user_tweets.text.values[:10]
    
    print(user_tweets)
    all_X_test = tfidf_vectorizer.transform(user_tweets)
    
    weights = range(1,len(all_X_test.A));
    weights = sorted(weights,reverse=True)
    

    X_test = tfidf_vectorizer.transform(user_tweets)
    final_vect = X_test[0]
    
    podcast_descriptions = []
    final_urls = []
    
    
    for idx,vec_X in enumerate(X_test[:]):
        
    
       # weight = weights[idx]

        final_vect = vec_X + final_vect
    
        a= np.dot(vec_X,tfidf_train.T)
        a_sort = np.argsort(a.A)
        max_vals_idx =  a_sort.flatten()[::-1][:10]
        
        
        for idx in max_vals_idx:
            podcast_descriptions.append(data[idx])
          
        urls  = final_df.loc[max_vals_idx].website_url.values.tolist()
        final_urls = final_urls + random.sample(urls,5)
        
    
    return random.sample(final_urls,3),podcast_descriptions , max_vals_idx;

tfidf_urls, podcast_descriptions, max_vals_idx = get_from_tfidf('BarackObama')
# Now let's use TF-IDF to:

#(1) transform the data

#(2) apply dot product (cosine similarity) operations for recommendation




#data = [u'hello I love you more hi bye',u'hello I love you more hi bye say ', u'hello I love you more hi bye class class',u'create a python sql cpp',u'contact your mom dad son']


#print(podcast_descriptions)
# Let's look at the URLs!

def top_tfidf_feats(row, features, top_n=25):

    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df
def top_feats_in_doc(Xtr, features, row_id, top_n=25):

    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def get_from_listen_notes(descriptions):
    Xtr = tfidf_vectorizer.transform(descriptions)
    features = tfidf_vectorizer.get_feature_names()

    buzz_words = []
    for i in range(30):
        a = top_feats_in_doc(Xtr, features, row_id=i, top_n=50)
        word = a.feature.values.tolist()[0]
        buzz_words.append(word)

    final_string = set(buzz_words)
    buzz_words = list(final_string)
    final_string = random.choice(buzz_words)
    for i in range(1):

        trial_word = random.choice(buzz_words)
        final_string = final_string + "+" + trial_word
    print(final_string)
    response = unirest.get("https://listennotes.p.mashape.com/api/v1/search?genre_ids=68%2C100&language=English&len_max=60&len_min=2&offset=0&only_in=Only+search+in+these+fields&published_after=1390190241000&published_before=1490190241000&q=" + final_string + "&sort_by_date=0&type=episode",
      headers={
        "X-Mashape-Key": "2i4tvJvbnFmshAM8KCc4CZaPh7ejp1ZzKA0jsnVI8C398lRK4S",
        "Accept": "application/json"
      }
    )

    listen_notes_pods = response.body['results']


    response_listen_notes=[]
    if len(listen_notes_pods)> 0:
        rand_pod = random.choice(listen_notes_pods)
        response_listen_notes.append(rand_pod[u'audio'])

    return response_listen_notes

a = get_from_listen_notes(podcast_descriptions)

def split_hashtag(hashtagestring):
    fo = re.compile(r'#[A-Z]{2,}(?![a-z])|[A-Z][a-z]+')
    fi = fo.findall(hashtagestring)
    result = ''
    for var in fi:
        result += var + ' '
        #print (result)
    result = result.split()
    return result;

def get_from_hashtag(all_hashtags):
    all_bi_tokens = []
    for h_tag in all_hashtags:

        trial_bitoken = split_hashtag(h_tag)
       # print(trial_bitoken)
        if len(trial_bitoken) > 1:
            all_bi_tokens.append(trial_bitoken)

    hash_signal = random.choice(all_bi_tokens)
    final_hash_signal = hash_signal[0]
    h_tag_user = "@" + hash_signal[0]



    Celebrity = True;
    while Celebrity:
        for i in range(1,len(hash_signal)):
            final_hash_signal = final_hash_signal + "+" + hash_signal[i]
            h_tag_user = h_tag_user + hash_signal[i]

        try:
            user = api.get_user(h_tag_user)

        except tweepy.error.TweepError as e:
            time.sleep(180)
            user = api.get_user(h_tag_user)
        try:

            nbr_friends = user.followers_count


        except tweepy.error.TweepError as e:
            time.sleep(180)
            nbr_friends = user.followers_count

        if nbr_friends > 1000000:
            Celebrity = False;

        else:
            hash_signal = random.choice(all_bi_tokens)
            final_hash_signal = hash_signal[0]
            h_tag_user = "@" + hash_signal[0]

    response = unirest.get("https://listennotes.p.mashape.com/api/v1/search?genre_ids=68%2C110&language=English&len_max=50&len_min=2&offset=0&only_in=Only+search+in+these+fields&published_after=1390190241000&published_before=1490190241000&q=" + final_hash_signal + "&sort_by_date=0&type=episode",
    headers={
    "X-Mashape-Key": "2i4tvJvbnFmshAM8KCc4CZaPh7ejp1ZzKA0jsnVI8C398lRK4S",
    "Accept": "application/json"
              }
            )

    listen_notes_pods = response.body['results']


    response_listen_notes=[]
    if len(listen_notes_pods)> 0:
        rand_pod = random.choice(listen_notes_pods)
      #  print(rand_pod)
        response_listen_notes.append(rand_pod[u'audio'])


    return response_listen_notes, h_tag_user,final_hash_signal

url_hashtag,b,c = get_from_hashtag(["BarackObama"] )


def get_user_tweets(screen_name):
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    try:
        user = api.get_user(screen_name)
    except tweepy.error.TweepError as e:
        time.sleep(180)
        user = api.get_user(screen_name)
    try:
        friends = user.friends()

    except tweepy.error.TweepError as e:
        time.sleep(180)
        friends = user.friends()
    #print(user.screen_name)
    all_friends = []
    for friend in friends:
        all_friends.append(friend.screen_name)

    alltweets = []
    if len(new_tweets) < 1:

        return []
    #save most recent tweets
    alltweets.extend(new_tweets)

    #save the id of the oldest tweet less one

    if len(new_tweets) > 0:

        oldest = alltweets[-1].id - 1

    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        #print "getting tweets before %s" % (oldest)

    #all subsiquent requests use the max_id param to prevent duplicates
        try:
            new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)

        except tweepy.error.TweepError as e:
            time.sleep(180)
            new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)


    #save most recent tweets
        alltweets.extend(new_tweets)

        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        #print(oldest)

        #print "...%s tweets downloaded so far" % (len(alltweets))

        #transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
    return all_friends,outtweets

def clean_weird_chars(stream_data):
    new_data = []
    for idx,sentence in enumerate(stream_data):
        new_sentence=str()
        try:

            new_sentence =  re.sub('\S*@\S*\s?', '', sentence)

        except:
             continue;

        try:

            new_sentence =  re.sub('\s+', ' ', new_sentence)

        except:
             continue;

        try:

            new_sentence =  re.sub("\'", "", new_sentence)

        except:
             continue;
        try:
            re.sub(r'^https?:\/\/.*[\r\n]*', '', new_sentence, flags=re.MULTILINE)

        except:
             continue;

        new_data.append(new_sentence)

    return new_data;

def clean_asian(streamdata):
    bad_ids= []
    good_ids = []
    for idx,_sentence in enumerate(streamdata):

        if isinstance(_sentence,float):
            bad_ids.append(idx)
            continue

        tokens = _sentence.split()
        new_string = str()

        ADD_STRING = True
        for t in tokens:
            try:
                t.encode('utf-8')

                new_string = new_string +  t.encode('utf-8')+ ' '

            except:
                bad_ids.append(idx)
                ADD_STRING = False
                break
        if ADD_STRING:
            new_data.append(new_string)
            good_ids.append(idx);

    return good_ids,bad_ids;

last_podcast = clean_weird_chars(podcast_descriptions)

def sent_to_words(sentences):
    for sentence in sentences:
        
        try:
            a = gensim.utils.simple_preprocess(str(sentence), deacc=True)
            
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        except:
            
            continue
            

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def lda_words(descriptions,nbr_words=2):
    last_podcast = clean_weird_chars(podcast_descriptions)
    data_words = list(sent_to_words(last_podcast))
    
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    #print(trigram_mod[bigram_mod[data_words[0]]])
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en')

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
        #Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=1, 
                                           random_state=400,
                                           update_every=100,
                                           chunksize=10,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    
    top_words_per_topic = []
    for t in range(lda_model.num_topics):
        top_words_per_topic.extend([(t, ) + x for x in lda_model.show_topic(t, topn = 20)])

        pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("top_words.csv")
        
    #top_words_lda = pd.read_csv('top_words.csv')
    
    
    string_candidates =np.array(top_words_per_topic)[:nbr_words,1]
    final_string = string_candidates[0] + "+" + string_candidates[1]
    response = unirest.get("https://listennotes.p.mashape.com/api/v1/search?genre_ids=68%2C100&language=English&len_max=60&len_min=2&offset=0&only_in=Only+search+in+these+fields&published_after=1390190241000&published_before=1490190241000&q=" + final_string + "&sort_by_date=0&type=episode",
      headers={
        "X-Mashape-Key": "2i4tvJvbnFmshAM8KCc4CZaPh7ejp1ZzKA0jsnVI8C398lRK4S",
        "Accept": "application/json"
      }
    )

    listen_notes_pods = response.body['results']


    response_listen_notes=[]
    if len(listen_notes_pods)> 0:
        rand_pod = random.choice(listen_notes_pods)
        response_listen_notes.append(rand_pod[u'audio'])
    
    
    return response_listen_notes



def listen_notes_favorite():
    # These code snippets use an open-source library. http://unirest.io/python
    response = unirest.get("https://listennotes.p.mashape.com/api/v1/best_podcasts?page=1",
              headers={
        "X-Mashape-Key": "2i4tvJvbnFmshAM8KCc4CZaPh7ejp1ZzKA0jsnVI8C398lRK4S",
    "Accept": "application/json"
        }
    )

    listen_notes_pods = response.body['channels']
   # print(listen_notes_pods[1]['website'])
    response_listen_notes=[]
    if len(listen_notes_pods) > 5:

        for i in range(3):
            response_listen_notes.append(listen_notes_pods[i][u'website'])

    else:

        for i in range(len(listen_notes_pods)):
            response_listen_notes.append(listen_notes_pods[i][u'website'])

    url_string = str();
    for i in response_listen_notes:
        url_string = url_string + " " + str(i)

    return url_string

#Get history of tweets 
#public_tweets = api.home_timeline()
#podcast_url = tfidf_urls[0] + " "+ tfidf_urls[1] + " "  +tfidf_urls[2] + " " + url_listen_notes[0] + " " + url_hashtag[0]
#podcast_url = url_hashtag[0]
try:
    public_tweets = api.mentions_timeline()
except tweepy.error.TweepError as e:
    time.sleep(180)
    public_tweets = api.mentions_timeline()
    
print(len(public_tweets))
#look at first tweet only for now
idx= 0 ;

print("***CREATING TWEETBOT")
#loop through twwets in timeline
for i in range(10):
    print("loop: " + str(i))
    for idx,tweet in enumerate(public_tweets):
       
        #print("found tweet :" + str(idx))
        print(tweet.text)
        #break on other tweets
        if idx >0:
            break;

        #split tweet to get username
        try:
            buffer_str =str(tweet.text).split()
        except tweepy.error.TweepError as e:
            time.sleep(180)
            buffer_str =str(tweet.text).split()

        #get into first tweet
        if idx >= 0:
                #m = buffer_str[0] + " podcast"
                #print(m)
                #print(tweet.id)
                #print('tweet content : ' + str(tweet.text))

                
                #user info
                sn = str()
                try:
                    #user = api.get_user(tweet.id)
                    #user screen name
                    sn = tweet.user.screen_name
                except tweepy.error.TweepError as e:
                    time.sleep(180)
                    
                    #user screen name
                    sn = tweet.user.screen_name
                print(sn)
                    
                #print("sending recommendation to @" + str(sn))

                #get all twitter history from user
                #all_tweets = get_user_tweets(sn)
                #print("getting obama tweets")
                
                try:
                    all_tweets = get_user_tweets('BarackObama')
                except tweepy.error.TweepError as e:
                    time.sleep(180)
                    all_tweets = get_user_tweets('BarackObama')
                
                tfidf_urls, podcast_descriptions, max_vals_idx = get_from_tfidf('BarackObama');
                
                tfidf_buzz_words_urls = get_from_listen_notes(podcast_descriptions)
                #lda_url = lda_words(descriptions,nbr_words=2)
                
                podcast_url = tfidf_urls[0] + " " + tfidf_urls[1] + " " +  tfidf_urls[2] + " "
                if len(tfidf_buzz_words_urls) > 0:
                    podcast_url = podcast_url  + tfidf_buzz_words_urls[0]
                #url_hashtag,b,url_signal = get_from_hashtag(["BarackObama"] )
                
                m = "@" + sn + " listen to "  + podcast_url
                #m = u"@" +sn + u" haha " + u'https://itunes.apple.com/us/podcast/ted-talks-daily/id160904630?mt=2&uo=4'
                
               
                print(m)
                #ok now notify user , case 1  : user has no much history
                #all_tweets =range(15)
                continue;
                break;
                if sn != "PFather101" and len(all_tweets) < 10:
                    try:
                        favorite_podcast =listen_notes_favorite()
                        m = "@" + sn + " listen to "  + favorite_podcast
                        s = api.update_status(m, tweet.id)
                    except tweepy.error.TweepError as e:
                        time.sleep(180)
                        s = api.update_status(m, tweet.id)
                    
                    print("reply sent going to sleep")
                    #time.sleep(1*60)
                    print("awake")

                #case 2 : user has history
                if sn != "PFather101" and len(all_tweets) > 10:
                    try:
                        s = api.update_status(m, tweet.id)
                    except tweepy.error.TweepError as e:
                        time.sleep(180)
                        s = api.update_status(m, tweet.id)  
                    print("reply sent to @" + str(sn))
                    print("reply sent going to sleep")
                    #time.sleep(1*60)
                    print("awake")
        idx= idx+1
    print("going to sleep") 
    time.sleep(5*60) 
    time.sleep(5*60) 
    time.sleep(5*60) 

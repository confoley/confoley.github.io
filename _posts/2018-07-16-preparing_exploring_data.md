---
layout: post
title: Part 1, Classifying Political News Media Text with Natural Language Processing
date: 2018-07-16
---
For my final capstone project as part of the Data Science Immersive program at General Assembly NYC, I decided to create a political text classification model using Natural Language Processing. As with all data science projects, this was a non-linear, iterative process that required extensive data cleaning. I learned a great deal about NLP in the process, and I look forward to further projects on this topic. The full repo for this project can be found [here](https://github.com/confoley/Capstone). 


**Purpose**

In 2018, I don't think I need to remind anyone of the current news media landscape in the United States. Needless to say, the media culture is extremely polarized and the societal effects of how people source their information is becoming increasingly evident. It hasn’t always been this way, and it is important to be able to know what kind of source a piece of political news text is coming from before deciding what to do with it. Political Science is often defined as the study of power. In the era of Trump, the power of language is increasingly relevant and important. **In this project, I aim to build a model that predicts whether or not political media text is right wing or not right wing.** I could have chosen to call these labels “right” and “left”, but I chose more of a "one vs. rest" terminology to reflect the fact that the political spectrum isn't as simple as Right and Left. [This website](http://www.allgeneralizationsarefalse.com/) has a great "media bias chart" for visualization. There is more to say about far right news media entering the mainstream, but we can save that for another day. Furthermore, this project is a stepping stone to future projects using multi class targets that more accurately reflect the kinds of political news media that are out there. Likewise, it is a stepping stone to more intimidating and complex projects that tackle social media language as well. 

### Gathering and Load Data

I collected data for 36,854 unique articles from a variety of right wing and not right wing sources. The features collected were title, description, date posted, url, author, and source. Choosing obviously partisan sources, I labeled whether they are right wing or not based on the source. Therefore, since I would only be starting with title and description as features, a challenge I faced is making sure that any giveaway of the source is removed from an article’s text features. All of my data was scraped from [News API](https://newsapi.org/) except for Infowars and Democracy Now, for which I used BeautifulSoup to scrape article information from their respective RSS feeds.

My right wing sources included:
- Fox News
- Breitbart News
- National Review
- Infowars

Not right wing sources:
- MSNBC
- Huffington Post
- Vice News
- CNN
- Democracy Now!

I scraped the data in separate notebooks and then loaded in all of the individual .csv files into my main notebook. The notebooks for this data collection process can be found in the main folder of the GitHub repo. 

### Clean Data
The initial data cleaning was quite simple and straightforward compared to the iterative process of removing noisy n-grams that would come once I began exploring and modeling. I dropped all of the rows with missing titles or descriptions, as those are the only features I would start with in my model. 


```python
# Check for missing values
text.isna().sum()
```




    author         9466
    description      69
    publishedAt       0
    source            0
    title             1
    url               0
    yes_right         0
    dtype: int64




```python
# Dropped all rows with missing text, as that is all I will be using as features
text.dropna(subset=['description','title'], inplace=True)
```


```python
# It's okay if there is no author
text.fillna('no_author', inplace=True)
```


```python
text.shape
```




    (36854, 7)




```python
text.yes_right.value_counts()[0]/len(text.yes_right) 
# Baseline accuracy is 53.5%, almost a 50/50 split in target
# There are a bit more not right sources than right
```




    0.53470450968687255



It's barely worth including Infowars and Democracy Now in the model as it would have taken quite a long time to gather a comparable amount of articles as those from News API by scraping from an RSS feed. However, I got as many articles as I could over a week’s time.


```python
text.source.value_counts()
# Democracy Now and Infowars don't produce quite as much content...
```




    fox news           6381
    national review    5348
    huffington post    4950
    msnbc              4950
    vice news          4949
    breitbart          4948
    cnn                4776
    infowars            471
    democracy now        81
    Name: source, dtype: int64



**Removing Source Giveaways from the Title and Description**

There was a multi-step process to remove source giveaways from the articles' text features. Here I create a function to use in the `.apply()` function to remove any mention of any source from all of the title and description text, as well as remove that specific article's author from the text. This cleaned up a lot of source giveaways, but there was still much that slipped through the cracks.


```python
# Many CNN articles had "CNN" after the author's name
text['author'] = [a.replace(', CNN', '') for a in text['author']]
```


```python
def remove_source_info(row):
    sources = ['Breitbart', 'CNN', 'Fox News', 'National Review', 'Vice News',
               'Democracy Now', 'Infowars', 'MSNBC', 'Huffington Post']
    for source in sources:
        row['description'] = row['description'].replace(source, '')
        row['title'] = row['title'].replace(source, '')
    row['title'] = row['title'].replace(row['author'], '')
    row['description'] = row['description'].replace(row['author'], '')
    return row
```


```python
text = text.apply(remove_source_info, axis=1)
```


```python
text.query("source=='Fox News'").head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author</th>
      <th>description</th>
      <th>publishedAt</th>
      <th>source</th>
      <th>title</th>
      <th>url</th>
      <th>yes_right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9724</th>
      <td>no_author</td>
      <td>Judge Napolitano and Marie Harf discuss the Tr...</td>
      <td>2018-06-20T18:51:36Z</td>
      <td>Fox News</td>
      <td>Freedom Watch: Napolitano and Harf dig in on i...</td>
      <td>http://video.foxnews.com/v/5799861947001/</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9725</th>
      <td>no_author</td>
      <td>Find out where and who caught a rare cotton ca...</td>
      <td>2018-06-20T18:50:00Z</td>
      <td>Fox News</td>
      <td>See it: Cotton candy-colored lobster caught</td>
      <td>http://video.foxnews.com/v/5799865520001/</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9726</th>
      <td>no_author</td>
      <td>Steve Harrigan reports from outside a detentio...</td>
      <td>2018-06-20T18:47:58Z</td>
      <td>Fox News</td>
      <td>Media not given access to 'tender age' shelters</td>
      <td>http://video.foxnews.com/v/5799862990001/</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9727</th>
      <td>no_author</td>
      <td>Reports: More than 2,000 children have been se...</td>
      <td>2018-06-20T18:47:54Z</td>
      <td>Fox News</td>
      <td>Critics denounce 'tender age' shelters</td>
      <td>http://video.foxnews.com/v/5799860019001/</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9728</th>
      <td>no_author</td>
      <td>Nearly a year after the body of little boy was...</td>
      <td>2018-06-20T18:38:34Z</td>
      <td>Fox News</td>
      <td>‘Little Jacob’ has been identified</td>
      <td>http://video.foxnews.com/v/5799856889001/</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### NLP Feature Engineering

Using NLP tools from the TextBlob library, I tagged each word of the title and description with a part of speech and added the normalized value counts for each part of speech as a new feature. The meaning of those parts of speech tags can be found [here](https://www.clips.uantwerpen.be/pages/mbsp-tags). I also used TextBlob’s sentiment analysis scoring to create features for polarity (positive or negative) and subjectivity. I then took the difference between title polarity and description polarity as well as the difference between title subjectivity and description subjectivity to measure any possible discrepancy in sentiment between title and description. Additionally, I created an average title word length feature. I could have spent weeks just engineering features using NLP tools, but this seemed like a good base to start with for this project.


```python
text['combined'] = text.title + ' ' + text.description # all text together
```

For creation of features, I treated title and description separately, but for modeling with TfIdf I combined the two into one document for each row.

I created a feature of average word length in the title using a RegEx tokenizer.


```python
tokenizer = nltk.RegexpTokenizer(r'\w+')
title_tokens = [tokenizer.tokenize(w) for w in text.title]

avg_word_length = []
for title in title_tokens:
    wordlen = []
    for word in title:
        wordlen.append(len(word))
        if len(wordlen)==len(title):
            avg_word_length.append(np.sum(wordlen)/len(wordlen))

text['avg_word_len_title'] = avg_word_length
```

#### Sentiment Analysis

I used TextBlob to create polarity and subjectivity features for both the title and description.


```python
text['title_polarity'] = [TextBlob(w).sentiment.polarity for w in text.title]

text['title_subjectivity'] = [TextBlob(w).sentiment.subjectivity for w in text.title]

text['desc_polarity'] = [TextBlob(w).sentiment.polarity for w in text.description]

text['desc_subjectivity'] = [TextBlob(w).sentiment.subjectivity for w in text.description]

text['subj_difference'] = text['title_subjectivity'] - text['desc_subjectivity']
text['polarity_difference']  = text['title_polarity'] - text['desc_polarity']
```

#### Parts of Speech Tagging

Also using TextBlob, I tagged each word with a part of speech and then took the normalized value counts for each part of speech in the document. 


```python
title_tags = [TextBlob(w.lower(), tokenizer=tokenizer).tags for w in text.title]
```


```python
title_tags[0]
```




    [('peter', 'NN'),
     ('fonda', 'NN'),
     ('lying', 'VBG'),
     ('gash', 'JJ'),
     ('kirstjen', 'NNS'),
     ('nielsen', 'VBN'),
     ('should', 'MD'),
     ('be', 'VB'),
     ('whipped', 'VBN'),
     ('naked', 'JJ'),
     ('in', 'IN'),
     ('public', 'JJ')]




```python
tags_counts = []
for row in title_tags:
    tags = [n[1] for n in row]
    tags_counts.append(tags)
```


```python
title_parts_of_speech = []
for n in tags_counts:
    foo = dict(pd.Series(n).value_counts(normalize=True))
    title_parts_of_speech.append(foo)
```


```python
title_parts_of_speech = pd.DataFrame(title_parts_of_speech).fillna(0)
# A NaN value means that part of speech did not appear in the text
```


```python
# Label as title parts of speech
title_parts_of_speech.columns = [str(n) + '_title' for n in title_parts_of_speech.columns]
```

I then followed the same process for description.


```python
# Checking for correct length
desc_parts_of_speech.shape[0] == title_parts_of_speech.shape[0]
```




    True




```python
pos_tags = pd.concat([title_parts_of_speech, desc_parts_of_speech], axis=1)
```


```python
# Concatenate all created features together with target
df = pd.concat([text[['title_polarity', 'title_subjectivity', 'desc_polarity',
                      'desc_subjectivity', 'subj_difference', 'polarity_difference',
                      'avg_word_len_title']], pos_tags, text.yes_right], axis=1)
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title_polarity</th>
      <th>title_subjectivity</th>
      <th>desc_polarity</th>
      <th>desc_subjectivity</th>
      <th>subj_difference</th>
      <th>polarity_difference</th>
      <th>avg_word_len_title</th>
      <th>CC_title</th>
      <th>CD_title</th>
      <th>DT_title</th>
      <th>...</th>
      <th>VBD_desc</th>
      <th>VBG_desc</th>
      <th>VBN_desc</th>
      <th>VBP_desc</th>
      <th>VBZ_desc</th>
      <th>WDT_desc</th>
      <th>WP_desc</th>
      <th>WP$_desc</th>
      <th>WRB_desc</th>
      <th>yes_right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.50</td>
      <td>0.233333</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.233333</td>
      <td>0.000000</td>
      <td>5.166667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.068182</td>
      <td>0.045455</td>
      <td>0.022727</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.022727</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.25</td>
      <td>0.900000</td>
      <td>0.250000</td>
      <td>0.650000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>5.750000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.160000</td>
      <td>0.040000</td>
      <td>0.0</td>
      <td>0.04</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.40</td>
      <td>0.300000</td>
      <td>0.555417</td>
      <td>0.666667</td>
      <td>-0.366667</td>
      <td>-0.155417</td>
      <td>6.545455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.020000</td>
      <td>0.060000</td>
      <td>0.020000</td>
      <td>0.1</td>
      <td>0.04</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 79 columns</p>
</div>



Every created features is already scaled except for a few, which I just simply manually scaled.


```python
def scaled_checker(df):
    for col in df.columns:
        if max(df[col]) > 1:
            print(col)
        if min(df[col]) < 0:
            print(col)
        else:
            pass
        
scaled_checker(df)
```

    subj_difference
    polarity_difference
    avg_word_len_title



```python
df['subj_difference'] = (df['subj_difference'] - min(df['subj_difference'])) \n
    /(max(df['subj_difference'])-min(df['subj_difference']))
df['polarity_difference'] = (df['polarity_difference'] - min(df['polarity_difference']))/ \n
    (max(df['polarity_difference'])-min(df['polarity_difference']))
df['avg_word_len_title'] = (df['avg_word_len_title'] - min(df['avg_word_len_title']))/ \n
    (max(df['avg_word_len_title'])-min(df['avg_word_len_title']))

```


```python
scaled_checker(df) # numerical data is scaled
```


```python
# Save my mostly cleaned datasets
text.to_csv('./datasets/text2.csv')
df.to_csv('./datasets/df2.csv')
```

### Explore Data

Exploring the sentiment analysis features, there is not much significant to note. Correlations between features and the target are extremely weak. All distributions had the same shape, with all of polarity having a normal distribution and all description distributions being skewed right. In other words, most text was neutral in terms of polarity and most text was more objective than subjective. One difference is that there was more subjective language in the descriptions than in the titles. 


```python
num_corrs = df[['title_polarity','title_subjectivity','desc_polarity',
                'desc_subjectivity', 'subj_difference','polarity_difference',
                'avg_word_len_title','yes_right']].corr()
mask = np.zeros_like(num_corrs, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(15,8))
plt.title('Correlations Between Sentiment Analysis Features and Target', fontsize=20)
sns.heatmap(num_corrs, linewidths=0.5, mask=mask, annot=True);
# Very insignificant 
```


![png](/images/preparing_exploring_data_files/preparing_exploring_data_40_0.png)



```python
# Looking at features based on target
rightwing = df[df.yes_right==1]
notrightwing = df[df.yes_right==0]
```


```python
figure, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

rightwing.title_polarity.plot(kind='hist', ax=ax[0,1],
                              title='Title Polarity: Right')
notrightwing.title_polarity.plot(kind='hist', ax=ax[0,0],
                                 title='Title Polarity: Not Right')
notrightwing.desc_polarity.plot(kind='hist',
                                ax=ax[1,0], title='Description Polarity: Not Right')
rightwing.desc_polarity.plot(kind='hist',
                             ax=ax[1,1], title='Description Polarity: Right');
```


![png](/images/preparing_exploring_data_files/preparing_exploring_data_42_0.png)



```python
figure, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

rightwing.title_subjectivity.plot(kind='hist', ax=ax[0,1],
                                  title='Title Subjectivity: Right')
notrightwing.title_subjectivity.plot(kind='hist', ax=ax[0,0],
                                     title='Title Subjectivity: Not Right')
notrightwing.desc_subjectivity.plot(kind='hist', ax=ax[1,0],
                                    title='Description Subjectivity: Not Right')
rightwing.desc_subjectivity.plot(kind='hist', ax=ax[1,1],
                                 title='Description Subjectivity: Right');
```


![png](/images/preparing_exploring_data_files/preparing_exploring_data_43_0.png)


### Exploring Word Counts

In the following several sections, I used Count Vectorizer to find the highest count single-grams, bi-grams, tri-grams, and quad-grams in the titles and descriptions, sorted by both target classes. After initially looking at the counts and seeing a lot of source information and noise, I created a custom preprocessor that removes stop grams of noise and source information as well as lemmatize the text. I also created a custom group of stop words that contain source information and appended it to SciKitLearn’s list of English stop words, which I used in all of my Count Vectorizer and TfIdf models. The initial words counts demonstrated that there is much noise in the data, and TfIdf might be the best choice for a vectorizer as it will penalize terms that appear very frequently across documents. However, if some of these terms only appear frequently in only one unique source, I decided to remove them manually.  

There were a couple interesting interpretations, especially as the n-grams increased, but the most common n-grams in the documents were mostly similar between left and right until the tri-gram level, when noticeable differences began to form. 

**Set Up Vectorizer**


```python
lemmatizer = WordNetLemmatizer()

def my_preprocessor(doc):
    stop_grams = ['national review','National Review','Fox News','Content Uploads',
                  'content uploads','fox news', 'Associated Press','associated press',
                  'Fox amp Friends','Rachel Maddow','Morning Joe','morning joe',
                  'Breitbart News', 'fast facts', 'Fast Facts','Fox &', 'Fox & Friends',
                  'Ali Velshi','Stephanie Ruhle','Raw video', '& Friends', 'Ari Melber',
                  'amp Friends', 'Content uploads', 'Geraldo Rivera']
    for stop in stop_grams:
        doc = doc.replace(stop,'')
    return lemmatizer.lemmatize(doc.lower())
```


```python
cust_stop_words = ['CNN','cnn','amp','huffpost','fox','reuters','ap','vice','breitbart',
                   'nationalreview', 'www','msnbc','infowars', 'foxnews','Vice','Breitbart',
                   'National Review','Fox News','Reuters','Fast Facts','Infowars','Vice',
                   'MSNBC','www','AP','Huffpost','HuffPost','Maddow', '&', 'Ingraham', 'ingraham']
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
original_stopwords = list(ENGLISH_STOP_WORDS)
cust_stop_words += original_stopwords
```

#### Single Grams for Title


```python
right = text['yes_right'] == 1
notright = text['yes_right'] == 0
```


```python
# Separate title and description based on target class
right_title = text[right].title
right_desc = text[right].description
notright_title = text[notright].title
notright_desc = text[notright].description
```


```python
stem = PorterStemmer()
lemmatizer = WordNetLemmatizer()
```


```python
def get_top_grams(df_r, df_nr, n):
    cvec = CountVectorizer(preprocessor=my_preprocessor,
                           ngram_range=(n,n), stop_words=cust_stop_words, min_df=5)
    cvec.fit(df_r)
    word_counts = pd.DataFrame(cvec.transform(df_r).todense(),
                       columns=cvec.get_feature_names())
    counts = word_counts.sum(axis=0)
    counts = pd.DataFrame(counts.sort_values(ascending = False),columns=['right']).head(15)
    
    cvec = CountVectorizer(preprocessor=my_preprocessor,
                           ngram_range=(n,n), stop_words=cust_stop_words, min_df=5)
    cvec.fit(df_nr)
    word_counts2 = pd.DataFrame(cvec.transform(df_nr).todense(),
                         columns=cvec.get_feature_names())
    counts2 = word_counts2.sum(axis=0)
    counts2 = pd.DataFrame(counts2.sort_values(ascending = False),columns=['not right']).head(15)
    return counts, counts2
```


```python
r
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>trump</th>
      <td>2981</td>
    </tr>
    <tr>
      <th>report</th>
      <td>569</td>
    </tr>
    <tr>
      <th>north</th>
      <td>523</td>
    </tr>
    <tr>
      <th>korea</th>
      <td>502</td>
    </tr>
    <tr>
      <th>says</th>
      <td>481</td>
    </tr>
    <tr>
      <th>new</th>
      <td>453</td>
    </tr>
    <tr>
      <th>house</th>
      <td>419</td>
    </tr>
    <tr>
      <th>border</th>
      <td>419</td>
    </tr>
    <tr>
      <th>kim</th>
      <td>397</td>
    </tr>
    <tr>
      <th>summit</th>
      <td>390</td>
    </tr>
    <tr>
      <th>president</th>
      <td>386</td>
    </tr>
    <tr>
      <th>day</th>
      <td>375</td>
    </tr>
    <tr>
      <th>police</th>
      <td>366</td>
    </tr>
    <tr>
      <th>immigration</th>
      <td>340</td>
    </tr>
    <tr>
      <th>donald</th>
      <td>304</td>
    </tr>
  </tbody>
</table>
</div>




```python
nr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>not right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>trump</th>
      <td>3872</td>
    </tr>
    <tr>
      <th>new</th>
      <td>1043</td>
    </tr>
    <tr>
      <th>says</th>
      <td>585</td>
    </tr>
    <tr>
      <th>house</th>
      <td>471</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>430</td>
    </tr>
    <tr>
      <th>north</th>
      <td>414</td>
    </tr>
    <tr>
      <th>korea</th>
      <td>401</td>
    </tr>
    <tr>
      <th>cohen</th>
      <td>391</td>
    </tr>
    <tr>
      <th>mueller</th>
      <td>377</td>
    </tr>
    <tr>
      <th>white</th>
      <td>350</td>
    </tr>
    <tr>
      <th>kim</th>
      <td>347</td>
    </tr>
    <tr>
      <th>18</th>
      <td>339</td>
    </tr>
    <tr>
      <th>world</th>
      <td>338</td>
    </tr>
    <tr>
      <th>like</th>
      <td>333</td>
    </tr>
    <tr>
      <th>gop</th>
      <td>328</td>
    </tr>
  </tbody>
</table>
</div>



#### Single Grams for Description


```python
r, nr = get_top_grams(right_desc, notright_desc, 1)
```


```python
r
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>trump</th>
      <td>3830</td>
    </tr>
    <tr>
      <th>president</th>
      <td>3216</td>
    </tr>
    <tr>
      <th>new</th>
      <td>1479</td>
    </tr>
    <tr>
      <th>house</th>
      <td>1214</td>
    </tr>
    <tr>
      <th>said</th>
      <td>1205</td>
    </tr>
    <tr>
      <th>donald</th>
      <td>1111</td>
    </tr>
    <tr>
      <th>north</th>
      <td>938</td>
    </tr>
    <tr>
      <th>state</th>
      <td>909</td>
    </tr>
    <tr>
      <th>says</th>
      <td>846</td>
    </tr>
    <tr>
      <th>news</th>
      <td>760</td>
    </tr>
    <tr>
      <th>year</th>
      <td>735</td>
    </tr>
    <tr>
      <th>white</th>
      <td>734</td>
    </tr>
    <tr>
      <th>police</th>
      <td>703</td>
    </tr>
    <tr>
      <th>tuesday</th>
      <td>682</td>
    </tr>
    <tr>
      <th>people</th>
      <td>632</td>
    </tr>
  </tbody>
</table>
</div>




```python
nr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>not right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>trump</th>
      <td>4756</td>
    </tr>
    <tr>
      <th>president</th>
      <td>2799</td>
    </tr>
    <tr>
      <th>new</th>
      <td>2052</td>
    </tr>
    <tr>
      <th>donald</th>
      <td>1374</td>
    </tr>
    <tr>
      <th>said</th>
      <td>1093</td>
    </tr>
    <tr>
      <th>house</th>
      <td>1039</td>
    </tr>
    <tr>
      <th>says</th>
      <td>840</td>
    </tr>
    <tr>
      <th>people</th>
      <td>831</td>
    </tr>
    <tr>
      <th>discuss</th>
      <td>777</td>
    </tr>
    <tr>
      <th>white</th>
      <td>768</td>
    </tr>
    <tr>
      <th>north</th>
      <td>721</td>
    </tr>
    <tr>
      <th>michael</th>
      <td>710</td>
    </tr>
    <tr>
      <th>news</th>
      <td>687</td>
    </tr>
    <tr>
      <th>day</th>
      <td>623</td>
    </tr>
    <tr>
      <th>year</th>
      <td>584</td>
    </tr>
  </tbody>
</table>
</div>



#### BiGrams for Title


```python
r, nr = get_top_grams(right_title, notright_title, 2)
```


```python
r
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>north korea</th>
      <td>417</td>
    </tr>
    <tr>
      <th>donald trump</th>
      <td>303</td>
    </tr>
    <tr>
      <th>white house</th>
      <td>230</td>
    </tr>
    <tr>
      <th>kim jong</th>
      <td>182</td>
    </tr>
    <tr>
      <th>president trump</th>
      <td>181</td>
    </tr>
    <tr>
      <th>ig report</th>
      <td>141</td>
    </tr>
    <tr>
      <th>supreme court</th>
      <td>129</td>
    </tr>
    <tr>
      <th>trump kim</th>
      <td>121</td>
    </tr>
    <tr>
      <th>judicial activism</th>
      <td>104</td>
    </tr>
    <tr>
      <th>day liberal</th>
      <td>104</td>
    </tr>
    <tr>
      <th>liberal judicial</th>
      <td>104</td>
    </tr>
    <tr>
      <th>new york</th>
      <td>100</td>
    </tr>
    <tr>
      <th>year old</th>
      <td>91</td>
    </tr>
    <tr>
      <th>kim summit</th>
      <td>82</td>
    </tr>
    <tr>
      <th>cartoons day</th>
      <td>80</td>
    </tr>
  </tbody>
</table>
</div>




```python
nr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>not right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>north korea</th>
      <td>321</td>
    </tr>
    <tr>
      <th>white house</th>
      <td>247</td>
    </tr>
    <tr>
      <th>michael cohen</th>
      <td>196</td>
    </tr>
    <tr>
      <th>kim jong</th>
      <td>165</td>
    </tr>
    <tr>
      <th>donald trump</th>
      <td>160</td>
    </tr>
    <tr>
      <th>stormy daniels</th>
      <td>113</td>
    </tr>
    <tr>
      <th>world cup</th>
      <td>111</td>
    </tr>
    <tr>
      <th>royal wedding</th>
      <td>99</td>
    </tr>
    <tr>
      <th>scott pruitt</th>
      <td>98</td>
    </tr>
    <tr>
      <th>president trump</th>
      <td>96</td>
    </tr>
    <tr>
      <th>new york</th>
      <td>87</td>
    </tr>
    <tr>
      <th>korea summit</th>
      <td>86</td>
    </tr>
    <tr>
      <th>family separation</th>
      <td>85</td>
    </tr>
    <tr>
      <th>trump kim</th>
      <td>83</td>
    </tr>
    <tr>
      <th>kanye west</th>
      <td>77</td>
    </tr>
  </tbody>
</table>
</div>



#### BiGrams for Description


```python
r, nr = get_top_grams(right_desc, notright_desc, 2)
```


```python
r # Seeing some noise here 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>president trump</th>
      <td>1187</td>
    </tr>
    <tr>
      <th>donald trump</th>
      <td>1088</td>
    </tr>
    <tr>
      <th>president donald</th>
      <td>932</td>
    </tr>
    <tr>
      <th>white house</th>
      <td>621</td>
    </tr>
    <tr>
      <th>new york</th>
      <td>481</td>
    </tr>
    <tr>
      <th>kim jong</th>
      <td>474</td>
    </tr>
    <tr>
      <th>north korea</th>
      <td>473</td>
    </tr>
    <tr>
      <th>year old</th>
      <td>339</td>
    </tr>
    <tr>
      <th>united states</th>
      <td>314</td>
    </tr>
    <tr>
      <th>north korean</th>
      <td>313</td>
    </tr>
    <tr>
      <th>trump administration</th>
      <td>257</td>
    </tr>
    <tr>
      <th>wp com</th>
      <td>238</td>
    </tr>
    <tr>
      <th>content uploads</th>
      <td>238</td>
    </tr>
    <tr>
      <th>com com</th>
      <td>238</td>
    </tr>
    <tr>
      <th>wp content</th>
      <td>238</td>
    </tr>
  </tbody>
</table>
</div>




```python
nr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>not right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>donald trump</th>
      <td>1352</td>
    </tr>
    <tr>
      <th>president trump</th>
      <td>904</td>
    </tr>
    <tr>
      <th>president donald</th>
      <td>673</td>
    </tr>
    <tr>
      <th>white house</th>
      <td>640</td>
    </tr>
    <tr>
      <th>new york</th>
      <td>420</td>
    </tr>
    <tr>
      <th>kim jong</th>
      <td>414</td>
    </tr>
    <tr>
      <th>michael cohen</th>
      <td>405</td>
    </tr>
    <tr>
      <th>trump administration</th>
      <td>401</td>
    </tr>
    <tr>
      <th>north korea</th>
      <td>386</td>
    </tr>
    <tr>
      <th>north korean</th>
      <td>254</td>
    </tr>
    <tr>
      <th>special counsel</th>
      <td>236</td>
    </tr>
    <tr>
      <th>robert mueller</th>
      <td>233</td>
    </tr>
    <tr>
      <th>rudy giuliani</th>
      <td>224</td>
    </tr>
    <tr>
      <th>united states</th>
      <td>223</td>
    </tr>
    <tr>
      <th>stormy daniels</th>
      <td>213</td>
    </tr>
  </tbody>
</table>
</div>



#### TriGrams for Title


```python
r, nr = get_top_grams(right_title, notright_title, 3)
```


```python
r # Mentions of MS 13 Gang and "judicial activism"
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>liberal judicial activism</th>
      <td>104</td>
    </tr>
    <tr>
      <th>day liberal judicial</th>
      <td>104</td>
    </tr>
    <tr>
      <th>north korea summit</th>
      <td>75</td>
    </tr>
    <tr>
      <th>trump kim summit</th>
      <td>66</td>
    </tr>
    <tr>
      <th>new york times</th>
      <td>27</td>
    </tr>
    <tr>
      <th>border patrol agents</th>
      <td>26</td>
    </tr>
    <tr>
      <th>south china sea</th>
      <td>26</td>
    </tr>
    <tr>
      <th>trump kim jong</th>
      <td>23</td>
    </tr>
    <tr>
      <th>judicial activism june</th>
      <td>22</td>
    </tr>
    <tr>
      <th>cartoons day march</th>
      <td>21</td>
    </tr>
    <tr>
      <th>judicial activism march</th>
      <td>20</td>
    </tr>
    <tr>
      <th>ms 13 gang</th>
      <td>20</td>
    </tr>
    <tr>
      <th>judicial activism april</th>
      <td>19</td>
    </tr>
    <tr>
      <th>cartoons day april</th>
      <td>19</td>
    </tr>
    <tr>
      <th>judicial activism february</th>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
nr
# Interesting: NYT is not one of my sources
# Mentions of Stormy Daniels and immigration policy
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>not right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>north korea summit</th>
      <td>73</td>
    </tr>
    <tr>
      <th>trump kim summit</th>
      <td>37</td>
    </tr>
    <tr>
      <th>family separation policy</th>
      <td>29</td>
    </tr>
    <tr>
      <th>trump tower meeting</th>
      <td>28</td>
    </tr>
    <tr>
      <th>quickly catch day</th>
      <td>21</td>
    </tr>
    <tr>
      <th>catch day news</th>
      <td>21</td>
    </tr>
    <tr>
      <th>trump legal team</th>
      <td>21</td>
    </tr>
    <tr>
      <th>trump north korea</th>
      <td>20</td>
    </tr>
    <tr>
      <th>iran nuclear deal</th>
      <td>20</td>
    </tr>
    <tr>
      <th>stormy daniels lawyer</th>
      <td>19</td>
    </tr>
    <tr>
      <th>daily horoscope june</th>
      <td>19</td>
    </tr>
    <tr>
      <th>trump kim jong</th>
      <td>18</td>
    </tr>
    <tr>
      <th>mtp daily june</th>
      <td>17</td>
    </tr>
    <tr>
      <th>santa fe high</th>
      <td>17</td>
    </tr>
    <tr>
      <th>sarah huckabee sanders</th>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>



#### TriGrams for Description


```python
r, nr = get_top_grams(right_desc, notright_desc, 3)
```


```python
r # For some reason "content uploads" is still appearing. Source giveaway. 
# Also image noise data.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>president donald trump</th>
      <td>926</td>
    </tr>
    <tr>
      <th>wp com com</th>
      <td>238</td>
    </tr>
    <tr>
      <th>wp content uploads</th>
      <td>238</td>
    </tr>
    <tr>
      <th>com wp content</th>
      <td>238</td>
    </tr>
    <tr>
      <th>com com wp</th>
      <td>238</td>
    </tr>
    <tr>
      <th>content uploads 2018</th>
      <td>233</td>
    </tr>
    <tr>
      <th>jpg fit 1024</th>
      <td>218</td>
    </tr>
    <tr>
      <th>1024 2c597 ssl</th>
      <td>216</td>
    </tr>
    <tr>
      <th>fit 1024 2c597</th>
      <td>216</td>
    </tr>
    <tr>
      <th>dictator kim jong</th>
      <td>125</td>
    </tr>
    <tr>
      <th>new york times</th>
      <td>118</td>
    </tr>
    <tr>
      <th>special counsel robert</th>
      <td>110</td>
    </tr>
    <tr>
      <th>leader kim jong</th>
      <td>108</td>
    </tr>
    <tr>
      <th>counsel robert mueller</th>
      <td>108</td>
    </tr>
    <tr>
      <th>new york city</th>
      <td>102</td>
    </tr>
  </tbody>
</table>
</div>




```python
nr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>not right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>president donald trump</th>
      <td>672</td>
    </tr>
    <tr>
      <th>north korean leader</th>
      <td>158</td>
    </tr>
    <tr>
      <th>new york times</th>
      <td>155</td>
    </tr>
    <tr>
      <th>leader kim jong</th>
      <td>149</td>
    </tr>
    <tr>
      <th>korean leader kim</th>
      <td>133</td>
    </tr>
    <tr>
      <th>special counsel robert</th>
      <td>125</td>
    </tr>
    <tr>
      <th>counsel robert mueller</th>
      <td>123</td>
    </tr>
    <tr>
      <th>fbi director james</th>
      <td>81</td>
    </tr>
    <tr>
      <th>director james comey</th>
      <td>81</td>
    </tr>
    <tr>
      <th>joy reid panel</th>
      <td>81</td>
    </tr>
    <tr>
      <th>reid panel discuss</th>
      <td>74</td>
    </tr>
    <tr>
      <th>lawyer michael cohen</th>
      <td>69</td>
    </tr>
    <tr>
      <th>attorney general jeff</th>
      <td>64</td>
    </tr>
    <tr>
      <th>general jeff sessions</th>
      <td>64</td>
    </tr>
    <tr>
      <th>new york city</th>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>



#### QuadGrams for Title


```python
r, nr = get_top_grams(right_title, notright_title, 4)
```


```python
r
# Again, lost of "judicial activism" and MS 13 appears 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>day liberal judicial activism</th>
      <td>104</td>
    </tr>
    <tr>
      <th>liberal judicial activism june</th>
      <td>22</td>
    </tr>
    <tr>
      <th>liberal judicial activism march</th>
      <td>20</td>
    </tr>
    <tr>
      <th>liberal judicial activism february</th>
      <td>19</td>
    </tr>
    <tr>
      <th>liberal judicial activism april</th>
      <td>19</td>
    </tr>
    <tr>
      <th>things caught eye today</th>
      <td>11</td>
    </tr>
    <tr>
      <th>north korea kim jong</th>
      <td>9</td>
    </tr>
    <tr>
      <th>fashion notes melania trump</th>
      <td>9</td>
    </tr>
    <tr>
      <th>santa fe high school</th>
      <td>7</td>
    </tr>
    <tr>
      <th>new york attorney general</th>
      <td>7</td>
    </tr>
    <tr>
      <th>public sector union dues</th>
      <td>7</td>
    </tr>
    <tr>
      <th>ms 13 gang members</th>
      <td>7</td>
    </tr>
    <tr>
      <th>fashion designer kate spade</th>
      <td>7</td>
    </tr>
    <tr>
      <th>southern poverty law center</th>
      <td>7</td>
    </tr>
    <tr>
      <th>trump says north korea</th>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
nr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>not right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>quickly catch day news</th>
      <td>21</td>
    </tr>
    <tr>
      <th>santa fe high school</th>
      <td>15</td>
    </tr>
    <tr>
      <th>trump family separation policy</th>
      <td>14</td>
    </tr>
    <tr>
      <th>prince harry meghan markle</th>
      <td>13</td>
    </tr>
    <tr>
      <th>new albums heavy rotation</th>
      <td>11</td>
    </tr>
    <tr>
      <th>new shows netflix stream</th>
      <td>10</td>
    </tr>
    <tr>
      <th>watch hulu new week</th>
      <td>10</td>
    </tr>
    <tr>
      <th>watch amazon prime new</th>
      <td>10</td>
    </tr>
    <tr>
      <th>trump north korea summit</th>
      <td>10</td>
    </tr>
    <tr>
      <th>amazon prime new week</th>
      <td>10</td>
    </tr>
    <tr>
      <th>best new shows netflix</th>
      <td>10</td>
    </tr>
    <tr>
      <th>shows netflix stream right</th>
      <td>10</td>
    </tr>
    <tr>
      <th>ranking best new shows</th>
      <td>10</td>
    </tr>
    <tr>
      <th>funniest tweets women week</th>
      <td>9</td>
    </tr>
    <tr>
      <th>20 funniest tweets women</th>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



#### QuadGrams for Description


```python
r, nr = get_top_grams(right_desc, notright_desc, 4)
```


```python
r
# Very noisy
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>com wp content uploads</th>
      <td>238</td>
    </tr>
    <tr>
      <th>com com wp content</th>
      <td>238</td>
    </tr>
    <tr>
      <th>wp com com wp</th>
      <td>238</td>
    </tr>
    <tr>
      <th>wp content uploads 2018</th>
      <td>233</td>
    </tr>
    <tr>
      <th>fit 1024 2c597 ssl</th>
      <td>216</td>
    </tr>
    <tr>
      <th>jpg fit 1024 2c597</th>
      <td>209</td>
    </tr>
    <tr>
      <th>special counsel robert mueller</th>
      <td>108</td>
    </tr>
    <tr>
      <th>north korean dictator kim</th>
      <td>94</td>
    </tr>
    <tr>
      <th>korean dictator kim jong</th>
      <td>94</td>
    </tr>
    <tr>
      <th>north korean leader kim</th>
      <td>91</td>
    </tr>
    <tr>
      <th>korean leader kim jong</th>
      <td>90</td>
    </tr>
    <tr>
      <th>https i0 wp com</th>
      <td>83</td>
    </tr>
    <tr>
      <th>i0 wp com com</th>
      <td>83</td>
    </tr>
    <tr>
      <th>i2 wp com com</th>
      <td>78</td>
    </tr>
    <tr>
      <th>https i2 wp com</th>
      <td>78</td>
    </tr>
  </tbody>
</table>
</div>




```python
nr
# Less noise, but still source giveaways like "Chris Matthews". 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>not right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>north korean leader kim</th>
      <td>133</td>
    </tr>
    <tr>
      <th>korean leader kim jong</th>
      <td>132</td>
    </tr>
    <tr>
      <th>special counsel robert mueller</th>
      <td>123</td>
    </tr>
    <tr>
      <th>fbi director james comey</th>
      <td>81</td>
    </tr>
    <tr>
      <th>joy reid panel discuss</th>
      <td>74</td>
    </tr>
    <tr>
      <th>attorney general jeff sessions</th>
      <td>63</td>
    </tr>
    <tr>
      <th>discuss political news day</th>
      <td>56</td>
    </tr>
    <tr>
      <th>matthews panel guests discuss</th>
      <td>56</td>
    </tr>
    <tr>
      <th>panel guests discuss political</th>
      <td>56</td>
    </tr>
    <tr>
      <th>guests discuss political news</th>
      <td>56</td>
    </tr>
    <tr>
      <th>chris matthews panel guests</th>
      <td>56</td>
    </tr>
    <tr>
      <th>mika discuss big news</th>
      <td>55</td>
    </tr>
    <tr>
      <th>joe mika discuss big</th>
      <td>55</td>
    </tr>
    <tr>
      <th>discuss big news day</th>
      <td>55</td>
    </tr>
    <tr>
      <th>hayes discusses day political</th>
      <td>55</td>
    </tr>
  </tbody>
</table>
</div>



### Wordclouds

These provide a nice, fun visualization of words found in right wing and not right wing title text. The larger the word the more it appears. 


```python
plt.figure(figsize=(15,8))
wordcloud = WordCloud(font_path='/Library/Fonts/Verdana.ttf',
                      max_words=(100),
                      width=2000, height=1000,
                      relative_scaling = 0.5,
                      background_color='white',
                      colormap='Dark2'
                  
).generate(' '.join(text[right].title))
plt.imshow(wordcloud)
plt.title("Right Wing Title Text", fontsize=24)
plt.axis("off")
plt.show()
```


![png](/images/preparing_exploring_data_files/preparing_exploring_data_84_0.png)



```python
plt.figure(figsize=(15,8))
wordcloud = WordCloud(font_path='/Library/Fonts/Verdana.ttf',
                      max_words=(100),
                      width=2000, height=1000,
                      relative_scaling = 0.5,
                      background_color='white',
                      colormap='Dark2'
                  
).generate(' '.join(text[notright].title))
plt.imshow(wordcloud)
plt.title("Not Right Wing Title Text", fontsize=24)
plt.axis("off")
plt.show()
```


![png](/images/preparing_exploring_data_files/preparing_exploring_data_85_0.png)


In the next post I will go over the modeling process and evaluation (the fun part!) of this project. 

To be continued...

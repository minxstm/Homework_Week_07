

```python
# Dependencies

import numpy as np
import pandas as pd
import json
import tweepy
import matplotlib.pyplot as plt
import seaborn as sns
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Twitter API Keys
consumer_key = 'ARVovD6xUGU2Aaa2xQe46eHbp'
consumer_secret = 'W08pv4OJYV740ha5hSGTe4QsXvVNJBeGHNf90OtwY3WQWjoviW'
access_token = '728085680492937217-1mBV8HZXZ5mY7yhGDgIwkOXRjRk2kuU'
access_token_secret = 'QNCmmZVCXmfz6nnvua1xV1caMtRw0y1vz1NF8btBFRKZw'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

target_user = ("@BBCWorld","@CBSNews","@CNN","@FoxNews","@nytimes")

counter = 1
sentiments = []

for user in target_user:
    
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []
    
    for x in range(5):
        
        public_tweets = api.user_timeline(user, count=20, result_type="recent")
            
        for tweet in public_tweets:   
        
            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]
            tweets_ago = counter

            
            sentiments.append({"User": user,
                                "Date": tweet["created_at"], 
                               "Compound": compound,
                               "Positive": pos,
                               "Negative": neu,
                               "Neutral": neg,
                               "Tweets Ago": counter})
            
            counter = counter + 1
```


```python
# Convert to DataFrame
sentiments_pd = pd.DataFrame.from_dict(sentiments)
sentiments_pd
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
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweets Ago</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>Wed Mar 28 04:18:40 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0000</td>
      <td>Wed Mar 28 04:18:40 +0000 2018</td>
      <td>0.645</td>
      <td>0.177</td>
      <td>0.177</td>
      <td>2</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.5719</td>
      <td>Wed Mar 28 04:18:40 +0000 2018</td>
      <td>0.730</td>
      <td>0.000</td>
      <td>0.270</td>
      <td>3</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.7783</td>
      <td>Wed Mar 28 04:18:40 +0000 2018</td>
      <td>0.473</td>
      <td>0.527</td>
      <td>0.000</td>
      <td>4</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>Wed Mar 28 04:05:47 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>5</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.4019</td>
      <td>Wed Mar 28 02:14:16 +0000 2018</td>
      <td>0.881</td>
      <td>0.000</td>
      <td>0.119</td>
      <td>6</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.4019</td>
      <td>Wed Mar 28 01:44:28 +0000 2018</td>
      <td>0.891</td>
      <td>0.109</td>
      <td>0.000</td>
      <td>7</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0000</td>
      <td>Wed Mar 28 01:34:04 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>8</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.0772</td>
      <td>Wed Mar 28 01:34:04 +0000 2018</td>
      <td>0.822</td>
      <td>0.178</td>
      <td>0.000</td>
      <td>9</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0000</td>
      <td>Wed Mar 28 00:41:30 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>10</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.2732</td>
      <td>Wed Mar 28 00:33:33 +0000 2018</td>
      <td>0.792</td>
      <td>0.000</td>
      <td>0.208</td>
      <td>11</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.4404</td>
      <td>Wed Mar 28 00:16:30 +0000 2018</td>
      <td>0.804</td>
      <td>0.000</td>
      <td>0.196</td>
      <td>12</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.5859</td>
      <td>Wed Mar 28 00:04:15 +0000 2018</td>
      <td>0.817</td>
      <td>0.000</td>
      <td>0.183</td>
      <td>13</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0000</td>
      <td>Tue Mar 27 23:50:38 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>14</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-0.7841</td>
      <td>Tue Mar 27 23:50:38 +0000 2018</td>
      <td>0.537</td>
      <td>0.463</td>
      <td>0.000</td>
      <td>15</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.5095</td>
      <td>Tue Mar 27 21:37:00 +0000 2018</td>
      <td>0.798</td>
      <td>0.000</td>
      <td>0.202</td>
      <td>16</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0000</td>
      <td>Tue Mar 27 21:20:33 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>17</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-0.2516</td>
      <td>Tue Mar 27 20:14:24 +0000 2018</td>
      <td>0.603</td>
      <td>0.229</td>
      <td>0.168</td>
      <td>18</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0000</td>
      <td>Tue Mar 27 19:19:12 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>19</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.6705</td>
      <td>Tue Mar 27 18:14:45 +0000 2018</td>
      <td>0.538</td>
      <td>0.130</td>
      <td>0.332</td>
      <td>20</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.0000</td>
      <td>Wed Mar 28 04:18:40 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>21</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.0000</td>
      <td>Wed Mar 28 04:18:40 +0000 2018</td>
      <td>0.645</td>
      <td>0.177</td>
      <td>0.177</td>
      <td>22</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.5719</td>
      <td>Wed Mar 28 04:18:40 +0000 2018</td>
      <td>0.730</td>
      <td>0.000</td>
      <td>0.270</td>
      <td>23</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-0.7783</td>
      <td>Wed Mar 28 04:18:40 +0000 2018</td>
      <td>0.473</td>
      <td>0.527</td>
      <td>0.000</td>
      <td>24</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.0000</td>
      <td>Wed Mar 28 04:05:47 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>25</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.4019</td>
      <td>Wed Mar 28 02:14:16 +0000 2018</td>
      <td>0.881</td>
      <td>0.000</td>
      <td>0.119</td>
      <td>26</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-0.4019</td>
      <td>Wed Mar 28 01:44:28 +0000 2018</td>
      <td>0.891</td>
      <td>0.109</td>
      <td>0.000</td>
      <td>27</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0000</td>
      <td>Wed Mar 28 01:34:04 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>28</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-0.0772</td>
      <td>Wed Mar 28 01:34:04 +0000 2018</td>
      <td>0.822</td>
      <td>0.178</td>
      <td>0.000</td>
      <td>29</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.0000</td>
      <td>Wed Mar 28 00:41:30 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>30</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>0.0258</td>
      <td>Wed Mar 28 02:26:03 +0000 2018</td>
      <td>0.823</td>
      <td>0.086</td>
      <td>0.091</td>
      <td>471</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>471</th>
      <td>0.4939</td>
      <td>Wed Mar 28 02:17:02 +0000 2018</td>
      <td>0.789</td>
      <td>0.000</td>
      <td>0.211</td>
      <td>472</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>472</th>
      <td>0.0000</td>
      <td>Wed Mar 28 02:02:02 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>473</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>473</th>
      <td>0.4404</td>
      <td>Wed Mar 28 01:51:02 +0000 2018</td>
      <td>0.734</td>
      <td>0.000</td>
      <td>0.266</td>
      <td>474</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>474</th>
      <td>0.0000</td>
      <td>Wed Mar 28 01:41:04 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>475</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>475</th>
      <td>-0.8860</td>
      <td>Wed Mar 28 01:32:09 +0000 2018</td>
      <td>0.691</td>
      <td>0.309</td>
      <td>0.000</td>
      <td>476</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>476</th>
      <td>0.0000</td>
      <td>Wed Mar 28 01:17:05 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>477</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>477</th>
      <td>0.4019</td>
      <td>Wed Mar 28 01:10:50 +0000 2018</td>
      <td>0.876</td>
      <td>0.000</td>
      <td>0.124</td>
      <td>478</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>478</th>
      <td>0.0000</td>
      <td>Wed Mar 28 01:02:06 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>479</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>479</th>
      <td>-0.4215</td>
      <td>Wed Mar 28 00:51:06 +0000 2018</td>
      <td>0.865</td>
      <td>0.135</td>
      <td>0.000</td>
      <td>480</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>480</th>
      <td>0.0000</td>
      <td>Wed Mar 28 04:21:03 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>481</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>481</th>
      <td>0.0000</td>
      <td>Wed Mar 28 04:11:06 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>482</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>482</th>
      <td>0.0000</td>
      <td>Wed Mar 28 04:02:03 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>483</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>483</th>
      <td>-0.3400</td>
      <td>Wed Mar 28 03:55:32 +0000 2018</td>
      <td>0.789</td>
      <td>0.211</td>
      <td>0.000</td>
      <td>484</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>484</th>
      <td>0.0000</td>
      <td>Wed Mar 28 03:47:04 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>485</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>485</th>
      <td>-0.8442</td>
      <td>Wed Mar 28 03:32:03 +0000 2018</td>
      <td>0.620</td>
      <td>0.304</td>
      <td>0.076</td>
      <td>486</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>486</th>
      <td>-0.2263</td>
      <td>Wed Mar 28 03:17:04 +0000 2018</td>
      <td>0.909</td>
      <td>0.091</td>
      <td>0.000</td>
      <td>487</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>487</th>
      <td>-0.7717</td>
      <td>Wed Mar 28 03:02:02 +0000 2018</td>
      <td>0.637</td>
      <td>0.288</td>
      <td>0.075</td>
      <td>488</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>488</th>
      <td>0.4588</td>
      <td>Wed Mar 28 02:51:05 +0000 2018</td>
      <td>0.870</td>
      <td>0.000</td>
      <td>0.130</td>
      <td>489</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>489</th>
      <td>0.4019</td>
      <td>Wed Mar 28 02:38:06 +0000 2018</td>
      <td>0.863</td>
      <td>0.000</td>
      <td>0.137</td>
      <td>490</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>490</th>
      <td>0.0258</td>
      <td>Wed Mar 28 02:26:03 +0000 2018</td>
      <td>0.823</td>
      <td>0.086</td>
      <td>0.091</td>
      <td>491</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>491</th>
      <td>0.4939</td>
      <td>Wed Mar 28 02:17:02 +0000 2018</td>
      <td>0.789</td>
      <td>0.000</td>
      <td>0.211</td>
      <td>492</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>492</th>
      <td>0.0000</td>
      <td>Wed Mar 28 02:02:02 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>493</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>493</th>
      <td>0.4404</td>
      <td>Wed Mar 28 01:51:02 +0000 2018</td>
      <td>0.734</td>
      <td>0.000</td>
      <td>0.266</td>
      <td>494</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>494</th>
      <td>0.0000</td>
      <td>Wed Mar 28 01:41:04 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>495</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>495</th>
      <td>-0.8860</td>
      <td>Wed Mar 28 01:32:09 +0000 2018</td>
      <td>0.691</td>
      <td>0.309</td>
      <td>0.000</td>
      <td>496</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>496</th>
      <td>0.0000</td>
      <td>Wed Mar 28 01:17:05 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>497</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>497</th>
      <td>0.4019</td>
      <td>Wed Mar 28 01:10:50 +0000 2018</td>
      <td>0.876</td>
      <td>0.000</td>
      <td>0.124</td>
      <td>498</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>498</th>
      <td>0.0000</td>
      <td>Wed Mar 28 01:02:06 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>499</td>
      <td>@nytimes</td>
    </tr>
    <tr>
      <th>499</th>
      <td>-0.4215</td>
      <td>Wed Mar 28 00:51:06 +0000 2018</td>
      <td>0.865</td>
      <td>0.135</td>
      <td>0.000</td>
      <td>500</td>
      <td>@nytimes</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 7 columns</p>
</div>




```python
# Create plot
plt.plot(np.arange(len(sentiments_pd["Compound"])),
         sentiments_pd["Compound"], marker="o", linewidth=0.5,
         alpha=0.8)

# # Incorporate the other graph properties
plt.title("Sentiment Analysis of Tweets (%s) for %s" % (time.strftime("%x"), target_user))
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
plt.savefig("ScatterPlot.png")
plt.show()
```


![png](output_3_0.png)



```python
#group by user
group_user_pd= sentiments_pd.groupby(["User"]).mean()["Compound"]

group_user_pd
```




    User
    @BBCWorld    0.058010
    @CBSNews    -0.271185
    @CNN        -0.149130
    @FoxNews    -0.217195
    @nytimes    -0.063350
    Name: Compound, dtype: float64




```python
users = group_user_pd
x_axis = np.arange(len(group_user_pd))
plt.bar(x_axis, users, color='r', alpha=0.5, align="edge")

plt.title("Bar Chart News Org Sentiments")
plt.xlabel("News Org")
plt.ylabel("Vader Compound Score")

tick_locations = [value+0.4 for value in x_axis]
plt.xticks(tick_locations, users)

plt.savefig("BarChart.png")
```


![png](output_5_0.png)



```python
#BBSWorld has the most positive compound score
#CBSNews has the lowest vader compound score
#All but BBSWorld has negative vader compound scores
```

---
output: github_document
bibliography: references.bib
---
# ReadMe

## Spotify User Behaviour Predictor

-   Author: Macy Chan

Spotify user behaviour analysis and prediction model

This project will be used for a self-built Spotify playlist recommendation application - Music and Mood.  
[Click here](https://macy-chan.com/MusicAndMood.html) to learn more about the details of the application.


## Introduction
Here we attempt to analyze Spotify user behavior by explanatory data analysis (EDA), statistic inference and machine learning.

The data set that was used in this project is Kaggle's [Spotify Song Attributes](https://www.kaggle.com/geomack/spotifyclassification/home) dataset. It was sourced from Kaggle and can be found [here](https://www.kaggle.com/mrmorj/dataset-of-songs-in-spotify). Each row in the data set contains a number of features of songs from 2017 and a binary variable `target` that represents whether the user liked the song (encoded as 1) or not (encoded as 0). See the documentation of all the features [here](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/).


## Methodology

This notebook aims at analyzing Spotify users’ music preferences and building a prediction model for playlist recommendations. Such music preferences are based on song features from the [Spotify data set](https://www.kaggle.com/mrmorj/dataset-of-songs-in-spotify), including but not limited to `danceability`, `length of song`, `key`, `loudness`. The same song features can be queried through Spotify API’s song feature [here](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-several-audio-features). 

Within this notebook, firstly, an explanatory data analysis is done to visualize the data set and to explore the potential of identifying user music preference from given data. Secondly, a statistic inference is preformed to see whether the user from the data set shows any significant preference on these features. Lastly, a prediction model is developed through machine learning in order to predict user preference for new songs.


## What can the algorithm do more?

For further application of the trained model, we can refit the prediction model by reading users' favorite playlists or music history in the last 30 days. Once we have a model to predict user preference, we can give the algorithm some of the users' go-to playlists as an prediction input. The application [Music and Mood](https://macy-chan.com/MusicAndMood.html), which was developed by me earlier, can then recommend new songs base on the song preference. Bamp! Within a few clicks, users will have a bunch of recommended songs in a playlist ready for them on the road.


### Upcoming updates
- PCA feature engineering on `genres` and `word embedding sentiment analysis`.
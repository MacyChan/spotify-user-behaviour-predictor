#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# ## Methodology
# 
# This notebook is dedicated for analyzing Spotify user behavior and building a prediction model for Spotify playlist recommendation application, [Music and Mood](https://macy-chan.com/MusicAndMood.html), which is also developed by me. The aim for this project is to analyze if particular user has song selection preference base on different song features such as `danceability`, `length of song`, `key`, `loudness` and so on, where these features are also provided by Spotify API in song feature query [here](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-several-audio-features).
# 
# If there is any actual preference base on song characteristics for a user, we can further build up a machine learning model, by reading users' favorite playlist or last listen in 30 days, the model can recommend new songs base on their song preference.

# ## Why do you start this project?
# 
# I believe everyone has slightly OCD in their lives. Have you ever seen a email inbox of your friends and it drove you crazy simply because there are 20k unread email? Have you ever seen a computer desktop full of unorganized files and you just feel dizzy? Have you every borrow your friends internet browsers for research and you just couldn't help to close the millions tabs that he/she has opened in the past couple of days? weeks? months?
# 
# If you have similar experiences, you are my friend. One of the tasks I need myself to do from time to time is to organize my Spotify playlist. I have a personal playlist that I maintain it frequently. Every now and then, I would go through it, remove the songs that no longer fit my taste, and pull out the Today's Top Hit (yes, this playlist only), go through that as well, and add the new songs that I like into my personal playlist at one sit. My mind told me not to be bother so much but my heart want to listen to new songs and those songs have to be in my personal playlist. Unfortunately not every song fits my type. 
# 
# One day, when I was repeating this time consuming playlist process, this idea popped. Why don't I design an algorithm to help me finish this task, so that I have more time to organize my laptop folder?

# ## Difference with Spotify
# 
# You may have asked, Spotify also has song recommendation, is it not good enough for you? Short answer is Yes, it is not good enough. I enjoy Spotify recommended song and the algorithm is doing a good to recommend songs for me. Most of the recommended songs fits my mood and I give a credit for that. However, I have no control on what songs I would like to pick. Given my raw and desperate story about, I want to be recommend only from some designated playlist, which are the go-to playlist that I like and grab songs that I like from that playlist.

# ## What can the algorithm do more?
# If it works, it works. Other users can provide some playlist that they like, or their past 30 days listen history as an input of the algorithm. Additionally, give it some of your go-to playlists. The algorithm will help you grab the songs that you like from your go-to playlist. Bamp! You have a bunch of new songs ready for you on the road.

---
output: github_document
bibliography: doc/spotify_user_behaviour_refs.bib
---

# Spotify User Behaviour Predictor

-   author: Macy Chan

Spotify user behaviour and prediction model analysis for Spotify Playlist Recommendation System

Music and Mood (here)[https://macy-chan.com/MusicAndMood.html]

## About

Here we attempt to analys spotify user behavior by EDA and ML.

The data set that was used in this project is Kaggle's [Spotify Song Attributes](https://www.kaggle.com/geomack/spotifyclassification/home) dataset. It was sourced from Kaggle and can be found [here](https://www.kaggle.com/mrmorj/dataset-of-songs-in-spotify). Each row in the data set contains a number of features of songs from 2017 and a binary variable `target` that represents whether the user liked the song (encoded as 1) or not (encoded as 0). See the documentation of all the features [here](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/).

## Report

The final report can be found [here](https://macychan.github.io/spotify-user-behaviour-predictor/).

## Usage

To replicate the analysis, clone this GitHub repository, install the [dependencies](#dependencies) listed below, and run the following commands at the command line/terminal from the root directory of this project:

    # download data
    # python src/download_data.py --out_type=feather --url=http://mlr.cs.umass.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data # --out_file=data/raw/wdbc.feather
    # run eda report
    #Rscript -e "rmarkdown::render('src/breast_cancer_eda.Rmd')"

## Dependencies

-   Python 3.7.3 and Python packages:

    -   docopt==0.6.2
    -   requests==2.22.0
    -   pandas==0.24.2
    -   feather-format==0.4.0

-   R version 3.6.1 and R packages:

    -   knitr==1.26
    -   tidyverse==1.2.1
    -   ggridges==0.5.1
    -   ggthemes==4.2.0

# References

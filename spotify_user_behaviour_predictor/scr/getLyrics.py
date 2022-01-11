# author: Macy Chan
# date: 2022-01-10

"""Get Lyrics by artist and song_title in csv. Output a csv file with artist, song_title and lyrics. "../data/credentials.json" has Genius credentials.

Usage: src/getLyrics.py

Options:

"""

from docopt import docopt
import requests
import os, sys
import pandas as pd
import json
import urllib.parse
import re
import lyricsgenius
from alive_progress import alive_bar


opt = docopt(__doc__)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def getLyrics(genius, title, artist):
    """
    Find lyrics of song using Genius API

    Parameters
    ----------
    title :
        song title
    artist :
        artist of song

    Returns
    ----------
        lyrics of song
    """
    with HiddenPrints():
        song = genius.search_song(title, artist)
        if song:
            lyrics = song.lyrics
            return lyrics


def main():
    print("Checking URL connection...")
    try:
        with open("../data/credentials.json") as f:
            login = json.load(f)
        token = login["token"]
        genius = lyricsgenius.Genius(token, retries=5)

        spotify_df = pd.read_csv("../data/spotify_data.csv")
        spotify_df_lyrics = spotify_df.loc[:, "song_title":"artist"]
        spotify_df_lyrics["lyrics"] = ""

        with alive_bar(len(spotify_df_lyrics), bar="bubbles", spinner="notes2") as bar:
            for i in range(len(spotify_df_lyrics)):
                lyrics = getLyrics(
                    genius,
                    spotify_df_lyrics.iloc[i]["song_title"],
                    spotify_df_lyrics.iloc[i]["artist"],
                )

                if lyrics:
                    lyrics = re.sub("\[(.*?)\]", "", lyrics)
                    spotify_df_lyrics["lyrics"][i] = lyrics
                bar()

        spotify_df_lyrics.to_csv("../data/spotify_df_processed.csv", index=False)

    except Exception as req:
        print(req)


if __name__ == "__main__":
    main()

# author: Macy Chan
# date: 2021-11-19

"""Downloads data from the web and unzip the data.

Usage: src/down_data.py

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

        spotify_df = pd.read_csv("../data/spotify_data_with_artist.csv")
        spotify_df["lyrics"] = ""

        with alive_bar(len(spotify_df), bar="bubbles", spinner="notes2") as bar:
            for i in range(len(spotify_df)):
                lyrics = getLyrics(
                    genius,
                    spotify_df.iloc[i]["song_title"],
                    spotify_df.iloc[i]["artist"],
                )

                if lyrics:
                    lyrics = re.sub("\[(.*?)\]", "", lyrics)
                    spotify_df["lyrics"][i] = lyrics
                bar()

        spotify_df.to_csv("../data/spotify_df_processed_test.csv", index=False)

    except Exception as req:
        print(req)


if __name__ == "__main__":
    main()
    # main(opt["--url"], opt["--out_folder"])

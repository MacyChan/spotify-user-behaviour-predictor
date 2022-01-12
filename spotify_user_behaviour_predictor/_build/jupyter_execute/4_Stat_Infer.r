library(tidyverse)
library(broom)
library(GGally)

spotify_df <- read_csv("data/spotify_data.csv")
head(spotify_df)

spotify_df_num <- spotify_df[2:15]
head(spotify_df_num)

ML_reg <- lm( target ~ ., data = spotify_df_num) |> tidy(conf.int = TRUE)

ML_reg<- ML_reg |>
    mutate(Significant = p.value < 0.05)

ML_reg

ML_reg |>
    filter(Significant == TRUE) |>
    select(term) 

ggpairs(data = spotify_df_num)

import pandas as pd
import numpy as np
import os
import sys
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import utils.consts

def __create_possible_genres(df_artists):
    # Cria uma lista de todos os generos
    possible_genres = set()

    for genres in df_artists['genre']:
        for genre in genres.split(','):
            possible_genres.add(genre)
    
    if not os.path.exists(utils.consts.POSSIBLE_GENRES_PATH):
        np.savetxt(utils.consts.POSSIBLE_GENRES_PATH, np.array(list(possible_genres)), fmt='%s')
    
    return possible_genres

def __create_arts_df(df_artists, possible_genres):
    if not os.path.exists(utils.consts.ARTS_PATH):
        #Cria o dataframe com as imagens e as classes
        for i, artist in df_artists.iterrows():
            arts_number = artist['paintings']
            current_genres = artist['genre'].split(',')
            artist_name = artist['name'].replace(' ', '_')
            art_artist_dict = {}

            art_artist_dict['artist id'] = np.repeat(artist['id'], arts_number)
            art_artist_dict['image path'] = np.array([f'{utils.consts.IMAGES_PATH}/{artist_name}/{artist_name}_{art_num}.jpg' for art_num in range(1, arts_number + 1)])
            for genre in possible_genres:
                if genre in current_genres:
                    art_artist_dict[genre] = np.ones((arts_number), dtype=np.int8)
                else:
                    art_artist_dict[genre] = np.zeros((arts_number), dtype=np.int8)
            df_art_artist = pd.DataFrame(art_artist_dict)
            if i == 0:
                df_arts = df_art_artist
            else:
                df_arts = pd.concat([df_arts, df_art_artist], ignore_index=True)
        
        df_arts.to_csv(utils.consts.ARTS_PATH, index=False)

def __fix_encoding_imgs_error(df_artists):
    erro1 = 'Albrecht_DuÔòá├¬rer'
    erro2 = 'Albrecht_Du╠êrer'

    correto = df_artists.iloc[19]['name'].replace(' ', '_')


    if os.path.exists(f'dataset/images/images/{erro1}'):
        shutil.rmtree(f'dataset/images/images/{erro1}')

    if os.path.exists(f'dataset/images/images/{erro2}'):
        os.rename(
            f'dataset/images/images/{erro2}', 
            f'dataset/images/images/{correto}'
        )

        for i, file in enumerate(os.listdir(f'dataset/images/images/{correto}')):
            os.rename(
                f'dataset/images/images/{correto}/{file}', 
                f'dataset/images/images/{correto}/{correto}_{i+1}.jpg'
            )

def preprocess_dataset():
    df_artists = pd.read_csv(utils.consts.ARTISTS_PATH)

    possible_genres = __create_possible_genres(df_artists)
    __create_arts_df(df_artists, possible_genres)
    __fix_encoding_imgs_error(df_artists)


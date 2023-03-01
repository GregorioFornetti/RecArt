import pandas as pd
import numpy as np

possible_genres = ['gen 1', 'gen 2', 'gen 3', 'gen 4', 'gen 5', 'gen 6', 'gen 7']

artists_teste = pd.DataFrame(
    {
        'id': [0, 1],
        'name': ['Artista 1', 'Artista 2'],
        'years': [30, 35],
        'genres': ['Expressionism', 'Expressionism, Impressionism'],
        'wikipedia': ['site 1', 'site 2']
    }
)

imgs_preds_teste = pd.DataFrame(
    {
        'artist id': [0, 1, 0],
        'image path': ['path1', 'path2', 'path3'],
        'gen 1': [0, 1, 0],
        'gen 2': [0, 1, 1],
        'gen 3': [1, 0, 1],
        'gen 4': [1, 0, 1],
        'gen 5': [0, 1, 0],
        'gen 6': [0, 1, 0],
        'gen 7': [0, 1, 1]
    }
)

img_pred_teste = pd.DataFrame(
    [[0, 0, 1, 1, 0, 0, 0]], columns=['gen 1', 'gen 2', 'gen 3', 'gen 4', 'gen 5', 'gen 6', 'gen 7']
)

errors = np.abs(imgs_preds_teste[possible_genres].to_numpy() - img_pred_teste.to_numpy())
similarities = np.mean(1 - errors, axis=1)

teste_join = imgs_preds_teste.join(artists_teste.set_index('id'), on='artist id')
teste_join['similaridade'] = similarities

rec_df = teste_join[['name', 'wikipedia', 'genres', 'image path', 'similaridade']]
rec_df = rec_df.rename(columns={
    'name': 'artist name',
    'wikipedia': 'artist wiki',
    'genres': 'artist genres'
})
rec_df = rec_df.sort_values(by='similaridade', ascending=False)

print(rec_df)
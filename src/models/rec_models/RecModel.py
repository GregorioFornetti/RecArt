import numpy as np
import pandas as pd

from abc import ABC, abstractclassmethod

from utils.functions import read_and_resize_img, load_possible_genres, load_artists_dataset

class RecModel(ABC):

    @abstractclassmethod
    def _init_params(self):
        '''
        Inicializa os parâmetros que serão utilizados no __init__ para inicializar o
        modelo.
        Os atributos que precisam ser inicializados são:
        
        - model
        '''
        pass

    def __init__(self):
        '''
        Inicializa o modelo de recomendações
        '''
        self._init_params()
        self.imgs_preds = self.model.get_imgs_predictions()

    def predict(self, image_path):
        '''
        Irá prever os rotulos/classes da imagem fornecida, e recomendar imagens semelhantes

        ---

        ## Parâmetros

        ---

        image_path: String
            Caminho relativo até a imagem que será classificada

        --- 

        ## Retorno

        ---

        Um dicionário seguindo a seguinte estrutura:
            {
            "predictions": { 
                "selected":
                    [
                        {
                            "nome": nome movimento artístico
                            "probability": probabilidade da imagem pertencer ao movimento ártistico (valor real entre 0 e 1)
                        }
                        ...
                    ]
                "not_selected":
                    [
                        {
                            "nome": nome movimento artístico
                            "probability": probabilidade da imagem pertencer ao movimento ártistico (valor real entre 0 e 1)
                        }
                        ...
                    ]
                }
            "similar images": DataFrame contendo informações de similaridade da imagem atual para cada imagem no dataset. 
                              
                             Esse DataFrame vai estar ordenado para que a primeira imagem seja a mais semelhante e assim por diante.

                             O DataFrame possui as colunas: 

                             1 - "artist name": nome do artista
                             2 - "artist wiki": link para wikipidia sobre o artista
                             3 - "artist genres": movimentos ártisticos, separados por virgula, que o artista pertence
                             4 - "image path": caminho até a imagem em questão
                             5 - "similarity": valor de similaridade entre 0 e 1. Se for próximo a 1, a imagem atual é muito
                                semelhante a imagem enviada, caso seja próximo a 0, a imagem não é semelhante
            
            }
        '''
        img_preds = self.model.predict_proba(image_path)

        all_imgs_preds = self.imgs_preds

        possible_genres = load_possible_genres()

        artists_df = load_artists_dataset()

        preds_dict = {"selected": [], "not selected": []}
        for i, pred in enumerate(img_preds):
            pred_dict = {
                "name": possible_genres[i],
                "probability": pred
            }

            if pred >= 0.5:
                preds_dict["selected"].append(pred_dict)
            else:
                preds_dict["not selected"].append(pred_dict)

        preds_dict["selected"] = sorted(preds_dict["selected"], key=lambda pred: pred['probability'], reverse=True)
        preds_dict["not selected"] = sorted(preds_dict["not selected"], key=lambda pred: pred['probability'], reverse=True)

        img_preds = pd.DataFrame([img_preds], columns=possible_genres)

        errors = np.abs(all_imgs_preds[possible_genres].to_numpy() - img_preds.to_numpy())
        similarities = np.mean(1 - errors, axis=1)

        rec_df = all_imgs_preds.join(artists_df.set_index('id'), on='artist id')
        rec_df['similarity'] = similarities
        rec_df = rec_df[['name', 'wikipedia', 'genre', 'image path', 'similarity']]
        rec_df = rec_df.rename(columns={
            'name': 'artist name',
            'wikipedia': 'artist wiki',
            'genre': 'artist genres'
        })
        rec_df = rec_df.sort_values(by='similarity', ascending=False)

        return {
            "predictions": preds_dict,
            "similar images": rec_df
        }

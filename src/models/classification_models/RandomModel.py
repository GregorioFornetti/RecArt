
import numpy as np

from classification_models.Model import Model
from utils.functions import load_arts_dataset, load_possible_genres

class RandomModel(Model):

    def _init_attributes(self):
        '''
        Inicializa os atributos que serão utilizados nos métodos genéricos do modelo

        Atualmente é necessário definer as variáveis:
        - model_name
        - img_shape
        '''
        self.model_name = 'Random'
        self.img_shape = (256, 256, 3)
        df_arts = load_arts_dataset()
        possible_genres = load_possible_genres()
        self.ones_proportion = df_arts[possible_genres].sum().sum() / (len(df_arts) * len(possible_genres))
        print(f'A proporção de 1\'s é de: {self.ones_proportion}')
    
    def _create_neural_network(self):
        '''
        Inicializa a estrutura da rede neural
        '''
        pass

    def predict_proba(self, img_path):
        return np.random.choice(
            np.array([0, 1]),
            size=self.OUTPUT_SIZE,
            p=[1 - self.ones_proportion, self.ones_proportion]
        )

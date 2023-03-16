
from utils.functions import read_and_resize_img, load_possible_genres, load_arts_dataset, load_train_dataset
import utils.consts
from classification_models.Model import Model

import pandas as pd
import numpy as np

class KNN(Model):

    def _init_attributes(self):
        '''
        Inicializa os atributos que serão utilizados nos métodos genéricos do modelo

        Atualmente é necessário definer as variáveis:
        - model_name
        - img_shape
        '''
        self.model_name = 'KNN'
        self.img_shape = (32, 32, 3)
        self.K = 5
    
    def _create_neural_network(self):
        '''
        Inicializa a estrutura da rede neural

        Deve ser definida em self.model
        '''
        pass

    def __init__(self, load=True):
        '''
        Responsável por inicializar a rede neural.

        ---

        ## Parâmetros

        ---

        load: Boolean (default=True)
            Caso seja verdadeiro, irá carregar o modelo de rede neural salvo, caso contrário, irá
            instanciar a rede neural com pesos padronizados (sendo necessário treina-la).
        ---
        '''
        self.OUTPUT_SIZE = len(load_possible_genres())

        self._init_attributes()
        self._create_folders()

        if load:
            self.fit_full()
    
    def fit_full(self, batch_size=32, epochs=10):
        '''
        Treina o modelo para produção. Após executar, o modelo (seus parâmetros aprendidos) serão salvos,
        podendo ser carregados na inicialização do modelo quando load=True

        ---

        ## Parâmetros

        ---

        batch_size: int (default=32)
            Tamanho do lote de treinamento
        
        ---
        
        epochs: int (default=10)
            Número de epochs para o treinamento
        
        ---
        '''
        train_df = load_arts_dataset().reset_index(drop=True)
        self.X = np.zeros((len(train_df), self.img_shape[0], self.img_shape[1], self.img_shape[2]))
        for i, art_info in train_df.iterrows():
            self.X[i] = read_and_resize_img(art_info['image path'], (self.img_shape[0], self.img_shape[1]))
        self.y = train_df[load_possible_genres()].values



    def fit_and_test(self, batch_size=32, epochs=10):
        '''
        Treina o modelo para testes. Após executar, será salvo os resultados dos testes

        ---

        ## Parâmetros

        ---

        batch_size: int (default=32)
            Tamanho do lote de treinamento
        
        ---
        
        epochs: int (default=10)
            Número de epochs para o treinamento
        
        ---

        ## Retorno

        ---

        Retorna uma tupla contendo dois DataFrames, o primeiro DataFrame com métricas para cada classe, e o segundo
        um DataFrame de métricas gerais. Esses dois DataFrames serão salvos no "save_path" do modelo.
        '''
        train_df = load_train_dataset().reset_index(drop=True)
        self.X = np.zeros((len(train_df), self.img_shape[0], self.img_shape[1], self.img_shape[2]))
        for i, art_info in train_df.iterrows():
            self.X[i] = read_and_resize_img(art_info['image path'], (self.img_shape[0], self.img_shape[1]))
        self.y = train_df[load_possible_genres()].to_numpy()
        return self.evaluate()
    
    def save_model(self):
        '''
        Salva os parâmetros aprendidos do modelo
        '''
        pass

    def predict_proba(self, img_path):
        '''
        Lê e ajusta a imagem e faz predições.

        ---

        ## Parâmetros

        ---

        img_path: String
            Caminho relativo para a imagem que irá ser classificada
        
        ---

        ## Retorno

        ---

        Retorna uma lista de probabilidades, variando de 0 a 1. Cada valor na lista é a probabilidade de pertencer a uma classe.
        Ex: lista[0] = 0.61 (61% de chance da imagem pertencer a classe na posição 0)
        '''
        image_to_predict = read_and_resize_img(img_path, (self.img_shape[0], self.img_shape[1]))

        distances = np.zeros((len(self.X), 2))
        for i in range(self.X.shape[0]):
            distances[i] = [np.sqrt(np.sum((image_to_predict - self.X[i]) ** 2)), i]
        
        distances_df = pd.DataFrame(distances, columns=['distance', 'index']).sort_values(by='distance')
        preds = self.y[distances_df['index'].iloc[:self.K].to_numpy().astype(int)].mean(axis=0)

        return preds

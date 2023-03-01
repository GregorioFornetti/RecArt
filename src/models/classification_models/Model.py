import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

import os
from abc import ABC, abstractclassmethod
from datetime import datetime

from utils.functions import read_and_resize_img, load_possible_genres, load_arts_dataset, load_train_for_test_generator, load_test_dataset, load_train_full_generator
import utils.consts



class Model(ABC):

    @abstractclassmethod
    def _init_attributes(self):
        '''
        Inicializa os atributos que serão utilizados nos métodos genéricos do modelo

        Atualmente é necessário definer as variáveis:
        - model_name
        - img_shape
        '''
        pass
    
    @abstractclassmethod
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
        self.__create_folders()
        
        if not load:
            self._create_neural_network()
        else:
            self.model = tf.keras.models.load_model(f'{self.save_path}/params')
    
    def __create_folders(self):
        '''
        Cria os diretórios para salvar os resultados do modelo
        '''
        self.save_path = f'{utils.consts.MODELS_SAVES_PATH}/{self.model_name}'
        self.params_path = f'{self.save_path}/params'
        self.results_path = f'{self.save_path}/results'

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        if not os.path.exists(self.params_path):
            os.makedirs(self.params_path)
        
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
    
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
        train_gen = load_train_full_generator((self.img_shape[0], self.img_shape[1]))
        self.history = self.model.fit(train_gen, batch_size=batch_size, epochs=epochs)
        self.save_model()


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
        train_gen = load_train_for_test_generator((self.img_shape[0], self.img_shape[1]))
        self.history = self.model.fit(train_gen, batch_size=batch_size, epochs=epochs)
        return self.evaluate()
    
    def save_model(self):
        '''
        Salva os parâmetros aprendidos do modelo
        '''
        self.model.save(self.params_path)

    def evaluate(self):
        '''
        Calcula as métricas de avaliação para a rede neural. Salva os resultados no caminho
        de "save" do modelo.

        ---

        ## Retorno

        ---

        Retorna uma tupla contendo dois DataFrames, o primeiro DataFrame com métricas para cada classe, e o segundo
        um DataFrame de métricas gerais. Esses dois DataFrames serão salvos no "save_path" do modelo.
        '''
        possible_genres = load_possible_genres()
        preds_df = pd.DataFrame(columns=possible_genres, dtype=int)
        test_df = load_test_dataset()
        
        for _, test_row in test_df.iterrows():
            pred = self.predict(test_row['image path'])
            pred_df = pd.DataFrame([pred], columns=possible_genres)
            preds_df = pd.concat([preds_df, pred_df], ignore_index=True)
        
        y_true = test_df[possible_genres].reset_index(drop=True)
        df_metrics = pd.DataFrame(columns=['genre', 'accuracy', 'precision', 'recall', 'f1'])

        tp = 0
        fp = 0
        fn = 0

        for genre in possible_genres:
            metrics = {}

            tp += ((preds_df[genre] == 1) & (y_true[genre] == 1)).sum()
            fp += ((preds_df[genre] == 1) & (y_true[genre] == 0)).sum()
            fn += ((preds_df[genre] == 0) & (y_true[genre] == 1)).sum()

            metrics['genre'] = [genre]
            metrics['accuracy'] = [accuracy_score(preds_df[genre], y_true[genre])]
            metrics['precision'] = [precision_score(preds_df[genre], y_true[genre])]
            metrics['recall'] = [recall_score(preds_df[genre], y_true[genre])]
            metrics['f1'] = [f1_score(preds_df[genre], y_true[genre])]

            df_metrics = pd.concat([df_metrics, pd.DataFrame(metrics)], ignore_index=True)
        
        results_dir = datetime.now().strftime("%d_%m_%Y_%H_%M")
        os.mkdir(f'{self.results_path}/{results_dir}')

        df_metrics.to_csv(f'{self.results_path}/{results_dir}/all_classes_results.csv', index=False)
        
        macro_precision = tp / (tp + fp)
        macro_recall = tp / (tp + fn)

        agg_results = pd.DataFrame({
            'mean accuracy': [df_metrics['accuracy'].mean()],
            'micro precision': [df_metrics['precision'].mean()],
            'micro recall': [df_metrics['recall'].mean()],
            'micro f1': [df_metrics['f1'].mean()],
            'macro precision': [macro_precision],
            'macro recall': [macro_recall],
            'macro f1': [2 * ((macro_precision * macro_recall) / (macro_precision + macro_recall))]
        })
        agg_results.to_csv(f'{self.results_path}/{results_dir}/agg_results.csv', index=False)

        return (df_metrics, agg_results)
            
        

    def predict(self, img_path):
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

        Retorna uma lista de valores que podem ser 0 ou 1. Se o valor for 0, a imagem não faz parte da classe, se for 1, sim.
        Ex: lista[0] = 1 (a imagem pertence ao gênero na posição 0) e lista[0] = 0 (a imagem não pertence ao gênero)
        '''
        return (self.predict_proba(img_path) > 0.5).astype(int)

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
        image = read_and_resize_img(img_path, (self.img_shape[0], self.img_shape[1]))
        preds = self.model.predict(image.numpy().reshape(1, 256, 256, 3), verbose=0)
        return preds[0]

    def save_imgs_predictions(self):
        '''
        Salva as predições de probabilidades para cada imagem

        ---

        Não há retorno, apenas irá salvar no "save_path" os resultados obtidos para cada imagem.
        '''
        df_arts = load_arts_dataset()
        df_values = []

        for i, art in df_arts.iterrows():
            df_values.append([art['artist id'], art['image path'], *self.predict_proba(art['image path'])])
            print(f'{i+1} de {len(df_arts)}', end='\r')
        
        imgs_preds = pd.DataFrame(df_values, columns=['artist id', 'image path', *load_possible_genres()])
        
        imgs_preds.to_csv(f'{self.save_path}/imgs_predicts.csv', index=False)
    
    def get_imgs_predictions(self):
        '''
        ## Retorno

        ---

        DataFrame contendo as predições feitas pelo modelo em todas as imagens do dataset. Cada linha possui
        o id do artista da imagem ("artist id"), o caminho até a imagem ("image path") e várias colunas com
        os nomes de cada gênero ártistico, com o valor da predição, variando de 0 a 1.
        OBS: para conseguir executar esse método é preciso ter salvo esse DataFrame utilizando o método
        "save_imgs_predictions"
        '''
        return pd.read_csv(f'{self.save_path}/imgs_predicts.csv')
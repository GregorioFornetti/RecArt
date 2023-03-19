# RecArt

Projeto da disciplina TÓPICOS AVANÇADOS EM INTELIGÊNCIA ARTIFICIAL

Este projeto tem como objetivo criar uma aplicação que receba uma pintura do usuário, 
de preferência uma obra artística, e a partir dessa imagem indica outras obras semelhantes à fornecida,
tendo como critério o movimento artístico. A relevância dessa aplicação se dá pela sugestão de obras artísticas 
que o usuário desconhecia e que podem ser de grande relevância para ele a partir do movimento artístico e não o conteúdo mais objetivo da obra.

O projeto foi nomeado como RecArt, pois o sistema proposto busca ''recomendar'' 
obras de arte, visto que o resultado final da aplicação são várias pinturas que o usuário possa ter interesse.
No entanto, vale ressaltar que o projeto proposto não é um sistema de recomendação, já que ele não coleta ou 
utiliza informações do usuário para fazer as recomendações

Para a elaboração do projeto, primeiramente a imagem fornecida será classificada dentro de um 
conjunto de possíveis movimentos artísticos que ela pode fazer parte, os quais estão limitados aos disponíveis na base de 
dados ultilizada. A imagem de entrada pode pertencer a múltiplos movimentos artísticos ao mesmo tempo,
logo nosso problema de classificação é do tipo multi-label. Esse classificador, 
como resultado final, irá gerar probabilidades da imagem pertencer a cada movimento artístico.

Em seguida, com as pontuações de cada classe geradas pelo classificador,
serão selecionadas as imagens, dentro do conjunto de dados de treinamento, 
com as pontuações mais semelhantes à imagem fornecida pelo usuário. Para finalizar, 
todo esse sistema será integrado em uma aplicação web, para que a interação com o projeto seja mais amigável ao usuário.

## Como instalar e executar

Para conseguir executar o projeto, é necessário fazer os seguintes passos:
1. baixar o repositório 

2. instalar todas as dependências disponiveis em "requirements.txt" (utilizando pip install requirements.txt)

3. baixar a base de dados utilizada, e armazenar dentro de uma pasta chamada "dataset" dentro do repositório
raiz do projeto. Para baixar, [clique aqui](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)

4. em "src\models\utils\consts.py", modificar a variavel "MAIN_DIR_ABSOLUTE_PATH" para o caminho absoluto do repositório
raiz em sua máquina

5. executar por completo os notebooks (disponiveis no repositório com mesmo nome): ResNet.ipynb, AlexNet.ipynb, KNN.ipynb, SimpleNeuralNet.ipynb
( isso pode levar muito tempo, porém é necessário, já que não foi possível salvar os parâmetros aprendidos no github, já que eles são muito grandes)

6. dentro da pasta "src/web" executar o comando "flask --app main.py run"

## Prints

### Homepage

![homepage](https://raw.githubusercontent.com/GregorioFornetti/RecArt/main/prints/home.jpg)

### Pagina de recomendações

![recomendações](https://raw.githubusercontent.com/GregorioFornetti/RecArt/main/prints/recs.jpg)

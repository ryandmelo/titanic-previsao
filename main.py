import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Carregamento dos dados do treino e do teste
dados_teste = pd.read_csv('./test.csv', sep=',')
dados_treino = pd.read_csv('./train.csv', sep=',')

# Eliminei as colunas, movi pra uma variavel só e usei para os dois datasets
colunas_excluir = ['Passageiro ID', 'Nome', 'Número do Ticket', 'Taxa do Ticket', 'Número da Cabine', 'Local de Embarque']
dados_teste = dados_teste.drop(columns=colunas_excluir).dropna()
dados_treino = dados_treino.drop(columns=colunas_excluir).dropna()

# Transformar a coluna 'Sexo' em valores numéricos 
encoder = LabelEncoder()
dados_teste['Sexo'] = encoder.fit_transform(dados_teste['Sexo'])
dados_treino['Sexo'] = encoder.fit_transform(dados_treino['Sexo'])

# Usei o "iloc" por sujestao de um colega
atributos_teste = dados_teste.iloc[:, 1:].values
resultado_teste = dados_teste.iloc[:, 0].values

atributos_treino = dados_treino.iloc[:, 1:].values
resultado_treino = dados_treino.iloc[:, 0].values

# Criar e treinar
modelo_bayes = GaussianNB()
modelo_bayes.fit(atributos_treino, resultado_treino)

predicoes = modelo_bayes.predict(atributos_teste)

precisao_modelo = accuracy_score(resultado_teste, predicoes)

# resultado
print(f'A precisão do modelo foi de {precisao_modelo * 100:.2f}%')
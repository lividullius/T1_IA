import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dados = pd.read_csv("Dados-T1-IA.csv", sep=";")

print(dados.shape)
print(dados.head())

X = dados.drop(columns=("Resultado"))
Y = dados["Resultado"].values

xTreino, xTemporario, yTreino, yTemporario = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
xTeste, xValidacao, yTeste, yValidacao = train_test_split(xTemporario, yTemporario, test_size=0.5, random_state=42, stratify=yTemporario)

# Teste da quantidade de árvores
#melhorAcuracia = 0
#melhorArvores = 0

#for arvores in [10, 25, 50, 100, 150, 200]:
    #modelo = RandomForestClassifier(
        #n_estimators=arvores,
        #random_state=42
    #)
    #modelo.fit(xTreino, yTreino)
    #previsoes = modelo.predict(xValidacao)
    #acuracia = accuracy_score(yValidacao, previsoes)
    #print(
        #"Árvores:", arvores,
        #"Acurácia:", acuracia
    #)
    #if acuracia > melhorAcuracia:
        #melhorAcuracia = acuracia
        #melhorArvores = arvores

#print("\nMelhor quantidade de árvores:", melhorArvores)
#print("Melhor acurácia:", melhorAcuracia)


# Teste da profundidade
#melhorAcuracia = 0
#melhorProfundidade = 0

#for profundidade in [3, 5, 7, 10, None]:
    #modelo = RandomForestClassifier(
        #n_estimators=100,
        #max_depth=profundidade,
        #random_state=42
    #)
    #modelo.fit(xTreino, yTreino)
    #previsoes = modelo.predict(xValidacao)
    #acuracia = accuracy_score(yValidacao, previsoes)
    #print(
        #"Profundidade:", profundidade,
        #"Acurácia:", acuracia
    #)
    #if acuracia > melhorAcuracia:
        #melhorAcuracia = acuracia
        #melhorProfundidade = profundidade

#print("\nMelhor profundidade:", melhorProfundidade)
#print("Melhor acurácia:", melhorAcuracia)


# Teste do número mínimo de amostras por folha
#melhorAcuracia = 0
#melhorFolha = 0

#for folha in [1, 2, 4]:
    #modelo = RandomForestClassifier(
        #n_estimators=100,
        #max_depth=10,
        #min_samples_leaf=folha,
        #random_state=42
    #)
    #modelo.fit(xTreino, yTreino)
    #previsoes = modelo.predict(xValidacao)
    #acuracia = accuracy_score(yValidacao, previsoes)
    #print(
        #"Min samples leaf:", folha,
        #"Acurácia:", acuracia
    #)
    #if acuracia > melhorAcuracia:
        #melhorAcuracia = acuracia
        #melhorFolha = folha

#print("\nMelhor min_samples_leaf:", melhorFolha)
#print("Melhor acurácia:", melhorAcuracia)


# Modelo final com os melhores parâmetros
modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=1,
    random_state=42
)

modelo.fit(xTreino, yTreino)

# Resultados da validação
previsoes = modelo.predict(xValidacao)

acuracia = accuracy_score(yValidacao, previsoes)
precisao = precision_score(yValidacao, previsoes, average="weighted")
recall = recall_score(yValidacao, previsoes, average="weighted")
f1 = f1_score(yValidacao, previsoes, average="weighted")

print("\nResultados da validação:")
print("Acurácia:", acuracia)
print("Precisão:", precisao)
print("Recall:", recall)
print("F-measure:", f1)


# Resultados do teste final
previsoesTeste = modelo.predict(xTeste)

acuraciaTeste = accuracy_score(yTeste, previsoesTeste)
precisaoTeste = precision_score(yTeste, previsoesTeste, average="weighted")
recallTeste = recall_score(yTeste, previsoesTeste, average="weighted")
f1Teste = f1_score(yTeste, previsoesTeste, average="weighted")

print("\nResultados do teste final:")
print("Acurácia:", acuraciaTeste)
print("Precisão:", precisaoTeste)
print("Recall:", recallTeste)
print("F-measure:", f1Teste)
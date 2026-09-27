import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dados = pd.read_csv("Dados-T1-IA.csv", sep=";")

print(dados.shape)
print(dados.columns)
print(dados.head())
X = dados.drop(columns=("Resultado"))
print(X.head())
Y = dados["Resultado"].values
print(Y)
#80% para treino e 20% para temporario
xTreino, xTemporario, yTreino, yTemporario = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
#Divide os 20% temporarios em teste e validacao
#Stratify serve para manter a proporção de empate, vitória de x ou o
xTeste, xValidacao, yTeste, yValidacao = train_test_split(xTemporario, yTemporario, test_size=0.5, random_state=42,stratify=yTemporario)
print("Treino: ", len(xTreino))
print("Teste: ", len(xTeste))
print("Validação: ", len(xValidacao))
melhorK = 0
melhorAcuracia = 0

#Testar o melhor k
#for k in range(1, 21):
#    modelo = neighbors.KNeighborsClassifier(n_neighbors=k)
#    modelo.fit(xTreino, yTreino)
#    acuracia = modelo.score(xValidacao, yValidacao)
#    print("K:", k, "Acurácia:", acuracia)
#    if acuracia > melhorAcuracia:
#        melhorAcuracia = acuracia
#        melhorK = k

#print("Melhor K:", melhorK)
#print("Melhor acurácia:", melhorAcuracia)

melhorK = 5
modelo = neighbors.KNeighborsClassifier(n_neighbors=melhorK)
modelo.fit(xTreino, yTreino)
previsoes = modelo.predict(xTeste)
acuracia = accuracy_score(yTeste, previsoes)
average="weighted"
precisao = precision_score(yTeste, previsoes, average="weighted")
recall = recall_score(yTeste, previsoes, average="weighted")
f1 = f1_score(yTeste, previsoes, average="weighted")
print("Acurácia:", acuracia)
print("Precisão:", precisao)
print("Recall:", recall)
print("F-measure:", f1)
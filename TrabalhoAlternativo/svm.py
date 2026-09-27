import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

dados = pd.read_csv("Dados-T1-IA.csv", sep=";")

print(dados.shape)
print(dados.head())

X = dados.drop(columns=("Resultado"))
Y = dados["Resultado"].values

xTreino, xTemporario, yTreino, yTemporario = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
xTeste, xValidacao, yTeste, yValidacao = train_test_split(xTemporario, yTemporario, test_size=0.5, random_state=42, stratify=yTemporario)


# Padroniza os dados para o SVM
normalizador_X = StandardScaler()

xTreino = normalizador_X.fit_transform(xTreino)
xValidacao = normalizador_X.transform(xValidacao)
xTeste = normalizador_X.transform(xTeste)


# Teste dos valores de C
# C=10 e C=100 tiveram a mesma acuracia.
# Foi escolhido C=10 por obter o mesmo resultado com menor penalizacao.

#for valorC in [0.1, 1, 10, 100]:
#    modelo = SVC(kernel="rbf", C=valorC, random_state=42)
#    modelo.fit(xTreino, yTreino)
#    previsoes = modelo.predict(xValidacao)
#    acuracia = accuracy_score(yValidacao, previsoes)
#    print("C:", valorC, "Acurácia:", acuracia)

# Teste dos kernels
# O kernel RBF apresentou 93,18% e o linear 68,18%.
# Por isso foi escolhido o kernel RBF.

#for kernel in ["linear", "rbf"]:
#    modelo = SVC(kernel=kernel, C=10, random_state=42)
#    modelo.fit(xTreino, yTreino)
#    previsoes = modelo.predict(xValidacao)
#    acuracia = accuracy_score(yValidacao, previsoes)
#    print("Kernel:", kernel, "Acurácia:", acuracia)


# Modelo final com os melhores parametros
modelo = SVC(kernel="rbf", C=10, random_state=42)
modelo.fit(xTreino, yTreino)


# Resultados da validacao
previsoes = modelo.predict(xValidacao)

acuracia = accuracy_score(yValidacao, previsoes)
precisao = precision_score(yValidacao, previsoes, average="weighted")
recall = recall_score(yValidacao, previsoes, average="weighted")
f1 = f1_score(yValidacao, previsoes, average="weighted")

print("\nResultados da validacao:")
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
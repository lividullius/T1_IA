import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dados = pd.read_csv("Dados-T1-IA.csv", sep=";")
print(dados.shape)
print(dados.head())

X = dados.drop(columns=("Resultado"))
Y = dados["Resultado"].values

xTreino, xTemporario, yTreino, yTemporario = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
xTeste, xValidacao, yTeste, yValidacao = train_test_split(xTemporario, yTemporario, test_size=0.5, random_state=42, stratify=yTemporario)
print("\nTreino:", len(xTreino))
print("Validação:", len(xValidacao))
print("Teste:", len(xTeste))

melhorAcuracia = 0
melhorProfundidade = 0


for profundidade in range(1, 11):

    modelo = DecisionTreeClassifier(
        criterion="gini",
        max_depth=profundidade,
        random_state=42
    )

    modelo.fit(xTreino, yTreino)

    previsoes = modelo.predict(xValidacao)

    acuracia = accuracy_score(yValidacao, previsoes)

    print("Profundidade:", profundidade, "Acurácia:", acuracia)

    if acuracia > melhorAcuracia:
        melhorAcuracia = acuracia
        melhorProfundidade = profundidade


print("\nMelhor profundidade:", melhorProfundidade)
print("Melhor acurácia:", melhorAcuracia)


#Cria a árvore usando a melhor profundidade
modelo = DecisionTreeClassifier(
    criterion="gini",
    max_depth=melhorProfundidade,
    random_state=42
)

modelo.fit(xTreino, yTreino)
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

previsoesTeste = modelo.predict(xTeste)

acuraciaTeste = accuracy_score(yTeste, previsoesTeste)
precisaoTeste = precision_score(yTeste, previsoesTeste, average="weighted")
recallTeste = recall_score(yTeste, previsoesTeste, average="weighted")
f1Teste = f1_score(yTeste, previsoesTeste, average="weighted")

print("\nResultados do teste:")
print("Acurácia:", acuraciaTeste)
print("Precisão:", precisaoTeste)
print("Recall:", recallTeste)
print("F-measure:", f1Teste)
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "dataset" / "processed"

# Abordagem anterior, mantida como referência:
# dados = pd.read_csv("Dados-T1-IA.csv", sep=";")
# X = dados.drop(columns="Resultado")
# Y = dados["Resultado"].values
# xTreino, xTemporario, yTreino, yTemporario = train_test_split(
#     X, Y, test_size=0.2, random_state=42, stratify=Y
# )
# xTeste, xValidacao, yTeste, yValidacao = train_test_split(
#     xTemporario, yTemporario, test_size=0.5, random_state=42,
#     stratify=yTemporario
# )
# A abordagem atual usa as particoes geradas pelo preprocessing e executa
# o Random Forest para as duas representacoes, ab1 e ab2.


def carregar_dados(abordagem):
    treino = pd.read_csv(DATA_DIR / f"{abordagem}_treino.csv")
    validacao = pd.read_csv(DATA_DIR / f"{abordagem}_validacao.csv")
    teste = pd.read_csv(DATA_DIR / f"{abordagem}_teste.csv")

    return (
        treino.drop(columns="classe"), treino["classe"],
        validacao.drop(columns="classe"), validacao["classe"],
        teste.drop(columns="classe"), teste["classe"],
    )

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


for abordagem in ("ab1", "ab2"):
    print(f"\nIniciando Treino RandomForest - Abordagem: {abordagem}")
    xTreino, yTreino, xValidacao, yValidacao, xTeste, yTeste = carregar_dados(abordagem)

    modelo = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=1, random_state=42,
    )
    modelo.fit(xTreino, yTreino)

    print(f"\nAbordagem: {abordagem}")
    for nome, features, classes in (
        ("Validação", xValidacao, yValidacao),
        ("Teste", xTeste, yTeste),
    ):
        previsoes = modelo.predict(features)
        print(f"{nome}:")
        print("Acurácia:", accuracy_score(classes, previsoes))
        print("Precisão:", precision_score(classes, previsoes, average="weighted"))
        print("Recall:", recall_score(classes, previsoes, average="weighted"))
        print("F-measure:", f1_score(classes, previsoes, average="weighted"))
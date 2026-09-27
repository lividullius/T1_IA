import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
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
# O fluxo novo nao divide os dados novamente: ele usa as particoes prontas
# e executa o mesmo modelo separadamente para ab1 e ab2.


def carregar_dados(abordagem):
    treino = pd.read_csv(DATA_DIR / f"{abordagem}_treino.csv")
    validacao = pd.read_csv(DATA_DIR / f"{abordagem}_validacao.csv")
    teste = pd.read_csv(DATA_DIR / f"{abordagem}_teste.csv")

    return (
        treino.drop(columns="classe"), treino["classe"],
        validacao.drop(columns="classe"), validacao["classe"],
        teste.drop(columns="classe"), teste["classe"],
    )


for abordagem in ("ab1", "ab2"):
    xTreino, yTreino, xValidacao, yValidacao, xTeste, yTeste = carregar_dados(abordagem)
    melhorAcuracia = 0
    melhorProfundidade = 0

    for profundidade in range(1, 11):
        modelo = DecisionTreeClassifier(
            criterion="gini", max_depth=profundidade, random_state=42,
        )
        modelo.fit(xTreino, yTreino)
        previsoes = modelo.predict(xValidacao)
        acuracia = accuracy_score(yValidacao, previsoes)

        if acuracia > melhorAcuracia:
            melhorAcuracia = acuracia
            melhorProfundidade = profundidade

    modelo = DecisionTreeClassifier(
        criterion="gini", max_depth=melhorProfundidade, random_state=42,
    )
    modelo.fit(xTreino, yTreino)

    print(f"\nAbordagem: {abordagem}")
    for nome, features, classes in (
        ("Validação", xValidacao, yValidacao),
        ("Teste", xTeste, yTeste),
    ):
        previsoes = modelo.predict(features)
        print(f"{nome} (profundidade {melhorProfundidade}):")
        print("Acurácia:", accuracy_score(classes, previsoes))
        print("Precisão:", precision_score(classes, previsoes, average="weighted"))
        print("Recall:", recall_score(classes, previsoes, average="weighted"))
        print("F-measure:", f1_score(classes, previsoes, average="weighted"))
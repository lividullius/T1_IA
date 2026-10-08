import pandas as pd
from joblib import dump, load
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "modelos" / "arvore"

def carregar_dados(abordagem):
    treino = pd.read_csv(DATA_DIR / f"{abordagem}_treino.csv")
    validacao = pd.read_csv(DATA_DIR / f"{abordagem}_validacao.csv")
    teste = pd.read_csv(DATA_DIR / f"{abordagem}_teste.csv")

    return (
        treino.drop(columns="classe"), treino["classe"],
        validacao.drop(columns="classe"), validacao["classe"],
        teste.drop(columns="classe"), teste["classe"],
    )


def caminho_modelo(abordagem):
    return MODEL_DIR / f"modelo_{abordagem}.joblib"


def exportar_modelo(modelo, abordagem, features):
    caminho = caminho_modelo(abordagem)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    dump({"modelo": modelo, "features": list(features)}, caminho)
    print(f"Modelo exportado para: {caminho}")


def inferir(features, abordagem="ab2"):
    pacote = load(caminho_modelo(abordagem))
    entrada = pd.DataFrame([features], columns=pacote["features"])
    return pacote["modelo"].predict(entrada)[0]


def main():
    for abordagem in ("ab1", "ab2"):
        xTreino, yTreino, xValidacao, yValidacao, xTeste, yTeste = carregar_dados(abordagem)
        colunas = list(xTreino.columns)

        resultados = []
        for profundidade in range(1, 11):
            modelo = DecisionTreeClassifier(
                criterion="gini", max_depth=profundidade, random_state=42,
            )
            modelo.fit(xTreino, yTreino)
            previsoes = modelo.predict(xValidacao)
            acuracia = accuracy_score(yValidacao, previsoes)
            resultados.append((profundidade, acuracia))

        resultados.sort(key=lambda r: r[1], reverse=True)
        melhorProfundidade, melhorAcuracia = resultados[0]

        print(f"\nAbordagem: {abordagem}")
        print("Busca de hiperparâmetro (max_depth) - top 3:")
        for profundidade, acuracia in resultados[:3]:
            print(f"  max_depth={profundidade}: acurácia={acuracia:.4f}")
        print(f"Escolhido: max_depth={melhorProfundidade} (acurácia={melhorAcuracia:.4f})")

        modelo = DecisionTreeClassifier(
            criterion="gini", max_depth=melhorProfundidade, random_state=42,
        )
        modelo.fit(xTreino, yTreino)

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

        exportar_modelo(modelo, abordagem, colunas)

def avaliar_modelo(modelo, xValidacao, yValidacao, xTeste, yTeste):

	resultados = {}

	for nome, features, classes in (
		("Validação", xValidacao, yValidacao),
		("Teste", xTeste, yTeste),
	):
		previsoes = modelo.predict(features)

		resultados[nome] = {
			"Acurácia": accuracy_score(classes, previsoes),
			"Precisão": precision_score(classes, previsoes, average="weighted"),
			"Recall": recall_score(classes, previsoes, average="weighted"),
			"F-measure": f1_score(classes, previsoes, average="weighted"),
		}

	return resultados

def executar(abordagem):

    xTreino, yTreino, xValidacao, yValidacao, xTeste, yTeste = carregar_dados(abordagem)
    resultados = []

    for profundidade in range(1, 11):
        modelo = DecisionTreeClassifier(
            criterion="gini",
            max_depth=profundidade,
            random_state=42,
        )

        modelo.fit(xTreino, yTreino)
        previsoes = modelo.predict(xValidacao)
        acuracia = accuracy_score(yValidacao, previsoes)
        resultados.append((profundidade, acuracia))

    resultados.sort(key=lambda r: r[1], reverse=True)
    melhorProfundidade, melhorAcuracia = resultados[0]

    modelo = DecisionTreeClassifier(
        criterion="gini",
        max_depth=melhorProfundidade,
        random_state=42,
    )
    modelo.fit(xTreino, yTreino)

    return avaliar_modelo(modelo, xValidacao, yValidacao, xTeste, yTeste,)


if __name__ == "__main__":
    main()
import pandas as pd
from joblib import dump, load
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "modelos" / "livre1"

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


def exportar_modelo(modelo, normalizador, abordagem, features):
	caminho = caminho_modelo(abordagem)
	caminho.parent.mkdir(parents=True, exist_ok=True)
	dump(
		{
			"modelo": modelo,
			"normalizador": normalizador,
			"features": list(features),
		},
		caminho,
	)
	print(f"Modelo exportado para: {caminho}")


def inferir(features, abordagem="ab2"):
	pacote = load(caminho_modelo(abordagem))
	entrada = pd.DataFrame([features], columns=pacote["features"])
	entrada = pacote["normalizador"].transform(entrada)
	return pacote["modelo"].predict(entrada)[0]


def main():
	for abordagem in ("ab1", "ab2"):
		xTreino, yTreino, xValidacao, yValidacao, xTeste, yTeste = carregar_dados(abordagem)
		colunas = list(xTreino.columns)
		normalizador_X = StandardScaler()
		xTreino = normalizador_X.fit_transform(xTreino)
		xValidacao = normalizador_X.transform(xValidacao)
		xTeste = normalizador_X.transform(xTeste)

		resultados = []
		for kernel in ["linear", "rbf"]:
			for valorC in [0.1, 1, 10, 100]:
				modelo = SVC(kernel=kernel, C=valorC, random_state=42)
				modelo.fit(xTreino, yTreino)
				previsoes = modelo.predict(xValidacao)
				acuracia = accuracy_score(yValidacao, previsoes)
				resultados.append((kernel, valorC, acuracia))

		# Em empate de acurácia, prefere-se o menor C (menor penalização).
		resultados.sort(key=lambda r: (-r[2], r[1]))
		melhorKernel, melhorC, melhorAcuracia = resultados[0]

		print(f"\nAbordagem: {abordagem}")
		print("Busca de hiperparâmetro (kernel, C) - top 3:")
		for kernel, valorC, acuracia in resultados[:3]:
			print(f"  kernel={kernel}, C={valorC}: acurácia={acuracia:.4f}")
		print(f"Escolhido: kernel={melhorKernel}, C={melhorC} (acurácia={melhorAcuracia:.4f})")

		modelo = SVC(kernel=melhorKernel, C=melhorC, random_state=42)
		modelo.fit(xTreino, yTreino)

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

		exportar_modelo(modelo, normalizador_X, abordagem, colunas)

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
    normalizador_X = StandardScaler()
    xTreino = normalizador_X.fit_transform(xTreino)
    xValidacao = normalizador_X.transform(xValidacao)
    xTeste = normalizador_X.transform(xTeste)
    resultados = []

    for C in [0.1, 1, 10, 100]:
        for kernel in ["linear", "rbf"]:

            modelo = SVC(
                C=C,
                kernel=kernel,
                random_state=42,
            )

            modelo.fit(xTreino, yTreino)
            previsoes = modelo.predict(xValidacao)
            acuracia = accuracy_score(yValidacao, previsoes)
            resultados.append((C, kernel, acuracia))

    resultados.sort(key=lambda r: r[2], reverse=True)
    melhorC, melhorKernel, melhorAcuracia = resultados[0]

    modelo = SVC(
        C=melhorC,
        kernel=melhorKernel,
        random_state=42,
    )

    modelo.fit(xTreino, yTreino)

    return avaliar_modelo(modelo, xValidacao, yValidacao, xTeste, yTeste,)


if __name__ == "__main__":
	main()
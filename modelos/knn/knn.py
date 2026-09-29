import pandas as pd
from joblib import dump, load
from pathlib import Path
from sklearn import neighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "dataset" / "processed"
MODEL_PATH = BASE_DIR / "modelos" / "knn" / "modelo_ab2.joblib"
FEATURES_AB2 = [
	"qtd_x", "qtd_o", "pos_ocupadas", "linhas_2x", "linhas_2o",
	"casas_vazias", "vez",
]

def carregar_dados(abordagem):
	treino = pd.read_csv(DATA_DIR / f"{abordagem}_treino.csv")
	validacao = pd.read_csv(DATA_DIR / f"{abordagem}_validacao.csv")
	teste = pd.read_csv(DATA_DIR / f"{abordagem}_teste.csv")

	return (
		treino.drop(columns="classe"), treino["classe"],
		validacao.drop(columns="classe"), validacao["classe"],
		teste.drop(columns="classe"), teste["classe"],
	)


def exportar_modelo(modelo):
	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	dump({"modelo": modelo, "features": FEATURES_AB2}, MODEL_PATH)
	print(f"Modelo exportado para: {MODEL_PATH}")


def carregar_modelo():
	return load(MODEL_PATH)


def inferir(features):
	pacote = carregar_modelo()
	entrada = pd.DataFrame([features], columns=FEATURES_AB2)
	return pacote["modelo"].predict(entrada)[0]


def main():
	for abordagem in ("ab1", "ab2"):
		xTreino, yTreino, xValidacao, yValidacao, xTeste, yTeste = carregar_dados(abordagem)

		resultados = []
		for k in range(1, 21):
			modelo = neighbors.KNeighborsClassifier(n_neighbors=k)
			modelo.fit(xTreino, yTreino)
			previsoes = modelo.predict(xValidacao)
			acuracia = accuracy_score(yValidacao, previsoes)
			resultados.append((k, acuracia))

		resultados.sort(key=lambda r: r[1], reverse=True)
		melhorK, melhorAcuracia = resultados[0]

		print(f"\nAbordagem: {abordagem}")
		print("Busca de hiperparâmetro (n_neighbors) - top 3:")
		for k, acuracia in resultados[:3]:
			print(f"  n_neighbors={k}: acurácia={acuracia:.4f}")
		print(f"Escolhido: n_neighbors={melhorK} (acurácia={melhorAcuracia:.4f})")

		modelo = neighbors.KNeighborsClassifier(n_neighbors=melhorK)
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

		if abordagem == "ab2":
			exportar_modelo(modelo)


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
	for k in range(1, 21):
		modelo = neighbors.KNeighborsClassifier(n_neighbors=k)
		modelo.fit(xTreino, yTreino)
		previsoes = modelo.predict(xValidacao)
		acuracia = accuracy_score(yValidacao, previsoes)
		resultados.append((k, acuracia))

	resultados.sort(key=lambda r: r[1], reverse=True)
	melhorK, melhorAcuracia = resultados[0]
	modelo = neighbors.KNeighborsClassifier(n_neighbors=melhorK)
	modelo.fit(xTreino, yTreino)
	return avaliar_modelo(modelo, xValidacao, yValidacao, xTeste, yTeste)

if __name__ == "__main__":
	main()
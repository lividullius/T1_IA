import pandas as pd
from joblib import dump, load
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "dataset" / "processed"
MODEL_PATH = BASE_DIR / "modelos" / "mlp" / "modelo_ab2.joblib"
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


def exportar_modelo(modelo, normalizador):
	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	dump(
		{"modelo": modelo, "normalizador": normalizador, "features": FEATURES_AB2},
		MODEL_PATH,
	)
	print(f"Modelo exportado para: {MODEL_PATH}")


def carregar_modelo():
	return load(MODEL_PATH)


def inferir(features):
	pacote = carregar_modelo()
	entrada = pd.DataFrame([features], columns=FEATURES_AB2)
	entrada = pacote["normalizador"].transform(entrada)
	return pacote["modelo"].predict(entrada)[0]


def main():
	for abordagem in ("ab1", "ab2"):
		xTreino, yTreino, xValidacao, yValidacao, xTeste, yTeste = carregar_dados(abordagem)
		normalizador_X = StandardScaler()
		xTreino = normalizador_X.fit_transform(xTreino)
		xValidacao = normalizador_X.transform(xValidacao)
		xTeste = normalizador_X.transform(xTeste)

		resultados = []
		for camadas in [(10,), (20,), (50,)]:
			for taxa in [0.001, 0.01]:
				mlp = MLPClassifier(
					hidden_layer_sizes=camadas, activation="logistic", solver="adam",
					learning_rate_init=taxa, max_iter=3000, random_state=42,
				)
				mlp.fit(xTreino, yTreino)
				previsoes = mlp.predict(xValidacao)
				acuracia = accuracy_score(yValidacao, previsoes)
				resultados.append((camadas, taxa, acuracia))

		resultados.sort(key=lambda r: r[2], reverse=True)
		melhorCamadas, melhorTaxa, melhorAcuracia = resultados[0]

		print(f"\nAbordagem: {abordagem}")
		print("Busca de hiperparâmetro (hidden_layer_sizes, learning_rate_init) - top 3:")
		for camadas, taxa, acuracia in resultados[:3]:
			print(f"  hidden_layer_sizes={camadas}, learning_rate_init={taxa}: acurácia={acuracia:.4f}")
		print(f"Escolhido: hidden_layer_sizes={melhorCamadas}, learning_rate_init={melhorTaxa} (acurácia={melhorAcuracia:.4f})")

		mlp = MLPClassifier(
			hidden_layer_sizes=melhorCamadas, activation="logistic", solver="adam",
			learning_rate_init=melhorTaxa, max_iter=3000, random_state=42,
		)
		mlp.fit(xTreino, yTreino)

		for nome, features, classes in (
			("Validação", xValidacao, yValidacao),
			("Teste", xTeste, yTeste),
		):
			previsoes = mlp.predict(features)
			print(f"{nome}:")
			print("Acurácia:", accuracy_score(classes, previsoes))
			print("Precisão:", precision_score(classes, previsoes, average="weighted"))
			print("Recall:", recall_score(classes, previsoes, average="weighted"))
			print("F-measure:", f1_score(classes, previsoes, average="weighted"))

		if abordagem == "ab2":
			exportar_modelo(mlp, normalizador_X)

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

	for camadas in [(10,), (20,), (50,)]:
		for taxa in [0.001, 0.01]:
			mlp = MLPClassifier(
				hidden_layer_sizes=camadas, activation="logistic", solver="adam",
				learning_rate_init=taxa, max_iter=3000, random_state=42,
			)
			mlp.fit(xTreino, yTreino)
			previsoes = mlp.predict(xValidacao)
			acuracia = accuracy_score(yValidacao, previsoes)
			resultados.append((camadas, taxa, acuracia))

	resultados.sort(key=lambda r: r[2], reverse=True)
	melhorCamadas, melhorTaxa, melhorAcuracia = resultados[0]

	mlp = MLPClassifier(
		hidden_layer_sizes=melhorCamadas, activation="logistic", solver="adam",
		learning_rate_init=melhorTaxa, max_iter=3000, random_state=42,
	)
	mlp.fit(xTreino, yTreino)

	return avaliar_modelo(mlp, xValidacao, yValidacao, xTeste, yTeste)


if __name__ == "__main__":
	main()

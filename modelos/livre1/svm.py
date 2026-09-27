import pandas as pd
from joblib import dump, load
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "dataset" / "processed"
MODEL_PATH = BASE_DIR / "modelos" / "livre1" / "modelo_ab2.joblib"
FEATURES_AB2 = [
	"qtd_x", "qtd_o", "pos_ocupadas", "linhas_2x", "linhas_2o",
	"casas_vazias", "vez",
]

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
# A abordagem nova substitui o dataset antigo pelos CSVs processados de ab1
# e ab2, mas mantem a normalizacao: fit_transform no treino e transform nos
# conjuntos de validacao e teste.


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


def main():
	for abordagem in ("ab1", "ab2"):
		xTreino, yTreino, xValidacao, yValidacao, xTeste, yTeste = carregar_dados(abordagem)
		normalizador_X = StandardScaler()
		xTreino = normalizador_X.fit_transform(xTreino)
		xValidacao = normalizador_X.transform(xValidacao)
		xTeste = normalizador_X.transform(xTeste)

		modelo = SVC(kernel="rbf", C=10, random_state=42)
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

		if abordagem == "ab2":
			exportar_modelo(modelo, normalizador_X)


if __name__ == "__main__":
	main()
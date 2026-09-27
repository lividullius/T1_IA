import pandas as pd
from pathlib import Path
from sklearn import neighbors
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
# Essa abordagem usava um dataset antigo, tres classes e uma divisao nova.
# A abordagem atual carrega os seis CSVs ja divididos pelo preprocessing,
# permitindo comparar ab1 e ab2 com exatamente as mesmas particoes.


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

	melhorK = 5
	modelo = neighbors.KNeighborsClassifier(n_neighbors=melhorK)
	modelo.fit(xTreino, yTreino)

	previsoes = modelo.predict(xValidacao)
	print(f"\nAbordagem: {abordagem}")
	print("Validação:")
	print("Acurácia:", accuracy_score(yValidacao, previsoes))
	print("Precisão:", precision_score(yValidacao, previsoes, average="weighted"))
	print("Recall:", recall_score(yValidacao, previsoes, average="weighted"))
	print("F-measure:", f1_score(yValidacao, previsoes, average="weighted"))

	previsoes = modelo.predict(xTeste)
	print("Teste:")
	print("Acurácia:", accuracy_score(yTeste, previsoes))
	print("Precisão:", precision_score(yTeste, previsoes, average="weighted"))
	print("Recall:", recall_score(yTeste, previsoes, average="weighted"))
	print("F-measure:", f1_score(yTeste, previsoes, average="weighted"))
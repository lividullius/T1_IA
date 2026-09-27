import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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
# Depois, o scaler era ajustado sobre essa divisao unica.
# Agora cada abordagem (ab1 e ab2) tem treino, validacao e teste proprios,
# e o scaler continua sendo ajustado somente no conjunto de treino.


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

	normalizador_X = StandardScaler()
	xTreino = normalizador_X.fit_transform(xTreino)
	xValidacao = normalizador_X.transform(xValidacao)
	xTeste = normalizador_X.transform(xTeste)

	mlp = MLPClassifier(
		hidden_layer_sizes=(20,), activation="logistic", solver="adam",
		learning_rate_init=0.001, max_iter=3000, random_state=42,
	)
	mlp.fit(xTreino, yTreino)

	print(f"\nAbordagem: {abordagem}")
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

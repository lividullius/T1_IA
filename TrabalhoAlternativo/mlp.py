import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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

#Normalizar os dados
normalizador_X = StandardScaler()
xTreino = normalizador_X.fit_transform(xTreino)
xValidacao = normalizador_X.transform(xValidacao)
xTeste = normalizador_X.transform(xTeste)

mlp = MLPClassifier(hidden_layer_sizes=(20,), activation='logistic', solver='adam', learning_rate_init=0.001, max_iter=3000, random_state=42)

#Treinar o modelo
mlp.fit(xTreino, yTreino)
yPredValidacao = mlp.predict(xValidacao)
print("\nPrimeiras previsões:")
print(yPredValidacao[:10])
print("\nResultados reais:")
print(yValidacao[:10])

#Calcular as métricas da validação
acuracia = accuracy_score(yValidacao, yPredValidacao)
precisao = precision_score(yValidacao, yPredValidacao, average="weighted")
recall = recall_score(yValidacao, yPredValidacao, average="weighted")
f1 = f1_score(yValidacao, yPredValidacao, average="weighted")

print("\nResultados da validação:")
print("Acurácia:", acuracia)
print("Precisão:", precisao)
print("Recall:", recall)
print("F-measure:", f1)

yPredTeste = mlp.predict(xTeste)

acuracia = accuracy_score(yTeste, yPredTeste)
precisao = precision_score(yTeste, yPredTeste, average="weighted")
recall = recall_score(yTeste, yPredTeste, average="weighted")
f1 = f1_score(yTeste, yPredTeste, average="weighted")

print("\nResultados do teste:")
print("Acurácia:", acuracia)
print("Precisão:", precisao)
print("Recall:", recall)
print("F-measure:", f1)

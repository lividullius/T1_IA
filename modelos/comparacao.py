from knn import knn
from mlp import mlp
from arvore import arvoresDecisao
from livre1 import svm
from livre2 import randomForest

abordagem = "ab2"
resultadoKNN = knn.executar(abordagem)
resultadoMLP = mlp.executar(abordagem)
resultadoArvore = arvoresDecisao.executar(abordagem)
resultadoSVM = svm.executar(abordagem)
resultadoRandomForest = randomForest.executar(abordagem)
print("\nKNN:")
print(resultadoKNN)
print("\nMLP:")
print(resultadoMLP)
print("\nArvores de Decisão")
print(resultadoArvore)
print("\nSVM")
print(resultadoSVM)
print("\nRandom Forest")
print(resultadoRandomForest)
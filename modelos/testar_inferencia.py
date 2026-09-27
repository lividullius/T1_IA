import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modelos.arvore.arvoresDecisao import inferir as inferir_arvore
from modelos.knn.knn import inferir as inferir_knn
from modelos.livre1.svm import inferir as inferir_svm
from modelos.livre2.randomForest import inferir as inferir_random_forest
from modelos.mlp.mlp import inferir as inferir_mlp


ENTRADAS = [
    ({"qtd_x": 3, "qtd_o": 3, "pos_ocupadas": 6, "linhas_2x": 1, "linhas_2o": 0, "casas_vazias": 3, "vez": 1}, "O venceu"),
    ({"qtd_x": 4, "qtd_o": 4, "pos_ocupadas": 8, "linhas_2x": 1, "linhas_2o": 0, "casas_vazias": 1, "vez": 1}, "O venceu"),
    ({"qtd_x": 5, "qtd_o": 4, "pos_ocupadas": 9, "linhas_2x": 0, "linhas_2o": 0, "casas_vazias": 0, "vez": 0}, "Empate"),
    ({"qtd_x": 3, "qtd_o": 3, "pos_ocupadas": 6, "linhas_2x": 0, "linhas_2o": 0, "casas_vazias": 3, "vez": 1}, "O venceu"),
    ({"qtd_x": 5, "qtd_o": 4, "pos_ocupadas": 9, "linhas_2x": 0, "linhas_2o": 0, "casas_vazias": 0, "vez": 0}, "Empate"),
    ({"qtd_x": 3, "qtd_o": 2, "pos_ocupadas": 5, "linhas_2x": 2, "linhas_2o": 1, "casas_vazias": 4, "vez": 0}, "Tem jogo"),
    ({"qtd_x": 3, "qtd_o": 2, "pos_ocupadas": 5, "linhas_2x": 0, "linhas_2o": 0, "casas_vazias": 4, "vez": 0}, "Tem jogo"),
    ({"qtd_x": 4, "qtd_o": 3, "pos_ocupadas": 7, "linhas_2x": 3, "linhas_2o": 0, "casas_vazias": 2, "vez": 0}, "Tem jogo"),
    ({"qtd_x": 4, "qtd_o": 3, "pos_ocupadas": 7, "linhas_2x": 1, "linhas_2o": 1, "casas_vazias": 2, "vez": 0}, "X venceu"),
]

MODELOS = {
    "KNN": inferir_knn,
    "MLP": inferir_mlp,
    "Arvore de decisao": inferir_arvore,
    "SVM": inferir_svm,
    "Random Forest": inferir_random_forest,
}

acertou = 0

for numero, (entrada, classe_esperada) in enumerate(ENTRADAS, start=1):
    print(f"\nEntrada {numero} | Classe esperada: {classe_esperada}")
    print(entrada)
    for nome_modelo, inferir in MODELOS.items():
        print(f"  {nome_modelo}: {inferir(entrada)}")
        acertou += (inferir(entrada) == classe_esperada)

print(f"\nTotal de acertos para os modelos: {acertou} de {len(ENTRADAS)*len(MODELOS)} ({acertou/(len(ENTRADAS)*len(MODELOS))*100:.2f}%)")   
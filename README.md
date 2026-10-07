# T1_IA
Repositório para desenvolvimento do T1 de IA 2026/2.

Classificação do estado de um tabuleiro de jogo da velha em 4 classes
(`X venceu`, `O venceu`, `Empate`, `Tem jogo`) usando KNN, MLP, Árvore de
Decisão, SVM e Random Forest.

## Requisitos

- Python 3.10+
- pandas >= 2.2, < 3
- scikit-learn >= 1.5, < 2

Instalação:

```bash
pip install -r requirements.txt
```

## Estrutura

```
dataset/
  raw/          dataset original da UCI (tic-tac-toe.data)
  processed/    CSVs gerados pelos scripts 
  explorar_dataset.py
  construir_dataset.py
  preprocessamento.py
  dividir_dataset.py
modelos/
  arvore/  knn/  mlp/  livre1/ (SVM)  livre2/ (Random Forest)
  testar_inferencia.py
frontend/
  jogo.py       jogo no terminal (humano contra a máquina aleatória)
```

## Como rodar

Os CSVs de `dataset/processed/` já estão no repositório, então não precisa gerar os dados para treinar os modelos. Se quiser gerar de novo (o resultado é sempre o mesmo, porque a seed é fixa):

```bash
cd dataset
python3 construir_dataset.py   # raw -> processed/dataset_4classes.csv
python3 dividir_dataset.py     # -> processed/ab1_*.csv e ab2_*.csv
```

- `ab1`: 9 features, uma por casa (`x`=1, `o`=-1, `b`=0)
- `ab2`: 7 features derivadas (contagens e linhas com 2 peças)

Treinar os modelos. Cada script testa as duas abordagens e salva `modelo_ab1.joblib` e `modelo_ab2.joblib` na própria pasta. Esses arquivos não entram no repositório:

```bash
python3 modelos/knn/knn.py
python3 modelos/mlp/mlp.py
python3 modelos/arvore/arvoresDecisao.py
python3 modelos/livre1/svm.py
python3 modelos/livre2/randomForest.py
```

Testar a inferência dos modelos treinados (abordagem 2):

```bash
python3 modelos/testar_inferencia.py
```

Jogar no terminal. O jogo pede o algoritmo e a abordagem; a IA classifica o estado a cada jogada:

```bash
python3 frontend/jogo.py
```


---
# T1 IA – Tic-Tac-Toe com ML: Plano de Implementação

Plano completo de implementação do trabalho prático de IA com classificação de estados do jogo da velha usando 5 algoritmos de ML, pré-processamento em 2 abordagens e frontend de jogo.

---

### Resumo do dataset original
- 958 instâncias, sem valores faltantes, sem duplicatas
- 2 classes originais: `positive` (X venceu, 626) e `negative` (332)
  - Das 332 `negative`: O venceu = 316, Empate = 16, "Tem jogo" = 0
- **Problema**: Precisamos de 4 classes, mas o dataset só tem 2. "Tem jogo" não existe e "Empate" tem apenas 16 instâncias → precisam ser sintetizados/gerados.

---

## Passos do Enunciado

| Passo | Descrição | Status |
|---|---|---|
| 1 | Objetivo: classificar em 4 estados | 
| 2 | Dataset: análise e adequação | 
| 3 | Pré-processamento (2 abordagens) | 
| 4 | Divisão treino/validação/teste | Feito (`dataset/dividir_dataset.py`) 
| 5 | 5 algoritmos classificadores | 
| 6 | Frontend mínimo | 

---

## Plano de Implementação

### Passo 2 + 3 — Dataset e Pré-processamento

#### 2.1 — Geração das 4 classes (arquivo: `dataset/construir_dataset.py`)

**Problema central**: o dataset UCI só tem `endgames` — não há estados "em andamento" (Tem jogo) e há apenas 16 empates. Solução:

| Classe | Fonte | Qtd alvo |
|---|---|---|
| X venceu | Dataset UCI (`positive`) — amostrar 200 | 200 |
| O venceu | Dataset UCI (`negative` com O vencendo) — amostrar 200 | 200 |
| Empate | 16 reais + gerar tabuleiros completos sem vencedor por força bruta | 200 |
| Tem jogo | Gerar tabuleiros incompletos válidos (sem vencedor ainda) | 200 |

**Geração de "Empate" extra**: enumerar combinações de tabuleiros 3x3 completos sem nenhum 3-em-linha, respeitando a regra de contagem x/o (x==o ou x==o+1).

**Geração de "Tem jogo"**: gerar tabuleiros com 1 a 7 peças jogadas, sem vencedor definido, com pelo menos 1 célula vazia.

#### 2.2 — Abordagem 1 de pré-processamento

Codificação das 9 células do tabuleiro:
- `x` → 1, `o` → -1, `b` → 0

9 features numéricas + label

#### 2.3 — Abordagem 2 de pré-processamento

7 features derivadas:
- `qtd_x` — quantidade de X no tabuleiro
- `qtd_o` — quantidade de O no tabuleiro
- `pos_ocupadas` — posições ocupadas
- `linhas_2x` — número de linhas/colunas/diagonais com exatamente 2 X e nenhum O
- `linhas_2o` — número de linhas/colunas/diagonais com exatamente 2 O e nenhum X
- `casas_vazias` — número de casas com `b`
- `vez` — jogador da vez (1=X, 0=O)

#### Passo 4 — Divisão do dataset (arquivo: `dataset/dividir_dataset.py`)

**Entrada**: `dataset/processed/dataset_4classes.csv` (800 linhas geradas pelo
`construir_dataset.py`, colunas do tabuleiro + `classe` + `origem`).

**Ordem das operações**: dividir primeiro o tabuleiro bruto em
treino/validação/teste, e só depois aplicar as codificações da abordagem 1
(2.2) e abordagem 2 (2.3) sobre cada partição. Isso garante que os mesmos
800 tabuleiros (e a mesma composição de treino/validação/teste) sejam usados
nas duas abordagens — senão a comparação entre abordagens no passo 5 fica
inválida (cada uma teria visto exemplos diferentes).

- **Split estratificado por `classe`: 60% treino / 20% validação / 20% teste**
  (120/40/40 por classe, já que são 200 amostras cada)
- **Seed fixa (mesma do `construir_dataset.py`, 42)** para reprodutibilidade
- **Cuidado com vazamento de dados na classe "Empate"**: como só existem 16
  tabuleiros de empate únicos (ver `entendendo-o-dataset.md`), as 200 linhas
  dessa classe são 184 duplicatas exatas dos mesmos 16 tabuleiros. Um split
  aleatório ingênuo por linha pode colocar cópias do **mesmo tabuleiro** em
  treino e teste ao mesmo tempo, inflando artificialmente a acurácia nessa
  classe. Solução: fazer o split por **grupo** (agrupando por tabuleiro
  único, não por linha) para a classe "Empate" — todas as cópias de um
  mesmo tabuleiro caem inteiramente em treino, validação **ou** teste, nunca
  espalhadas entre partições. As outras 3 classes não têm esse problema
  (linhas já são tabuleiros distintos) e podem usar split estratificado
  normal.
- Os mesmos splits (mesmos índices de linha) são usados por todos os 5
  algoritmos e pelas duas abordagens de pré-processamento
- Salvar como CSV em `dataset/processed/`:
  - `ab1_treino.csv`, `ab1_validacao.csv`, `ab1_teste.csv`
  - `ab2_treino.csv`, `ab2_validacao.csv`, `ab2_teste.csv`

---

### Passo 5 — Algoritmos de IA

Cada algoritmo é implementado na sua pasta, testando **ambas as abordagens** de pré-processamento. O conjunto de validação é usado para tunar hiperparâmetros. O conjunto de teste é usado apenas para avaliação final.

Métricas: **acurácia, precision, recall, F-measure** (por classe e macro).

#### 5.1 — k-NN (`modelos/knn/knn.py`)
- Hiperparâmetros: k ∈ {1, 3, 5, 7, 9, 11, 15}, métrica de distância (euclidean, manhattan)

#### 5.2 — MLP (`modelos/mlp/mlp.py`)
- Topologia a definir (ex: 9→64→32→4 para abordagem 1)
- Hiperparâmetros: hidden_layer_sizes, activation, solver, learning_rate
- Reportar topologia final no relatório

#### 5.3 — Árvore de Decisão (`modelos/arvore/arvore.py`)
- Hiperparâmetros: criterion (gini/entropy), max_depth, min_samples_split
- Gerar visualização da árvore

#### 5.4 — SVM (`modelos/livre1/svm.py`)
- Hiperparâmetros: kernel (rbf, linear), C, gamma
- Explicação: hiperplano de máxima margem em espaço de alta dimensão

#### 5.5 — Random Forest (`modelos/livre2/rf.py`)
- Hiperparâmetros: n_estimators, max_depth
- Explicação: ensemble de árvores com bagging e amostragem de features

#### 5.6 — Comparação (`modelos/comparar.py`)
- Tabela comparativa de todos os algoritmos × ambas abordagens
- Gráficos de barras das métricas
- Conclusão: melhor algoritmo + melhor abordagem de pré-processamento

---

### Passo 6 — Frontend (`frontend/jogo.py`)

Frontend **console-based** com:
- Tabuleiro desenhado em texto a cada jogada
- Player humano: entrada de coordenada (ex: "1 2")
- Máquina: jogada aleatória (posição livre ao acaso)
- Após cada jogada: classificação pelo melhor modelo da IA
- **Comportamento especial** (conforme enunciado):
  - Se IA detecta "fim de jogo" incorretamente → encerrar o jogo (erro contabilizado)
  - Se IA não detecta "fim de jogo" real → continuar o jogo (erro contabilizado)
- Exibir score ao final: acertos, erros, acurácia da IA durante a sessão

---



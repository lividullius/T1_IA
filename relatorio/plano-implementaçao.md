---
agent: devin-local
session: destiny-waitress
created: 2026-09-22T12:22:17Z
---
# T1 IA – Tic-Tac-Toe com ML: Plano de Implementação

Plano completo de implementação do trabalho prático de IA com classificação de estados do jogo da velha usando 5 algoritmos de ML, pré-processamento em 2 abordagens e frontend de jogo.

---

## Contexto e estado atual

### O que já foi feito
- Dataset UCI baixado (`dataset/raw/tic-tac-toe.data`, `tic-tac-toe.names`)
- Script de exploração do dataset (`dataset/explorar_dataset.py`)
- Relatório de análise do dataset (`relatorio/entendendo-o-dataset.md`)
- Estrutura de pastas criada: `modelos/arvore`, `modelos/knn`, `modelos/mlp`, `modelos/livre1`, `modelos/livre2`, `frontend/`, `relatorio/`

### Resumo do dataset original
- 958 instâncias, sem valores faltantes, sem duplicatas
- 2 classes originais: `positive` (X venceu, 626) e `negative` (332)
  - Das 332 `negative`: O venceu = 316, Empate = 16, "Tem jogo" = 0
- **Problema**: Precisamos de 4 classes, mas o dataset só tem 2. "Tem jogo" não existe e "Empate" tem apenas 16 instâncias → precisam ser sintetizados/gerados.

---

## Passos do Enunciado

| Passo | Descrição | Status |
|---|---|---|
| 1 | Objetivo: classificar em 4 estados | Entendido |
| 2 | Dataset: análise e adequação | Parcialmente feito (análise) |
| 3 | Pré-processamento (2 abordagens) | A fazer |
| 4 | Divisão treino/validação/teste | A fazer |
| 5 | 5 algoritmos classificadores | A fazer |
| 6 | Frontend mínimo | A fazer |

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

#### 2.4 — Divisão do dataset (arquivo: `dataset/dividir_dataset.py`)

- **Split físico estratificado: 60% treino / 20% validação / 20% teste**
- Os mesmos splits são usados por todos os 5 algoritmos
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

## Estrutura de Arquivos (resultado final)

```
T1_IA/
├── dataset/
│   ├── raw/                        (já existe)
│   ├── processed/
│   │   ├── dataset_completo.csv
│   │   ├── ab1_treino.csv
│   │   ├── ab1_validacao.csv
│   │   ├── ab1_teste.csv
│   │   ├── ab2_treino.csv
│   │   ├── ab2_validacao.csv
│   │   └── ab2_teste.csv
│   ├── explorar_dataset.py         (já existe)
│   ├── construir_dataset.py        (novo)
│   ├── preprocessamento.py         (novo)
│   └── dividir_dataset.py          (novo)
├── modelos/
│   ├── knn/knn.py
│   ├── mlp/mlp.py
│   ├── arvore/arvore.py
│   ├── livre1/svm.py
│   ├── livre2/rf.py
│   └── comparar.py
├── frontend/
│   └── jogo.py
└── relatorio/
    ├── entendendo-o-dataset.md     (já existe)
    └── ...
```

---

## Ordem de Implementação

1. `dataset/construir_dataset.py` — gerar 4 classes balanceadas (~200 por classe)
2. `dataset/preprocessamento.py` — codificar ambas as abordagens
3. `dataset/dividir_dataset.py` — split 60/20/20 estratificado
4. `modelos/arvore/arvore.py` — modelo interpretável, bom ponto de partida
5. `modelos/knn/knn.py`
6. `modelos/mlp/mlp.py`
7. `modelos/livre1/svm.py`
8. `modelos/livre2/rf.py`
9. `modelos/comparar.py` — comparação e gráficos
10. `frontend/jogo.py` — frontend com melhor modelo integrado

---

## Pontuação e Critérios

| Item | Pontos |
|---|---|
| Dataset (documentado) | 1,0 |
| 5 soluções de IA (1,0 por algoritmo) | 5,0 |
| Frontend | 1,0 |
| Relatório PPT | 1,5 |
| Vídeo (máx. 10 min, todos falam) | 1,5 |
| **Total** | **10,0** |

---

## Observações Críticas

- Dataset balanceado: ~200 instâncias por classe (800 total)
- Mesmo split de dados para todos os algoritmos
- Evitar overfitting — justificar parâmetros no relatório
- Registrar acertos/erros da IA no frontend
- MLP: documentar topologia usada
- Dois modelos livres (SVM e Random Forest): incluir explicação de como funcionam
- Relatório no formato PPT
- Vídeo de até 10 min com todos os integrantes presentes e falando
- Indicar quais ferramentas de IA foram usadas no relatório

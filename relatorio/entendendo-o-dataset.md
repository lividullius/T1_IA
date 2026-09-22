# Entendendo o dataset

Dataset: [Tic-Tac-Toe Endgame](https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame) (UCI Machine Learning Repository).
Script de exploração: `dataset/explorar_dataset.py`.

## Formato bruto

- CSV sem cabeçalho, **958 linhas**, 10 colunas.
- As 9 primeiras colunas são as casas do tabuleiro (`top-left` até `bottom-right`). Valores: `x`, `o`, `b` (vazio).
- A última coluna é `class`: `positive` (X venceu) ou `negative` (X não venceu).

## Integridade

| Checagem | Resultado |
|---|---|
| Total de linhas | **958** |
| Valores inválidos nas células | **0** |
| Classes inválidas | **0** |
| Linhas duplicadas | **0** |
| Valores faltantes | **0** |

Dataset limpo — nenhuma adequação necessária.

## Distribuição das classes

| Classe | Quantidade | % |
|---|---|---|
| `positive` (X venceu) | 626 | 65,3% |
| `negative` (X não venceu) | 332 | 34,7% |

Dataset desbalanceado. As 332 instâncias `negative` se dividem em:

| Subcategoria | Quantidade |
|---|---|
| O venceu | 316 |
| Empate (tabuleiro cheio, sem vencedor) | 16 |
| "Tem jogo" (em andamento) | **0** |

## Problema: o enunciado pede 4 classes

O dataset original tem apenas 2 classes, mas precisamos de 4:

| Classe necessária | Origem | Disponível |
|---|---|---|
| X venceu | `positive` do dataset | 626 → amostrar 200 |
| O venceu | `negative` com O vencendo | 316 → amostrar 200 |
| Empate | `negative` com tabuleiro cheio e sem vencedor | **apenas 16** → precisar gerar mais |
| Tem jogo | Não existe no dataset (só contém endgames) | **0** → gerar sinteticamente |

As classes "Empate" e "Tem jogo" precisarão ser geradas no passo de construção do dataset (`dataset/construir_dataset.py`).

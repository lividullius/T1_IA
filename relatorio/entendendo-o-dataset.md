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

## Descoberta: só existem 16 empates possíveis no total

Ao implementar a geração de "Empate" por força bruta, verificamos que **16 é o
total de tabuleiros completos (9 casas, 5 X / 4 O) sem vencedor
matematicamente possíveis** no jogo da velha — não é uma limitação do
dataset UCI, é o universo combinatório inteiro. Confirmado de duas formas:

1. Enumeração por força bruta de todas as `C(9,5) = 126` formas de
   distribuir 5 X's e 4 O's no tabuleiro, filtrando as que não têm
   3-em-linha para nenhum jogador → sempre 16.
2. Aplicando as 8 simetrias do tabuleiro (rotações/reflexões) sobre esses
   16 tabuleiros, o conjunto resultante continua tendo 16 elementos — ou
   seja, esse grupo de tabuleiros já é fechado sob simetria, não gera
   variantes novas.

**Decisão**: a classe "Empate" é completada até 200 amostras por
**oversampling** (reamostragem com reposição) dos 16 tabuleiros reais, em
vez de gerar tabuleiros novos (que não existem). Isso é registrado na
coluna `origem` do dataset gerado (`oversample_16_reais`), para
transparência e para possível discussão no relatório sobre o efeito da
duplicação nessa classe durante treino/avaliação.

Já a classe "Tem jogo" tem um universo bem maior (4297 tabuleiros válidos
com 1 a 7 peças, sem vencedor ainda, respeitando a regra de turnos), então
foi possível amostrar 200 tabuleiros únicos sem repetição, distribuídos de
forma estratificada entre as quantidades de peças jogadas.

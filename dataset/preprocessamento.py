"""Duas abordagens de pré-processamento para o dataset de 4 classes.

Aqui só ficam as *funções* de codificação (aplicadas linha a linha). A
divisão treino/validação/teste é feita separadamente em
`dividir_dataset.py`, que importa este módulo e aplica cada abordagem
**depois** de dividir os tabuleiros brutos — assim as duas abordagens usam
exatamente os mesmos exemplos em cada partição, tornando a comparação entre
elas (passo 5) válida.

Abordagem 1 (`codificar_abordagem1`): 9 features numéricas, uma por casa
do tabuleiro (`x`→1, `o`→-1, `b`→0).

Abordagem 2 (`codificar_abordagem2`): 7 features derivadas do tabuleiro
(contagens e padrões), descritas em `plano-implementaçao.md`.
"""
import csv
from pathlib import Path

from explorar_dataset import BOARD_COLS, WIN_LINES

IN_PATH = Path(__file__).parent / "processed" / "dataset_4classes.csv"

MAPA_ABORDAGEM1 = {"x": 1, "o": -1, "b": 0}

FEATURES_ABORDAGEM1 = BOARD_COLS
FEATURES_ABORDAGEM2 = [
    "qtd_x", "qtd_o", "pos_ocupadas",
    "linhas_2x", "linhas_2o", "casas_vazias", "vez",
]


def codificar_abordagem1(board):
    """9 células do tabuleiro codificadas como -1/0/1."""
    return {col: MAPA_ABORDAGEM1[valor] for col, valor in zip(BOARD_COLS, board)}


def _linhas_com_dois_e_nenhum(board, jogador, oponente):
    return sum(
        1 for linha in WIN_LINES
        if sum(1 for i in linha if board[i] == jogador) == 2
        and sum(1 for i in linha if board[i] == oponente) == 0
    )


def codificar_abordagem2(board):
    """7 features derivadas: contagens de peças e padrões de ameaça."""
    qtd_x = board.count("x")
    qtd_o = board.count("o")
    return {
        "qtd_x": qtd_x,
        "qtd_o": qtd_o,
        "pos_ocupadas": qtd_x + qtd_o,
        "linhas_2x": _linhas_com_dois_e_nenhum(board, "x", "o"),
        "linhas_2o": _linhas_com_dois_e_nenhum(board, "o", "x"),
        "casas_vazias": board.count("b"),
        "vez": 1 if qtd_x == qtd_o else 0,
    }


def load_dataset_4classes():
    with open(IN_PATH, newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = load_dataset_4classes()

    print(f"Lidas {len(rows)} linhas de {IN_PATH}\n")

    exemplo = rows[0]
    board = [exemplo[c] for c in BOARD_COLS]
    print(f"Exemplo (classe={exemplo['classe']}):")
    print(f"  tabuleiro bruto : {board}")
    print(f"  abordagem 1     : {codificar_abordagem1(board)}")
    print(f"  abordagem 2     : {codificar_abordagem2(board)}\n")

    print("Checagens de sanidade (abordagem 2, todas as linhas):")
    for r in rows:
        board = [r[c] for c in BOARD_COLS]
        feats = codificar_abordagem2(board)
        assert feats["pos_ocupadas"] == 9 - feats["casas_vazias"]
        assert feats["qtd_x"] - feats["qtd_o"] in (0, 1)
        assert (feats["vez"] == 1) == (feats["qtd_x"] == feats["qtd_o"])
    print("  OK — nenhuma inconsistência encontrada")


if __name__ == "__main__":
    main()

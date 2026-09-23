"""Construção do dataset com as 4 classes de estado do jogo da velha.

O dataset UCI só contém endgames, então faltam 2 das 4 classes exigidas:
- "Tem jogo": 0 instâncias no dataset original -> geradas aqui.
- "Empate": só 16 instâncias no dataset original, e são as ÚNICAS 16
  configurações de empate que existem matematicamente (ver
  entendendo-o-dataset.md) -> completadas até 200 por oversampling.

Saída: dataset/processed/dataset_4classes.csv, 200 linhas por classe (800
no total).
"""
import csv
import itertools
import random
from collections import Counter
from pathlib import Path

from explorar_dataset import BOARD_COLS, has_win, load_rows

SEED = 42
QTD_POR_CLASSE = 200
OUT_PATH = Path(__file__).parent / "processed" / "dataset_4classes.csv"


def board_para_linha(board, classe, origem):
    linha = dict(zip(BOARD_COLS, board))
    linha["classe"] = classe
    linha["origem"] = origem
    return linha


def amostrar_vencedores(rows, rng):
    """X venceu / O venceu: amostrados direto do dataset UCI."""
    positivas = [r for r in rows if r["class"] == "positive"]
    negativas_o = [
        r for r in rows
        if r["class"] == "negative" and has_win([r[c] for c in BOARD_COLS], "o")
    ]

    linhas = []
    for r in rng.sample(positivas, QTD_POR_CLASSE):
        linhas.append(board_para_linha([r[c] for c in BOARD_COLS], "X venceu", "uci_real"))
    for r in rng.sample(negativas_o, QTD_POR_CLASSE):
        linhas.append(board_para_linha([r[c] for c in BOARD_COLS], "O venceu", "uci_real"))
    return linhas


def gerar_empates_unicos():
    """Todos os tabuleiros completos (5 X / 4 O) sem vencedor. Dá sempre
    16 — é o total de empates possíveis no jogo da velha, não uma
    amostra."""
    empates = []
    for xs in itertools.combinations(range(9), 5):
        board = ["o"] * 9
        for i in xs:
            board[i] = "x"
        if not has_win(board, "x") and not has_win(board, "o"):
            empates.append(board)
    return empates


def amostrar_empates(rng):
    unicos = gerar_empates_unicos()
    return [
        board_para_linha(rng.choice(unicos), "Empate", "oversample_16_reais")
        for _ in range(QTD_POR_CLASSE)
    ]


def gerar_tem_jogo():
    """Tabuleiros incompletos (1 a 7 peças), sem vencedor ainda,
    respeitando a regra de turnos (X sempre joga primeiro: nx == no ou
    nx == no + 1)."""
    boards = []
    for k in range(1, 8):
        nx = (k + 1) // 2 if k % 2 else k // 2
        for cells in itertools.combinations(range(9), k):
            for xs in itertools.combinations(cells, nx):
                board = ["b"] * 9
                for i in cells:
                    board[i] = "x" if i in xs else "o"
                if not has_win(board, "x") and not has_win(board, "o"):
                    boards.append(board)
    return boards


def amostrar_tem_jogo(rng):
    todos = gerar_tem_jogo()
    return [
        board_para_linha(b, "Tem jogo", "gerado")
        for b in rng.sample(todos, QTD_POR_CLASSE)
    ]


def main():
    rng = random.Random(SEED)
    rows = load_rows()

    linhas = amostrar_vencedores(rows, rng) + amostrar_empates(rng) + amostrar_tem_jogo(rng)
    rng.shuffle(linhas)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BOARD_COLS + ["classe", "origem"])
        writer.writeheader()
        writer.writerows(linhas)

    print(f"Dataset salvo em {OUT_PATH} ({len(linhas)} linhas)")
    print(Counter(l["classe"] for l in linhas))


if __name__ == "__main__":
    main()

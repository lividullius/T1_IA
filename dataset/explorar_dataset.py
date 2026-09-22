"""Exploração e checagens de integridade do dataset bruto tic-tac-toe.data."""
import csv
from collections import Counter
from pathlib import Path

RAW_PATH = Path(__file__).parent / "raw" / "tic-tac-toe.data"

COLUMNS = [
    "top-left", "top-middle", "top-right",
    "middle-left", "middle-middle", "middle-right",
    "bottom-left", "bottom-middle", "bottom-right",
    "class",
]
BOARD_COLS = COLUMNS[:9]

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # linhas
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # colunas
    (0, 4, 8), (2, 4, 6),             # diagonais
]


def load_rows():
    with open(RAW_PATH, newline="") as f:
        return [dict(zip(COLUMNS, row)) for row in csv.reader(f) if row]


def has_win(board, player):
    return any(all(board[i] == player for i in line) for line in WIN_LINES)


def checar_integridade(rows):
    print("=== Integridade ===")
    print(f"Total de linhas: {len(rows)}")

    invalidos = sum(
        1 for r in rows for c in BOARD_COLS if r[c] not in ("x", "o", "b")
    )
    print(f"Valores inválidos nas células: {invalidos}")

    classes_invalidas = sum(1 for r in rows if r["class"] not in ("positive", "negative"))
    print(f"Classes inválidas: {classes_invalidas}")

    tuplas = [tuple(r[c] for c in COLUMNS) for r in rows]
    duplicadas = sum(v - 1 for v in Counter(tuplas).values() if v > 1)
    print(f"Linhas duplicadas extras: {duplicadas}")

    faltantes = sum(1 for r in rows for c in COLUMNS if not r[c])
    print(f"Valores faltantes: {faltantes}")


def distribuicao_classes(rows):
    print("\n=== Distribuição de classes ===")
    dist = Counter(r["class"] for r in rows)
    total = len(rows)
    for classe, qtd in dist.items():
        print(f"  {classe}: {qtd} ({100 * qtd / total:.1f}%)")


def checar_regras_jogo(rows):
    print("\n=== Consistência das regras ===")
    positivas = [r for r in rows if r["class"] == "positive"]
    negativas  = [r for r in rows if r["class"] == "negative"]

    sem_vitoria_x = sum(
        1 for r in positivas if not has_win([r[c] for c in BOARD_COLS], "x")
    )
    print(f"'positive' sem vitória de X: {sem_vitoria_x}  (esperado: 0)")

    o_venceu = empate = tem_jogo = 0
    for r in negativas:
        board = [r[c] for c in BOARD_COLS]
        if has_win(board, "o"):
            o_venceu += 1
        elif "b" not in board:
            empate += 1
        else:
            tem_jogo += 1

    print(f"'negative' -> O venceu: {o_venceu} | empate: {empate} | tem jogo: {tem_jogo}")


def checar_turnos(rows):
    print("\n=== Regra de turnos (X sempre comeca) ===")
    violacoes = 0
    for r in rows:
        board = [r[c] for c in BOARD_COLS]
        nx, no = board.count("x"), board.count("o")
        if nx != no and nx != no + 1:
            violacoes += 1
    print(f"Linhas que violam a regra #x == #o ou #x == #o+1: {violacoes}  (esperado: 0)")


def main():
    rows = load_rows()
    checar_integridade(rows)
    distribuicao_classes(rows)
    checar_regras_jogo(rows)
    checar_turnos(rows)


if __name__ == "__main__":
    main()

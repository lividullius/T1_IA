"""Exploração e checagens de integridade do dataset bruto tic-tac-toe.data (Passo 2)."""
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
BOARD_COLUMNS = COLUMNS[:9]

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # linhas
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # colunas
    (0, 4, 8), (2, 4, 6),             # diagonais
]


def load_rows():
    with open(RAW_PATH, newline="") as f:
        reader = csv.reader(f)
        rows = [dict(zip(COLUMNS, row)) for row in reader if row]
    return rows


def has_win(board, player):
    return any(all(board[i] == player for i in line) for line in WIN_LINES)


def main():
    rows = load_rows()
    print(f"2.3 - Total de linhas carregadas: {len(rows)}")

    # valores fora de {x,o,b}
    valores_invalidos = []
    for i, row in enumerate(rows):
        for col in BOARD_COLUMNS:
            if row[col] not in ("x", "o", "b"):
                valores_invalidos.append((i, col, row[col]))
    print(f"2.3 - Valores fora de {{x,o,b}} nas 9 colunas: {len(valores_invalidos)}")
    if valores_invalidos:
        print("   Exemplos:", valores_invalidos[:5])

    # classes fora de {positive, negative}
    classes_invalidas = [r["class"] for r in rows if r["class"] not in ("positive", "negative")]
    print(f"2.3 - Valores de classe fora de {{positive,negative}}: {len(classes_invalidas)}")

    # duplicadas
    tuplas = [tuple(r[c] for c in COLUMNS) for r in rows]
    contagem = Counter(tuplas)
    duplicadas = {k: v for k, v in contagem.items() if v > 1}
    total_linhas_duplicadas_extra = sum(v - 1 for v in duplicadas.values())
    print(f"2.3 - Combinações de linha distintas duplicadas: {len(duplicadas)}")
    print(f"2.3 - Total de linhas 'extras' por causa de duplicatas: {total_linhas_duplicadas_extra}")

    # nulos / faltantes
    faltantes = sum(1 for r in rows for c in COLUMNS if r[c] == "" or r[c] is None)
    print(f"2.3 - Valores nulos/faltantes: {faltantes}")

    # 2.4 - distribuição de classes
    dist = Counter(r["class"] for r in rows)
    total = len(rows)
    for classe in ("positive", "negative"):
        qtd = dist.get(classe, 0)
        pct = 100 * qtd / total
        print(f"2.4 - Classe '{classe}': {qtd} ({pct:.2f}%)")

    # 2.5 - validação de regras do jogo
    print("\n2.5 - Amostras 'positive' (deveria ter vitoria de X):")
    positivas = [r for r in rows if r["class"] == "positive"]
    for r in positivas[:5]:
        board = [r[c] for c in BOARD_COLUMNS]
        x_win = has_win(board, "x")
        o_win = has_win(board, "o")
        print(f"   {board} -> X venceu={x_win} O venceu={o_win}")

    print("\n2.5 - Amostras 'negative' (X nao venceu):")
    negativas = [r for r in rows if r["class"] == "negative"]
    for r in negativas[:5]:
        board = [r[c] for c in BOARD_COLUMNS]
        x_win = has_win(board, "x")
        o_win = has_win(board, "o")
        empate = "b" not in board and not x_win and not o_win
        print(f"   {board} -> X venceu={x_win} O venceu={o_win} empate(sem casas vazias e sem vencedor)={empate}")

    # checar TODAS as positive: X realmente venceu?
    positivas_sem_vitoria_x = [r for r in positivas if not has_win([r[c] for c in BOARD_COLUMNS], "x")]
    print(f"\n2.5 - Linhas 'positive' onde X NAO tem 3-em-linha: {len(positivas_sem_vitoria_x)}")

    # checar TODAS as negative: quantas tem O vencendo, quantas empate, quantas "tem jogo" (com 'b' sobrando e sem vencedor)
    neg_o_venceu = 0
    neg_empate_cheio = 0
    neg_tem_jogo = 0
    neg_x_venceu_incorretamente = 0
    neg_com_casa_vazia = 0
    for r in negativas:
        board = [r[c] for c in BOARD_COLUMNS]
        x_win = has_win(board, "x")
        o_win = has_win(board, "o")
        tem_vazio = "b" in board
        if tem_vazio:
            neg_com_casa_vazia += 1
        if x_win:
            neg_x_venceu_incorretamente += 1
        elif o_win:
            neg_o_venceu += 1
        elif tem_vazio:
            neg_tem_jogo += 1
        else:
            neg_empate_cheio += 1

    print(f"2.5 - Entre as 'negative': O venceu={neg_o_venceu}, empate com tabuleiro cheio={neg_empate_cheio}, "
          f"'tem jogo' (casas vazias, sem vencedor)={neg_tem_jogo}, X venceu por engano={neg_x_venceu_incorretamente}")
    print(f"2.5 - Linhas 'negative' com pelo menos 1 casa vazia ('b'): {neg_com_casa_vazia} de {len(negativas)}")

    # 2.6 - contagem de x/o/b por linha + validacao da regra de contagem
    violacoes_contagem = []
    for i, r in enumerate(rows):
        board = [r[c] for c in BOARD_COLUMNS]
        nx = board.count("x")
        no = board.count("o")
        nb = board.count("b")
        if not (nx == no or nx == no + 1):
            violacoes_contagem.append((i, nx, no, nb))
    print(f"\n2.6 - Linhas que violam a regra 'x == o' ou 'x == o+1': {len(violacoes_contagem)}")
    if violacoes_contagem:
        print("   Exemplos:", violacoes_contagem[:5])


if __name__ == "__main__":
    main()

"""Divide o dataset de 4 classes em treino/validação/teste (60/20/20) e
aplica as duas abordagens de pré-processamento em cada partição.

A classe "Empate" tem só 16 tabuleiros únicos (oversampled até 200 linhas
— ver entendendo-o-dataset.md), então ela é dividida por tabuleiro único
(todas as cópias de um tabuleiro ficam na mesma partição), e não por
linha, para não vazar o mesmo tabuleiro entre treino e teste.
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

from explorar_dataset import BOARD_COLS
from preprocessamento import (
    FEATURES_ABORDAGEM1, FEATURES_ABORDAGEM2,
    codificar_abordagem1, codificar_abordagem2,
    load_dataset_4classes,
)

SEED = 42
OUT_DIR = Path(__file__).parent / "processed"


def dividir_lista(itens, rng):
    """60/20/20."""
    itens = list(itens)
    rng.shuffle(itens)
    n_treino = round(len(itens) * 0.6)
    n_val = round(len(itens) * 0.2)
    return itens[:n_treino], itens[n_treino:n_treino + n_val], itens[n_treino + n_val:]


def dividir_empate(linhas, rng):
    """Agrupa por tabuleiro único antes de dividir, para não vazar
    duplicatas entre partições."""
    grupos = defaultdict(list)
    for linha in linhas:
        grupos[tuple(linha[c] for c in BOARD_COLS)].append(linha)

    grupos_treino, grupos_val, grupos_teste = dividir_lista(list(grupos.values()), rng)
    return [l for g in grupos_treino for l in g], [l for g in grupos_val for l in g], [l for g in grupos_teste for l in g]


def dividir_dataset(linhas, rng):
    treino, val, teste = [], [], []
    for classe in ["X venceu", "O venceu", "Empate", "Tem jogo"]:
        da_classe = [l for l in linhas if l["classe"] == classe]
        if classe == "Empate":
            t, v, s = dividir_empate(da_classe, rng)
        else:
            t, v, s = dividir_lista(da_classe, rng)
        treino += t
        val += v
        teste += s

    rng.shuffle(treino)
    rng.shuffle(val)
    rng.shuffle(teste)
    return treino, val, teste


def salvar_split(caminho, linhas, codificar, features):
    with open(caminho, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=features + ["classe"])
        writer.writeheader()
        for linha in linhas:
            board = [linha[c] for c in BOARD_COLS]
            feats = codificar(board)
            feats["classe"] = linha["classe"]
            writer.writerow(feats)


def main():
    rng = random.Random(SEED)
    treino, val, teste = dividir_dataset(load_dataset_4classes(), rng)

    abordagens = [
        ("ab1", codificar_abordagem1, FEATURES_ABORDAGEM1),
        ("ab2", codificar_abordagem2, FEATURES_ABORDAGEM2),
    ]
    splits = [("treino", treino), ("validacao", val), ("teste", teste)]

    for nome_ab, codificar, features in abordagens:
        for nome_split, linhas in splits:
            salvar_split(OUT_DIR / f"{nome_ab}_{nome_split}.csv", linhas, codificar, features)

    print(f"treino: {len(treino)}  validacao: {len(val)}  teste: {len(teste)}")


if __name__ == "__main__":
    main()

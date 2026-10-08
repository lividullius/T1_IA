import importlib
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "dataset"))

try:
    from explorar_dataset import has_win
    from preprocessamento import codificar_abordagem1, codificar_abordagem2
except ModuleNotFoundError:
    print("Não foi possível carregar os módulos do projeto.")
    print("Rode a partir da pasta do projeto:")
    print("  python frontend/jogo.py")
    raise SystemExit(1)

ALGORITMOS = (
    ("1", "KNN", "modelos.knn.knn", "modelos/knn/knn.py"),
    ("2", "MLP", "modelos.mlp.mlp", "modelos/mlp/mlp.py"),
    ("3", "Árvore de decisão", "modelos.arvore.arvoresDecisao", "modelos/arvore/arvoresDecisao.py"),
    ("4", "SVM", "modelos.livre1.svm", "modelos/livre1/svm.py"),
    ("5", "Random Forest", "modelos.livre2.randomForest", "modelos/livre2/randomForest.py"),
)

ABORDAGENS = (
    ("1", "ab1", "abordagem 1 - 9 casas do tabuleiro", codificar_abordagem1),
    ("2", "ab2", "abordagem 2 - 7 atributos derivados", codificar_abordagem2),
)


def configurar_terminal():
    for fluxo in (sys.stdout, sys.stdin):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8", errors="replace")


def escolher_algoritmo():
    print()
    print("  Escolha a IA")
    for chave, nome, *_resto in ALGORITMOS:
        print(f"  {chave}. {nome}")
    print()

    while True:
        texto = input("  Algoritmo (1-5): ").strip()
        for item in ALGORITMOS:
            if texto == item[0]:
                return carregar_algoritmo(item)
        print("  Escolha um número de 1 a 5.")


def escolher_abordagem():
    print()
    print("  Escolha a abordagem")
    for chave, _codigo, rotulo, _codificar in ABORDAGENS:
        print(f"  {chave}. {rotulo}")
    print()

    while True:
        texto = input("  Abordagem (1-2): ").strip()
        for item in ABORDAGENS:
            if texto == item[0]:
                return item
        print("  Escolha 1 ou 2.")


def carregar_algoritmo(item):
    _chave, nome, modulo, script = item
    try:
        mod = importlib.import_module(modulo)
    except ModuleNotFoundError:
        print("Faltam pandas e scikit-learn neste Python.")
        print("Na pasta do projeto:")
        print("  pip install -r requirements.txt")
        print("  python frontend/jogo.py")
        raise SystemExit(1)

    _chave_ab, codigo, rotulo, codificar = escolher_abordagem()
    if not mod.caminho_modelo(codigo).exists():
        print(f"Modelo de {nome}, {rotulo}, não encontrado.")
        print("O treino gera as duas abordagens e não vai no repositório.")
        print("Na pasta do projeto:")
        print(f"  python {script}")
        print("  python frontend/jogo.py")
        raise SystemExit(1)

    def classificar(board):
        return mod.inferir(codificar(board), codigo)

    return f"{nome}, {rotulo}", classificar


def estado_real(board):
    if has_win(board, "x"):
        return "X venceu"
    if has_win(board, "o"):
        return "O venceu"
    if "b" not in board:
        return "Empate"
    return "Tem jogo"


def deve_encerrar(real, previsto):
    if real != "Tem jogo" and previsto == "Tem jogo":
        return True
    if real == "Tem jogo" and previsto != "Tem jogo":
        return False
    return real != "Tem jogo"


def avaliar(real, previsto):
    if real == previsto:
        return True, f"IA: {previsto}."
    if real != "Tem jogo" and previsto == "Tem jogo":
        return False, (
            f"IA: {previsto}. Erro (real: {real}). "
            "Fim não detectado - partida encerrada."
        )
    if real == "Tem jogo":
        return False, (
            f"IA: {previsto}. Erro (real: Tem jogo). "
            "Fim incorreto - jogo continua."
        )
    return False, f"IA: {previsto}. Erro (real: {real})."


def placar(acertos, erros):
    total = acertos + erros
    if total == 0:
        return "Placar da IA: nenhuma jogada ainda."
    texto_acerto = "acerto" if acertos == 1 else "acertos"
    texto_erro = "erro" if erros == 1 else "erros"
    return (
        f"Placar da IA: {acertos} {texto_acerto}, {erros} {texto_erro} "
        f"({100 * acertos / total:.0f}%)."
    )


def marca(board, indice):
    valor = board[indice]
    if valor == "x":
        return "X"
    if valor == "o":
        return "O"
    return str(indice + 1)


def desenhar(board, acertos, erros, feedback, nome_ia):
    casas = [marca(board, i) for i in range(9)]

    def fileira(a, b, c):
        return f" {a} | {b} | {c} "

    print()
    print("  Jogo da velha")
    print("  Você é X     Máquina é O (aleatório)")
    print(f"  IA: {nome_ia}")
    print()
    for inicio in (0, 3, 6):
        print("  " + fileira(casas[inicio], casas[inicio + 1], casas[inicio + 2]))
        if inicio < 6:
            print("  ---+---+---")
    print()
    for linha in feedback:
        print(f"  {linha}")
    print(f"  {placar(acertos, erros)}")
    print()


def pedir_casa(board):
    while True:
        texto = input("  Casa (1-9) ou q para sair: ").strip().lower()
        if texto == "q":
            return None
        if texto not in list("123456789"):
            print("  Escolha um número de 1 a 9.")
            continue
        indice = int(texto) - 1
        if board[indice] != "b":
            print("  Essa casa já está ocupada.")
            continue
        return indice


def partida(acertos, erros, nome_ia, inferir):
    board = ["b"] * 9
    vez = "x"
    desenhar(board, acertos, erros, ["Você começa. Escolha uma casa."], nome_ia)

    while True:
        if vez == "x":
            casa = pedir_casa(board)
            if casa is None:
                return acertos, erros, False
            jogada = f"Você jogou na casa {casa + 1}."
        else:
            casa = random.choice([i for i, valor in enumerate(board) if valor == "b"])
            jogada = f"Máquina jogou na casa {casa + 1}."

        board[casa] = vez
        previsto = inferir(board)
        real = estado_real(board)
        acerto, texto_ia = avaliar(real, previsto)
        acertos += acerto
        erros += not acerto
        situacao = "acerto" if acerto else "erro"
        desenhar(board, acertos, erros, [jogada, f"{texto_ia}  ({situacao})"], nome_ia)

        if deve_encerrar(real, previsto):
            input("  Enter para seguir... ")
            return acertos, erros, True

        if vez == "x":
            input("  Enter para a máquina jogar... ")

        vez = "o" if vez == "x" else "x"


def main():
    configurar_terminal()
    nome_ia, inferir = escolher_algoritmo()
    acertos = erros = 0

    while True:
        acertos, erros, continuar = partida(acertos, erros, nome_ia, inferir)
        if not continuar:
            break
        resposta = input("  Outra partida? (s/n): ").strip().lower()
        if resposta != "s":
            break

    print()
    print(f"  {placar(acertos, erros)}")
    print()


if __name__ == "__main__":
    main()

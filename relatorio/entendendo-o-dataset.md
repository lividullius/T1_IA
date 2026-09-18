# Entendendo o dataset

Dataset: [Tic-Tac-Toe Endgame](https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame) (UCI Machine Learning Repository).
Arquivos: `dataset/raw/tic-tac-toe.data` (dados) e `dataset/raw/tic-tac-toe.names` (descrição das colunas).
Script usado para gerar as checagens abaixo: `dataset/explorar_dataset.py`.

## 2.2 — Formato bruto

- CSV sem cabeçalho, 958 linhas, 10 colunas.
- As 9 primeiras colunas são as casas do tabuleiro, na ordem: `top-left`, `top-middle`, `top-right`, `middle-left`, `middle-middle`, `middle-right`, `bottom-left`, `bottom-middle`, `bottom-right`. Valores possíveis: `x`, `o`, `b` (vazio).
- Última coluna é `class`: `positive` (X venceu) ou `negative` (X não venceu).
- Como o arquivo não vem com header, os nomes de coluna acima devem ser adicionados manualmente ao carregar.

## 2.3 — Integridade básica

| Checagem | Resultado |
|---|---|
| Total de linhas | **958** ✅ (confirma carregamento correto) |
| Valores fora de `{x,o,b}` nas 9 colunas | **0** |
| Valores de classe fora de `{positive,negative}` | **0** |
| Linhas duplicadas | **0** |
| Valores nulos/faltantes | **0** |

## 2.4 — Distribuição original das classes

| Classe | Quantidade | Percentual |
|---|---|---|
| `positive` | 626 | 65,34% |
| `negative` | 332 | 34,66% |

Confere com a expectativa do enunciado (~65% / ~35%).

## 2.5 — Consistência das regras do jogo

- Em todas as 626 linhas `positive`, X realmente tem 3-em-linha (0 linhas `positive` sem vitória de X confirmada).
- Entre as 332 linhas `negative`:
  - **316** têm vitória de O.
  - **16** são empate com tabuleiro completo (sem vencedor, sem casas vazias).
  - **0** ficaram com casas vazias e sem vencedor (ou seja, não existem posições "em andamento" no dataset).
- Achado relevante: **nem toda linha `negative` tem as 9 casas preenchidas**. 316 de 332 (95,2%) têm pelo menos uma casa `b` vazia — o jogo termina assim que O completa a vitória, então o restante do tabuleiro fica em branco. Apenas as 16 linhas de empate têm o tabuleiro totalmente preenchido.

## 2.6 — Contagem de X, O e casas vazias

- Para cada linha foi contado o número de `x`, `o` e `b`.
- Regra do jogo (X sempre começa): `#x == #o` ou `#x == #o + 1`.
- **0 linhas** violam essa regra — dataset consistente com a mecânica do jogo.
- Essa contagem por linha será reaproveitada como feature (quantidade de X, quantidade de O, casas vazias) na Abordagem 2 do pré-processamento (Passo 3).

## 2.7 — Conclusões e gancho para o Passo 3

- Dataset com **958 instâncias**, sem duplicatas, sem valores inválidos e sem valores faltantes.
- Distribuição original: **626 positive (65,34%)** vs **332 negative (34,66%)** — desbalanceado.
- Nenhuma inconsistência de regra encontrada (todas as `positive` têm vitória de X confirmada; contagem de X/O sempre respeita o turno alternado).
- O dataset bruto só tem **2 classes** (`positive`/`negative`), mas o trabalho pede **4** (X venceu / O venceu / Empate / Tem jogo). Com os números acima já dá pra prever a divisão:
  - `positive` (626) → vira **"X venceu"**.
  - `negative` (332) → se divide em **"O venceu" (316)** e **"Empate" (16)**.
  - **"Tem jogo" não existe no dataset original** (0 linhas com casas vazias e sem vencedor), pois o UCI só contém *endgames* (posições finais legais). Essa 4ª classe provavelmente precisará ser sintetizada/derivada de outra forma no Passo 3 — isso deve ser documentado como limitação do dataset original.

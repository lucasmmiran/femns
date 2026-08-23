# Validação do Poiseuille contra a solução analítica

Caso de validação **exata**: com o perfil desenvolvido imposto na entrada
(`inlet: {vx: {perfil: parabolico, vmax: 1.5}}`), a solução do canal reto é

    vx(y) = 6·y·(1−y),    vy = 0,    dp/dx = −12/Re

em **toda** a extensão do canal — não há região de desenvolvimento, então
qualquer desvio é erro de discretização (e, no caso ALE, do movimento de
malha), e não efeito de entrada.

Malha `meshes/poiseuille.msh`: domínio [0,2]×[0,1], 2785 nós de vértice,
h ≈ 0,03. MINI, Re=1, dt=0,001, 1000 iterações, advecção
`semi_lagrangian` com `sl_boundary: intercept`.

Reproduzir:

```bash
python scripts/run_simulation.py --config configs/poiseuille_perfil.yaml   # malha estática
python scripts/run_simulation.py --config configs/poiseuille_ale.yaml      # malha oscilando
python scripts/plot_poiseuille_validacao.py \
    --config configs/poiseuille_perfil.yaml --config-b configs/poiseuille_ale.yaml \
    --label "malha estatica" --label-b "ALE (nos oscilando)"
```

## Bug encontrado ao validar (e por que quase passou)

A primeira rodada deu **6,3% de erro** no pico e vazão 0,9364 em vez de
1,0. Não era discretização: a âncora da parábola vinha dos nós que
*restaram* com `ccName == 'inlet'`, e as quinas do contorno pertencem a
`top`/`bottom` por prioridade (no-slip, correto). A parábola zerava em
y=0,031 e y=0,969 — dentro do canal — em vez de nas paredes.

O que tornou o erro traiçoeiro: **a vazão era conservada seção a seção**
(0,93645 / 0,93650 / 0,93641 / 0,93638 em x = 0,02 / 0,5 / 1,0 / 1,5) e o
campo convergia liso. Parecia difusão numérica do MINI. Só fechou quando
a conta `(2/3)·vmax·span_errado = (2/3)·1,5·0,9375 = 0,9375` reproduziu o
valor medido.

Corrigido em `boundary.valores_da_condicao` (parâmetro `idx_span`, com a
extensão vindo de `IENboundElem`). Depois da correção o erro caiu **100×**.

## Resultado — malha estática

Seção em x = 1,0 (57 nós):

| métrica | valor |
|---|---|
| erro relativo máximo | **0,063%** de vmax |
| erro L2 | 0,037% de vmax |
| pico de vx | 1,49904 (exato: 1,5) |

Vazão por seção (exato: 1,0):

| x | 0,02 | 0,50 | 1,00 | 1,98 |
|---|---|---|---|---|
| vazão | 0,99938 | 0,99903 | 0,99898 | 0,99869 |

O perfil de erro tem a assinatura esperada da discretização P1: **zero nas
paredes** (onde a condição de contorno é imposta), crescendo em módulo até
o centro, onde a curvatura da parábola é maior, e sempre **negativo** — o
elemento linear subestima o pico. Imagens em
`solucoes/poiseuille_estatica/validacao/`.

## Resultado — malha oscilando (ALE)

Mesmo caso com `mesh_motion` ligado: todos os 2593 nós interiores (de
2785) oscilando com amplitude 0,3·h local, `omega = 125,66` (50 passos por
período com dt=0,001 — ver a armadilha de amostragem no README).

| métrica | malha estática | ALE (nós oscilando) |
|---|---|---|
| erro relativo máximo | 0,063% | **0,067%** |
| erro L2 | 0,037% | **0,032%** |
| pico de vx | 1,49904 | **1,49945** |
| tempo (1000 iterações) | ~63 s | 374 s |

**O ALE preserva a solução exata.** O erro é indistinguível do da malha
estática — o L2 é até ligeiramente menor, o que é ruído de amostragem
(os nós ALE caem em posições `y` diferentes, então os dois conjuntos não
são comparáveis ponto a ponto). O gráfico de perfil mostra os pontos ALE
deslocados em `y` em relação aos estáticos — confirmação visual de que a
malha de fato se moveu — e ainda assim sobre a curva analítica.

No gráfico de erro, a curva ALE é **serrilhada** enquanto a estática é
lisa: cada nó oscilante parou num deslocamento diferente no último passo,
então o erro de interpolação de cada um é independente do vizinho. A
envoltória, porém, é a mesma da malha estática (~0,06%), e o erro
continua ancorado em zero nas paredes (que não se movem).

Este é o teste que valida a correção ALE de ponta a ponta: se a
velocidade da malha `w` fosse ignorada no termo convectivo, ou se a
interpolação semi-Lagrangeana usasse a malha de chegada em vez da malha
onde o campo vive, o erro apareceria aqui — foi medido em até 15% da
escala de velocidade por passo (ver `CLAUDE.local.md`, 2026-08-16).

## Leitura

O Poiseuille com perfil imposto é o único caso do projeto com solução
analítica, e por isso o único que valida **magnitude absoluta** de erro —
os benchmarks anteriores (MINI vs Tri6, `dirichlet` vs `intercept`) só
comparavam variantes entre si, o que não detecta um erro comum às duas.
Foi exatamente o que aconteceu: o bug de 6,4% acima passaria despercebido
em qualquer comparação relativa.

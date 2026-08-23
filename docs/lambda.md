# O clamp de coordenadas baricêntricas em `interpolar_mini`/`interpolar_tri6`

Este documento existe por causa da decisão B de
`docs/semi_lagrangian_tri6.md` §5.B: o Lucas decidiu manter, no
`interpolar_tri6` novo, o mesmo clamp de λ que `interpolar_mini` já usava,
sem re-derivar a política para a base quadrática. Este documento faz essa
re-derivação — a resposta acabou sendo mais tranquilizadora do que a
suposição inicial (ver §3).

## 1. De onde vem o λ negativo

`localizar_pontos_partida` (`semi_lagrangian.py`) aceita como "dentro do
elemento" qualquer ponto cujo menor λ seja `>= -tol` (`tol=1e-9` por
padrão):

```python
dentro = lambdas.min(axis=1) >= -tol
```

Isso é proposital: a caminhada por vizinhança calcula λ por aritmética de
ponto flutuante (`baricentro`, razão de duas áreas), então um ponto
*exatamente* sobre uma aresta pode dar λ = -1e-16 em vez de 0.0 exato,
por erro de arredondamento puro — sem essa tolerância, pontos legítimos
bem perto de uma aresta oscilariam entre "dentro" e "andar mais um passo"
por ruído de última casa decimal.

O efeito colateral: quando `calculo_sl` chama `interpolar_mini` ou
`interpolar_tri6` nos nós localizados, um dos três λ pode chegar
levemente negativo — não um erro grosseiro de geometria, só o resíduo de
tolerância descrito acima, tipicamente da ordem de `tol` (1e-9) até a
precisão de máquina (~1e-16).

## 2. A álgebra do clamp

Implementado identicamente nas duas funções:

```python
li, lj, lk = np.clip(li, 0.0, 1.0), np.clip(lj, 0.0, 1.0), np.clip(lk, 0.0, 1.0)
soma = li + lj + lk
li, lj, lk = li / soma, lj / soma, lk / soma
```

Dois passos distintos, cada um com um papel diferente:

1. **`np.clip` para `[0,1]`**: zera qualquer λ negativo (e, simetricamente,
   recorta qualquer λ que passe de 1 — pode acontecer com o *outro* λ
   quando o terceiro é negativo, já que a soma dos três continua ~1).
2. **Renormalização pela soma**: depois do clip, a soma dos três já não é
   exatamente 1 (perdeu a parte negativa) — dividir pela soma restaura
   `li+lj+lk=1`.

O passo 2 não é cosmético — é o que **garante partição de unidade
exatamente**, para qualquer combinação de λ que some 1, clampada ou não.
Vale para as duas bases, por identidade algébrica pura:

- **MINI**: `Ni+Nj+Nk+Nb = (li+lj+lk) - 27·li·lj·lk + 27·li·lj·lk = li+lj+lk = 1`.
- **Tri6**: usando `Σλ²ᵢ = (Σλᵢ)² - 2Σλᵢλⱼ = 1 - 2Σλᵢλⱼ` (válido sempre que `Σλᵢ=1`):
  `ΣNᵢ + ΣNᵢⱼ = Σλᵢ(2λᵢ-1) + Σ4λᵢλⱼ = 2Σλᵢ² - Σλᵢ + 4Σλᵢλⱼ`
  `= 2(1-2Σλᵢλⱼ) - 1 + 4Σλᵢλⱼ = 1`.

Ou seja: **qualquer** λ que some 1 (negativo, maior que 1, o que for)
reproduz um campo constante exatamente nessa base — a renormalização é o
que impõe `Σλ=1` depois do clip, e a partição de unidade vem de graça.
Sem o passo 2, o clip sozinho quebraria isso (a soma ficaria < 1 sempre
que algum λ tivesse sido negativo, e um campo constante deixaria de ser
reproduzido exatamente).

## 3. MINI vs. Tri6: o clamp é "mais agressivo" no quadrático?

Essa foi a suspeita registrada no plano original (`semi_lagrangian_tri6.md`,
decisão B): a base do Tri6 é quadrática em λ, então um λ negativo "entraria
ao quadrado" e o erro do clamp seria maior. **Verificado abaixo: não é bem
assim** — a suposição era razoável a princípio, mas a derivação não
confirma.

O que importa não é o grau do polinômio em si, mas o **comportamento local
perto de λ=0** (onde o clamp age). Nas duas bases, a função de forma do
vértice associado ao λ que vai a zero tem derivada **não-nula** ali:

- MINI: `Ni = λᵢ - 9λᵢλⱼλₖ = λᵢ·(1 - 9λⱼλₖ)` → perto de `λᵢ=0`,
  `dNᵢ/dλᵢ ≈ 1 - 9λⱼλₖ` (um número O(1), não zero).
- Tri6: `Nᵢ = λᵢ(2λᵢ-1)` → `dNᵢ/dλᵢ = 4λᵢ-1`, que em `λᵢ=0` vale **-1**
  exatamente (não zero).

Como as duas derivadas são O(1) e não-nulas em λ=0, uma perturbação
`δλ ~ tol` produz uma perturbação na função de forma **da mesma ordem**,
`δN ~ tol`, nas duas bases — o termo "quadrático" (`2λᵢ²` no Tri6, ou o
produto triplo no MINI) só contribui em **segunda ordem** (`O(tol²)`,
desprezível frente a `O(tol)`).

Verificação numérica (perturbação `λₖ = -1e-9`, `λᵢ,λⱼ` absorvendo a
diferença para manter `Σλ=1`, 6 pontos aleatórios ao longo da aresta
oposta a `k`):

| ponto na aresta (`λᵢ₀`) | max\|diferença\| MINI | max\|diferença\| Tri6 | razão Tri6/MINI |
|---|---|---|---|
| 0.623 | 6.34e-09 | 2.49e-09 | 0.39 |
| 0.293 | 5.59e-09 | 2.83e-09 | 0.51 |
| 0.087 | 2.14e-09 | 3.65e-09 | 1.71 |
| 0.065 | 1.64e-09 | 3.74e-09 | 2.28 |
| 0.782 | 4.60e-09 | 3.13e-09 | 0.68 |
| 0.871 | 3.02e-09 | 3.49e-09 | 1.15 |

Ambas ficam na casa de `1e-9` (a ordem de `tol`), com a razão oscilando
em torno de 1 (às vezes o Tri6 é maior, às vezes o MINI) — **não há
amplificação sistemática do Tri6 sobre o MINI**. A suspeita inicial não
se confirma: o clamp idêntico está numericamente justificado para as
duas bases, não só por conveniência de manter o código igual.

## 4. Vantagens de manter o clamp idêntico

- **Verificado acima**: a ordem do erro introduzido é a mesma (`O(tol)`,
  hoje `~1e-9`) nas duas bases — não há motivo numérico para tratamento
  diferente.
- **Partição de unidade exata** (§2) é garantida por identidade
  algébrica para qualquer λ que some 1, nas duas bases — a
  renormalização faz o trabalho, independente do valor exato do clamp.
- Menos código, um parâmetro a menos, comportamento previsível e igual
  nos dois elementos — não há uma segunda política pra manter em sincronia
  se `tol` mudar no futuro.

## 5. Desvantagens / pontos em aberto

- A justificativa original de `interpolar_mini` foi escrita pensando só
  na base do MINI; **antes desta análise não estava provado** que o
  mesmo clamp seria numericamente equivalente para o Tri6 — é exatamente
  o tipo de suposição "parece óbvio, não foi verificado" que já causou
  dois bugs neste módulo (fórmula da bolha em 12/07, sinal de pressão do
  Tri6 em 14/08). Aqui a suposição sobreviveu à verificação, mas só
  porque foi de fato verificada — não porque fosse óbvia a priori.
- `tol=1e-9` é um valor fixo, calibrado para a malha/geometria típica
  deste projeto. Um elemento muito fino ou obtuso (ângulo interno
  pequeno) faz a premissa "λ negativo é da ordem de `tol`" parar de
  valer — mas essa fragilidade já existia no MINI e não é introduzida
  nem agravada pelo Tri6; só continua não resolvida.
- Manter o clamp idêntico fecha, por ora, a porta pra uma política
  **específica** do Tri6 (ex.: clampar diretamente os pesos `Nᵢ`/`Nᵢⱼ`
  em vez dos λ, ou usar uma tolerância diferente por elemento) — decisão
  explicitamente adiada, não descartada.

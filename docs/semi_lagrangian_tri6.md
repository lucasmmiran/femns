# Semi-Lagrangeano para o elemento Tri6: plano de implementação

Status: **implementado e validado** (ver §9). As decisões dos itens 5.A–C
foram tomadas pelo Lucas: A) restrição quadrática na aresta de saída;
B) clamp de λ idêntico ao MINI, herdado por analogia (ver `docs/lambda.md`);
C) docstring de `_fallback_dirichlet` corrigida para não afirmar uma
generalidade que só vale para o MINI. O restante deste documento é o plano
original, mantido como registro das decisões e do raciocínio por trás delas.

## 1. Por que hoje é bloqueado

`scripts/run_simulation.py` recusa a combinação explicitamente:

```python
if element == "tri6" and advection == "semi_lagrangian":
    raise ValueError(
        "advection=semi_lagrangian ainda nao suporta element=tri6 -- "
        "femns.semi_lagrangian.interpolate_mini e especifico da interpolacao P1+bolha do MINI."
    )
```

O motivo é de consistência numérica, não de esforço de implementação: a
interpolação no pé da característica precisa usar as mesmas funções de
forma da discretização. `interpolar_mini` tem a base do MINI cravada — 4
DOFs, `Ni = λi − 9λiλjλk` nos vértices e `Nb = 27λiλjλk` no centroide. O
Tri6 tem 6 DOFs com base quadrática (P2) e nenhum nó de centroide. Se o
guard não existisse, a função leria a 4ª coluna de `IEN` como bolha —
no Tri6 essa coluna é o nó de aresta v4 (aresta v1–v2, ver
`mesh.elem_tri6`). Pesos errados aplicados ao nó errado.

O risco concreto é que esse erro **não quebra visivelmente**: o campo
resultante sai liso e plausível, não diverge. É a mesma classe do bug
de sinal de pressão do elemento Tri6 encontrado em 2026-08-14, e do bug
da fórmula da bolha do MINI encontrado em 2026-07-12 — os dois só
apareceram por comparação contra um caso de física conhecida, não por
crash ou teste que falhasse sozinho.

## 2. O que já funciona sem mudança nenhuma

A busca por elemento é agnóstica ao elemento de velocidade:
`backtrace`, `baricentro` e `localizar_pontos_partida` só tocam
`IEN[:, 0:3]` (os 3 vértices) e valores nodais — e todo DOF do Tri6 é
nodal, igual ao MINI. Essas três funções não precisam de nenhuma
alteração.

O que é específico do MINI está concentrado em três pontos:
`interpolar_mini`, o desempacotamento `i, j, k, b = IEN[...].T` dentro
de `calculo_sl`, e a interpolação na aresta de saída em
`_fallback_interceptacao`.

## 3. A armadilha estrutural: `ne` tem dois significados

Em `calculo_sl`, o parâmetro `ne` é usado com dois sentidos diferentes
na mesma função:

```python
n_total = npoints + ne                                     # ne = numero de nos extras
elem_start = np.concatenate([node_to_elem, np.arange(ne)])  # ne = numero de elementos
```

No MINI os dois números coincidem — cada elemento contribui exatamente
um nó de centroide, então "número de elementos" e "número de nós
extras" são a mesma coisa. **No Tri6 não coincidem**: os nós extras são
de aresta, deduplicados entre elementos vizinhos (`mesh.elem_tri6`),
então o número de nós extras é da ordem de 1,5× o número de elementos,
não igual.

A segunda linha também embute a suposição "o nó extra `npoints + e`
pertence ao elemento `e`", que só é verdade para o centroide do MINI.

Se essa distinção não for feita explicitamente, o Tri6 monta
`elem_start` com o tamanho errado, com chutes de caminhada sem
correspondência real com os nós. O perigo, de novo, é que a busca por
caminhada tende a *corrigir* um chute ruim e convergir mesmo assim —
resultado plausível, mais caro, sem erro nenhum reportado.

**Decisão de projeto:** separar os dois sentidos em parâmetros
distintos (`n_extra`, número de nós extras, e `ne`, número de
elementos) em vez de inferir um do outro.

## 4. Mudanças por arquivo

### `src/femns/semi_lagrangian.py`

1. Nova função `interpolar_tri6(li, lj, lk, v_vert, v_ar)` — base P2:
   `Ni = λi(2λi − 1)` nos vértices, `N_ij = 4·λi·λj` nos nós de aresta
   opostos. A ordem das arestas segue a convenção já estabelecida em
   `mesh.elem_tri6` (v4 = aresta v1–v2, v5 = aresta v2–v3, v6 = aresta
   v3–v1): face local `f` corresponde à coluna `3 + f` de `IEN`,
   mapeamento direto, sem tabela auxiliar.
2. `calculo_sl` ganha um parâmetro `elemento: str = "mini"`, além de
   `n_extra` (novo) e `ne` (ressignificado para "número de elementos",
   consistente com o resto do código). Despacha para
   `interpolar_mini` ou `interpolar_tri6` conforme `elemento`, fatiando
   `IEN[:, :3]` (vértices) e `IEN[:, 3:6]` (arestas, só no Tri6).
3. `_fallback_interceptacao` passa a considerar o nó médio da aresta de
   saída quando `elemento == "tri6"` (ver §5.A).

### `scripts/run_simulation.py`

4. Remover o `raise ValueError` da checagem `tri6` + `semi_lagrangian`.
5. Construir o chute inicial de caminhada também para nós de aresta:
   varrer `IEN[:, 3:6]` e registrar um elemento adjacente por nó
   (qualquer um dos elementos que compartilham aquela aresta serve —
   é só semente, a caminhada corrige a partir daí).
6. Passar `elemento=element` e os dois parâmetros `n_extra`/`ne`
   corretos na chamada de `calculo_sl` (hoje na linha ~299).

## 5. Decisões numéricas em aberto

### A. Interpolação na aresta de saída (`_fallback_interceptacao`)

Hoje a função interpola linearmente entre os 2 nós de vértice da
aresta, e o docstring justifica isso como **exato**, não aproximação:
sobre qualquer aresta do triângulo uma coordenada baricêntrica é zero,
a correção de bolha `9λiλjλk` do MINI se anula ali, e o campo MINI
restrito à aresta é de fato P1.

Esse argumento **não se transporta para o Tri6**: o campo P2 restrito a
uma aresta é genuinamente quadrático (depende do nó de aresta, que tem
peso `4λiλj` não-nulo justamente ali). Manter interpolação linear
degradaria o tratamento de contorno de O(h²) para O(h) — precisão pior
bem onde a física de contorno é imposta.

**Proposta:** usar a restrição quadrática de fato. Com `s` a posição
relativa do cruzamento na aresta (mesmo `s = w2` que
`interceptar_contorno` já calcula), os três pesos ficam:

```
peso(v1)   = (1 - s)(1 - 2s)
peso(v2)   = s(2s - 1)
peso(aresta) = 4s(1 - s)
```

que reduz ao caso linear quando o nó de aresta está exatamente na média
dos dois vértices (não é o caso geral, mas é o caso de malha não
distorcida).

### B. Clamp de coordenada baricêntrica

`interpolar_mini` recorta `λ` para `[0, 1]` e renormaliza, para
absorver pequenos negativos residuais da tolerância numérica da
caminhada. Com base quadrática, um `λ` levemente negativo entra ao
quadrado (`λ²` aparece implicitamente em `λ(2λ−1)`), então o mesmo
clamp tem efeito relativamente maior.

**Proposta:** manter o clamp idêntico ao do MINI por enquanto (ele só
atua na faixa da tolerância numérica, não deveria dominar o resultado),
mas documentar explicitamente que a escolha foi herdada por analogia,
não re-derivada para a base P2 — para não repetir o padrão de "parece
óbvio, não foi verificado" que já causou os dois bugs anteriores.

### C. `_fallback_dirichlet` e nós de aresta com nome de contorno

O docstring atual afirma que "nós de centroide nunca têm nome de
contorno" — verdade para o MINI, onde o centroide é sempre interior ao
elemento. **No Tri6 isso deixa de valer**: nós de aresta que caem sobre
o contorno recebem nome via `estende_IENbound_tri6`.

O impacto prático é baixo, porque `run_simulation.py` já passa
`bc_dirichlet` pré-calculado (evitando o caminho de
`_fallback_dirichlet` em produção), mas os testes exercitam esse
fallback diretamente. **Proposta:** generalizar `_fallback_dirichlet`
para aceitar nós de aresta nomeados e corrigir o docstring — caso
contrário a documentação passa a descrever um invariante que não é mais
verdadeiro.

## 6. Validação planejada

O teste decisivo, seguindo a lição de 2026-07-12 (um teste "passa" pode
mascarar fórmula errada se os valores coincidirem nos dois casos):

- `interpolar_tri6` deve reproduzir **exatamente** (precisão de
  máquina) um polinômio quadrático arbitrário avaliado em pontos
  interiores aleatórios do elemento. Uma base P2 errada não passa nesse
  teste; um polinômio aleatório não deixa espaço para coincidência
  como a de bolha=0.
- O teste da aresta (§5.A) precisa de um valor no nó médio **bem
  diferente** da média dos extremos — se for a média, quadrática e
  linear coincidem e o teste não discrimina nada.
- `calculo_sl` com elemento Tri6: campo constante → constante, campo
  linear → exato, campo **quadrático → exato** (o MINI não consegue
  reproduzir quadrático exatamente; o Tri6 deve conseguir — esse é o
  teste que separa de fato os dois caminhos).
- Por fim, Poiseuille com `element: tri6`, `eulerian` vs
  `semi_lagrangian`, `dt` pequeno — mesmo protocolo usado para validar
  o MINI e o BiCGSTAB. Antes de comparar os campos, contar quantos nós
  de fato saem do domínio nessa configuração (ver
  `docs/relatorio_sl_boundary.md`, §1: um `dt` pequeno pode não
  exercitar o fallback de contorno e dar "idêntico" sem validar nada).

## 7. Ordem de implementação sugerida

1. Separar `n_extra`/`ne` em `calculo_sl` sem mudar o comportamento do
   MINI (suíte de testes continua verde).
2. `interpolar_tri6` + teste de exatidão em polinômio quadrático
   aleatório.
3. Despacho por `elemento` em `calculo_sl`.
4. Restrição quadrática na aresta de saída (§5.A).
5. Ligar em `run_simulation.py` (remover o guard, construir o chute de
   caminhada para nós de aresta).
6. Validação no Poiseuille com `element: tri6`.

## 8. Referências

Mesma base teórica de `docs/semi_lagrangian_strategy.pdf`: Pironneau
(1982), Donea & Huerta (2003), Devillers, Pion & Teillaud (2001),
Staniforth & Côté (1991). A convenção de funções de forma P2 usada aqui
é a mesma de `assembly.assemble_tri6` (Zienkiewicz vol. 1, cap. 8).

## 9. Validação (resultados)

### 9.1 Unitário: exatidão em campo quadrático

O ponto crítico da implementação — a restrição quadrática na aresta de
saída (§5.A) só é exercitada de fato quando um nó **interior** sai do
domínio, não quando só nós de contorno saem (mesma armadilha documentada
em `docs/relatorio_sl_boundary.md`, §1, para o degrau). Por isso a
validação principal desta base é **unitária, com valores construídos à
mão**, não o Poiseuille real (ver §9.2 sobre por que o Poiseuille não
teria discriminado nada aqui):

- `interpolar_tri6` reproduz exatamente um polinômio quadrático com
  coeficientes aleatórios (`test_interpolar_tri6_reproduz_quadratico_arbitrario`).
- `calculo_sl(elemento="tri6")` reproduz esse mesmo polinômio no pé da
  característica quando o ponto continua dentro do domínio
  (`test_calculo_sl_tri6_reproduz_campo_quadratico_interior`).
- `calculo_sl(elemento="tri6", fora_dominio="intercept")` reproduz um
  campo quadrático **construído para não colapsar em linear ao longo da
  aresta de saída** (`h(x) = 2 - 3x + 5x²`) exatamente no ponto de
  cruzamento (`test_calculo_sl_tri6_intercept_reproduz_campo_quadratico_na_saida`)
  — esse é o teste que de fato prova que a restrição quadrática está em
  uso, e não uma restrição linear que por acaso bateria em casos mais
  simples.

### 9.2 Integração: Poiseuille, `eulerian` vs `semi_lagrangian`, `element: tri6`

Malha `meshes/poiseuille.msh` (2785 nós de vértice, 5376 elementos,
10945 nós Tri6), Re=1, `dt=0.001`, 300 iterações, `sl_boundary:
intercept`. Configs: `configs/poiseuille_tri6.yaml` (eulerian) e
`configs/poiseuille_tri6_sl.yaml` (semi_lagrangian).

**Contagem de saída antes de comparar** (lição repetida de
`docs/relatorio_sl_boundary.md`): com o campo já desenvolvido (100
passos), `dt=0.001` produz **63 nós que saem do domínio, todos do
`inlet`** — onde `vx=1` é Dirichlet uniforme em toda a aresta. Como no
caso do degrau em `dt=0.01`, essa configuração **não exercita** a
restrição quadrática de fato (o valor ali é constante ao longo da
aresta, então linear e quadrático dão o mesmo resultado) — é por isso
que a prova de que a base quadrática está correta vem do §9.1, não
deste run. O papel deste run é validar a integração completa
(pipeline, convergência do BiCGSTAB, física razoável), não a fórmula
da aresta.

**Campos finais** (passo 300, `t=0.3`):

| campo | faixa `eulerian` | faixa `semi_lagrangian` | max\|dif\| | L2 rel | corr |
|---|---|---|---|---|---|
| `vx` | [0, 1.4845] | [0, 1.4845] | 5.43e-05 | 8.47e-06 | 1.000000 |
| `vy` | [-0.3437, 0.3437] | [-0.3437, 0.3437] | 1.23e-04 | 3.21e-05 | 1.000000 |
| `p`  | [0, 180.31] | [0, 180.28] | 4.20e-02 | 8.76e-06 | 1.000000 |

Mesma ordem de grandeza (10⁻⁴–10⁻⁵ relativo) da validação equivalente do
MINI (12/07/2026, ~10⁻⁴–10⁻³) — consistente com o esperado, já que a
única diferença entre os dois caminhos é a advecção, e ambos resolvem o
mesmo sistema de Stokes/Navier-Stokes por trás.

Queda de pressão entrada→saída: **38.99 (eulerian) vs 38.99
(semi_lagrangian)** — e bate com o valor já validado para o Tri6
eulerian em 2026-08-14 (+39.03, CLAUDE.local.md), reforçando que a
integração do semi-Lagrangeano não introduziu nenhum viés de pressão
(o mesmo tipo de bug de sinal que mordeu o Tri6 naquela ocasião).

Sem sinal de instabilidade, sem passo não-convergido do BiCGSTAB
(residual sempre < 1e-8), tempos de parede comparáveis entre os dois
caminhos (eulerian 51.7s, semi_lagrangian 44.8s para as 300 iterações —
a diferença é ruído de convergência do BiCGSTAB entre passos, não custo
sistemático da advecção).

**Conclusão:** a integração Tri6 + semi-Lagrangeano está numericamente
consistente com o caminho eulerian já validado, e a correção específica
da restrição quadrática está provada pelos testes unitários do §9.1, já
que nenhuma configuração realista do Poiseuille a essa malha/Re a
exercitou de fato.

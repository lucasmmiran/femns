# Tratamento de "fora do domínio" no semi-Lagrangeano: `dirichlet` vs `intercept`

Comparação entre os dois tratamentos do pé da característica que cai fora
da malha, selecionáveis por `simulation.sl_boundary` no config
(`femns.semi_lagrangian.calculo_sl(fora_dominio=...)`).

- **`dirichlet`** (histórico do projeto): usa o valor de Dirichlet do
  contorno **do próprio nó de chegada**; se aquele contorno não prescreve
  aquele componente, mantém o valor atual do nó — congela a advecção ali.
- **`intercept`** (código de referência do professor,
  `referencia/clSemiLagrangian.py::computeIntercept`): calcula onde a
  característica cruzou o contorno e interpola o campo **atual** nos dois
  nós daquela aresta.

Todos os números abaixo: malha `meshes/degrau.msh` (8704 elementos, 4497
pontos, 13201 nós com centroide), Re=1, elemento MINI, BiCGSTAB, GPU.

## 1. Quando os dois divergem

Só quando nós **interiores** saem do domínio. Medido no campo real do
degrau (1 passo explícito), variando `dt`:

| `dt` | nós que saem | dos quais interiores | max\|dif\| entre os modos |
|---|---|---|---|
| 0.01 | 15 | 0 | 0 (bit-a-bit) |
| 0.05 | 73 | 58 | 1.05e-01 |
| 0.10 | 149 | 134 | 1.37e-01 |
| 0.25 | 377 | 362 | 2.40e-01 |
| 0.50 | 735 | 720 | 3.54e-01 |
| 1.00 | 1380 | 1365 | 4.34e-01 |

Em `dt=0.01` os 15 nós que saem são **todos do `inlet`**, e a
característica sai pela mesma aresta em que o nó está: o cruzamento cai
sobre o próprio nó (peso 1.0) e a aresta inteira tem `vx=1` prescrito.
Os dois tratamentos dão exatamente o mesmo valor — daí o resultado
bit-a-bit idêntico. **O caso `dt=0.01` não discrimina os métodos.**

O `outlet` do degrau nunca dispara saída: o escoamento vai da esquerda
para a direita, então a característica de um nó do outlet aponta para
**dentro** do domínio. É justamente onde `dirichlet` congelaria (o outlet
só define `p`), mas essa situação não ocorre nesta geometria.

## 2. Campos finais (caso discriminante, `dt=0.1`)

100 iterações a `dt=0.1` (mesmo tempo físico final, t=10, das 1000
iterações a `dt=0.01`):

| campo | faixa `dirichlet` | faixa `intercept` | max\|A−B\| | L2 rel | corr |
|---|---|---|---|---|---|
| `vx` | [−0.003873, 1.396] | [−0.003873, 1.396] | 1.87e-03 | 3.49e-04 | 1.00000 |
| `vy` | [−0.4842, 0.244] | [−0.4842, 0.242] | 2.78e-03 | 1.88e-03 | 1.00000 |
| `p` | [−2.714, 57.18] | [−2.714, 57.28] | 1.32e-01 | 1.11e-03 | 1.00000 |

Diferença da ordem de 10⁻³–10⁻⁴ relativo, sem mudança estrutural: mesma
recirculação atrás do degrau, mesmo pico de pressão na quina. Imagens lado
a lado em `solucoes/comparacao_sl_boundary/` (não versionadas).

Conclusão física: nesta geometria o tratamento de contorno **não** muda a
solução de forma relevante. Ele importaria numa geometria com recirculação
atravessando um contorno sem velocidade prescrita — aí `dirichlet`
congelaria os nós envolvidos e `intercept` não.

## 3. Tempo de execução

Simulação completa (`run_simulation.py`, coluna `tempo_total_s` de
`solucoes/benchmarks.xlsx`):

| caso | `dt` | iterações | tempo total | s/iteração |
|---|---|---|---|---|
| `dirichlet` | 0.01 | 1000 | 50.02 s | 0.0491 |
| `intercept` | 0.01 | 1000 | 50.56 s | 0.0497 |
| `dirichlet` | 0.1 | 100 | 14.56 s | 0.1365 |
| `intercept` | 0.1 | 100 | 14.27 s | 0.1336 |

Custo por passo semi-Lagrangeano isolado (medido com um script de
benchmark hoje removido, 30 repetições):

| variante | ms/passo |
|---|---|
| numpy vetorizado, `dirichlet` | 2.495 |
| numpy vetorizado, `intercept` | 2.522 (+1.1%) |
| torch/cuda, `intercept` (`SemiLagrangianMini`) | 4.357 |

A interceptação **não custa nada de relevante**: o trabalho extra é
proporcional ao número de nós que saem (15 de 13201 em `dt=0.01`), não ao
tamanho da malha. A diferença de tempo total entre os dois casos está
dentro do ruído de medição.

## 4. Paralelização

O código de referência paraleliza com `multiprocessing.Pool`
(`getDepartElem`). Aqui isso não compensa, e o motivo **não** é o custo de
transporte entre processos (medido: 0.68 ms, 27% do passo — caberia):

- passo semi-Lagrangeano: 2.52 ms
- iteração completa: ~49 ms (o resto é BiCGSTAB na GPU)
- ⇒ o passo é **5.1%** do tempo de iteração

Por Amdahl, mesmo um passo instantâneo economizaria no máximo 5.1% do
tempo total. O multiprocessing existe na referência porque lá o laço é
Python nó a nó; vetorizado, esse custo já saiu do caminho crítico. O
paralelismo que de fato importa aqui é o SIMD do numpy (todos os nós por
operação).

Houve também uma via torch/GPU do semi-Lagrangeano
(`femns.semi_lagrangian_tri.SemiLagrangianMini`), que implementava esta
mesma interceptação mas ficava **mais lenta** que o numpy nesta malha
(4.36 ms vs 2.52 ms): a caminhada tem controle de fluxo dependente de
dado (`torch.where`, `.numel()`), que força sincronização host↔device a
cada iteração do laço. Ela só compensaria numa malha bem maior, onde o
trabalho por iteração amortizasse essas sincronizações. Como nunca foi
ligada ao `run_simulation.py` e o numpy vetorizado a superou aqui, esse
módulo (e os scripts de benchmark que o comparavam) foi **removido** —
`femns.semi_lagrangian` é a única implementação. O parágrafo fica como
registro do porquê.

> Cuidado de medição: cronometrar a via torch **sem aquecer** dava ~95 ms/passo
> (compilação de kernel + warm-up do alocador diluídos nas repetições) —
> 20× o valor real; o benchmark aquecia todas as variantes antes de medir.

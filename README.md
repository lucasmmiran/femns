# femns

Solver de elementos finitos (método de Galerkin) para as equações de
Navier-Stokes incompressíveis, com montagem vetorizada em GPU (PyTorch).

Discretização espacial em malhas triangulares, com avanço temporal por
Euler implícito e termo advectivo linearizado na velocidade do passo
anterior. Dois elementos disponíveis (`simulation.element` no config,
ver seção "Config" abaixo): **MINI** (padrão, P1 + bolha para
velocidade, P1 para pressão) e **Tri6** (Taylor-Hood, P2 para
velocidade — vértices + nó de aresta —, P1 para pressão).

## Estrutura do projeto

```
.
├── configs/
│   └── poiseuille.yaml     # parâmetros de uma simulação (malha, dt, Re, contornos, ...)
├── meshes/
│   └── poiseuille.geo/.msh # malha gerada no Gmsh (.geo é a fonte, .msh o resultado)
├── src/femns/               # pacote com a física/numérica do solver
│   ├── mesh.py              # leitura da malha, conectividade (NToN/EToE), elementos MINI e Tri6
│   ├── assembly.py          # montagem vetorizada das matrizes de elemento (K, M, Gx, Gy, Gvx, Gvy)
│   ├── boundary.py          # condições de contorno: nomes por nó, valores, aplicação na matriz
│   ├── moving_mesh.py        # deslocamento senoidal de um nó (experimento de malha móvel)
│   ├── solver.py             # montagem do sistema global e passo de tempo (Euler implícito)
│   ├── io.py                 # escrita dos resultados em VTK
│   └── plotting.py           # imagens (PNG) da distribuição espacial das variáveis
├── scripts/
│   ├── run_simulation.py    # CLI: orquestra mesh → assembly → boundary → solver → io
│   └── plot_solution.py     # CLI: gera as imagens da última solução de uma simulação
├── tests/                    # testes unitários (conectividade, valores analíticos dos elementos MINI/Tri6, contornos)
├── solucoes/                  # saída dos .vtk (gerada em runtime, ignorada pelo git)
└── pyproject.toml
```

A ideia por trás da separação: `src/femns` contém só lógica pura de
elementos finitos (dado um `X, Y, IEN`, monta matrizes; dado uma malha,
monta vizinhanças; dado uma matriz e índices de contorno, aplica BCs).
Nenhum desses módulos sabe ler arquivo de config, escrever log ou decidir
quantas iterações rodar — isso é o papel do `scripts/run_simulation.py`.
Essa divisão é o que permite testar `assembly.py` com um triângulo de 3
nós sem precisar de uma malha `.msh` nem rodar a simulação inteira.

### Responsabilidade de cada módulo

| Módulo | O que faz | O que NÃO faz |
|---|---|---|
| `mesh.py` | lê `.msh`, monta elemento MINI (nó de centroide) ou Tri6 (nós de aresta, deduplicados entre elementos vizinhos), conectividade nó-a-nó (`NToN`) e elemento-a-elemento (`EToE`) | não sabe de condição de contorno nem de matrizes de rigidez |
| `assembly.py` | monta `K, M, Gx, Gy, Gvx, Gvy` (rigidez, massa, gradiente) por elemento, vetorizado em `torch` | não monta o sistema global nem aplica contorno |
| `boundary.py` | resolve qual nome de contorno cada nó tem (por prioridade) e monta os índices/valores de Dirichlet; zera linhas da matriz global e põe 1 na diagonal | não decide *quais* são os valores físicos — isso vem do config |
| `moving_mesh.py` | desloca um nó da malha em trajetória senoidal e reinterpola sua velocidade a partir dos vizinhos (IDW) | não atualiza a malha inteira, só o nó indicado |
| `solver.py` | monta a matriz de bloco global (vx, vy, p) e resolve um passo de tempo | não conhece a malha nem faz I/O |
| `io.py` | escreve pontos/células/campos em `.vtk` via `meshio` | não decide nomes de arquivo/diretório de saída (isso é do script) |
| `plotting.py` | plota `vx`, `vy`, `p` e `\|v\|` sobre a malha triangular (`matplotlib`) e salva em PNG | não sabe achar o `.vtk` da última iteração nem ler o config (isso é do script) |

## Instalação

Requer Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

Isso instala o pacote `femns` em modo editável, mais as dependências de
runtime (`torch`, `meshio`, `numpy`, `scipy`, `tqdm`, `pyyaml`,
`matplotlib`) e de teste (`pytest`). Se houver GPU CUDA disponível, o
`torch` a detecta e usa automaticamente; caso contrário, roda em CPU.

## Como rodar uma simulação

```bash
source .venv/bin/activate
python scripts/run_simulation.py --config configs/poiseuille.yaml
```

O script lê o config, monta a malha e as matrizes, roda o número de
iterações configurado e escreve um `.vtk` por iteração em `output_dir`
(`solucoes/` por padrão), além de um `CondicaoDeContorno.vtk` inicial
com os valores de contorno aplicados.

### Imagens da última solução

```bash
python scripts/plot_solution.py --config configs/poiseuille.yaml
```

Acha o `.vtk` de maior número de iteração em `output_dir` (o último passo
de tempo escrito) e gera um PNG por variável (`vx`, `vy`, `p`) mais a
magnitude da velocidade `|v|`, em `output_dir/imagens/`. Também aceita um
`.vtk` específico (`--vtk caminho.vtk`) e um diretório de saída
alternativo (`--output-dir`) — ver `femns.plotting.plot_solucao` se
quiser plotar a partir de `points`/`cells`/`point_data` direto (sem
passar por um arquivo).

### Config (`configs/*.yaml`)

```yaml
mesh: meshes/poiseuille.msh
output_dir: solucoes

simulation:
  dt: 0.001
  iterations: 1000
  reynolds: 1
  advection: explicit  # explicit (padrao) ou semi_lagrangian
  sl_boundary: intercept  # intercept (padrao) ou dirichlet -- so vale com semi_lagrangian
  element: mini  # mini (padrao) ou tri6 -- semi_lagrangian ainda so suporta mini

boundary:
  priority: [outlet, inlet, top, bottom]   # do menor para o maior prioridade
  conditions:
    inlet:  {vx: 1.0, vy: 0.0}
    bottom: {vx: 0.0, vy: 0.0}
    top:    {vx: 0.0, vy: 0.0}
    outlet: {p: 0.0}

moving_point:
  enabled: true
  node_index: 345
  amplitude_factor: 0.1
  omega: 628.31853  # pi * 200

# Opcional -- se presente, acrescenta 1 linha de resumo nesse .xlsx ao final
# da simulacao (ver secao "Benchmark entre simulacoes" abaixo).
benchmark_xlsx: solucoes/benchmarks.xlsx
```

- **`boundary.priority`**: quando um nó pertence a mais de um contorno
  (ex.: quina da malha), vale o nome que aparecer **por último** nesta
  lista que o contém. A lista vai da prioridade mais baixa para a mais
  alta.
- **`boundary.conditions`**: cada contorno lista só os componentes
  (`vx`, `vy`, `p`) que são condição de Dirichlet ali. Um componente
  ausente fica livre (natural) naquele contorno — é assim que o outlet
  fica livre em velocidade e as paredes/entrada ficam livres em pressão.
- **`moving_point`**: experimento de malha móvel/ALE — desloca **um** nó
  (`node_index`) em órbita circular a cada iteração, reinterpolando a
  velocidade dele a partir dos vizinhos (IDW). Defina `enabled: false`
  para desativar.
- **`mesh_motion`**: experimento de malha móvel/ALE — faz **todos os nós
  interiores** (os que não estão em nenhum contorno) oscilarem:

  ```yaml
  mesh_motion:
    enabled: true
    amplitude_factor: 0.3   # amplitude = fator × distância ao vizinho mais próximo
    omega: 31.41593         # ver "amostragem no tempo" abaixo
    seed: 0                 # fase/direção sorteadas — mesma semente, mesma oscilação
  ```

  **Amostragem no tempo (armadilha):** o que importa não é `omega` sozinho,
  e sim **quantos passos cabem num período**, `2π/(ω·dt)`. Com `omega: 628.31853`
  (π·200, o default do `moving_point`) e `dt: 0.01` dá **exatamente 1** —
  a malha é amostrada sempre na mesma fase e fica *parada* numa configuração
  deslocada, sem erro nenhum e parecendo uma malha móvel. O
  `run_simulation.py` aborta abaixo de 4 passos por período e avisa abaixo
  de 10. Para `dt: 0.01`, `omega: 31.41593` dá 20 passos por período.

  Cada nó oscila **ao longo de uma reta**, com direção e fase próprias
  (`pos(t) = pos₀ + A·sin(ωt + φ)·[dirₓ, dir_y]`), então vizinhos não se
  movem juntos e a malha "treme" — é o caso mais severo para a busca por
  elemento da advecção semi-Lagrangeana. A amplitude é **local** (por nó),
  não um `h` global: acompanha o refino da malha.

  Os nós de contorno nunca se movem, e os nós extras do elemento
  (centroide do MINI, nós de aresta do Tri6) são recalculados dos
  vértices a cada passo (`mesh.atualiza_nos_extras`) — se não fossem, o
  centroide sairia de dentro do triângulo.

  **Correção ALE do termo convectivo.** Numa malha que se move, o que
  transporta é o quanto o fluido anda *em relação à malha*. A correção
  tem forma diferente em cada caminho de advecção:

  - `advection: explicit` — a velocidade convectiva vira `v − w`, com `w`
    a velocidade da malha (`moving_mesh.velocidade_malha`, derivada
    **analítica** da oscilação, não diferença finita). Com `w = 0` recai
    exatamente no caso Euleriano.
  - `advection: semi_lagrangian` — a correção é **geométrica**, não um
    termo a subtrair: o pé da característica é um ponto do espaço físico,
    então o backtrace parte da posição *nova* do nó, mas o campo que se
    interpola lá é o do passo anterior, discretizado na malha **velha**.
    `calculo_sl` recebe as duas geometrias (`X, Y` de chegada e
    `X_campo, Y_campo` do campo). Sem isso o campo seria avaliado numa
    geometria que não é a dele — medido no degrau, muda até **15% da
    escala de velocidade**, em 98% dos nós.

  Com isto ligado o **sistema é remontado a cada passo** (assembly +
  matriz de bloco + contorno), já que a geometria muda. No degrau isso
  custou ~22% a mais de tempo (18,4 s vs 15,0 s em 20 iterações). A cada
  passo é verificado se algum elemento inverteu (`mesh.area_com_sinal`);
  se inverter, a simulação **aborta** em vez de seguir — `assembly` toma
  `abs` da área mas não dos coeficientes `bi`/`ci`, então um elemento
  invertido entraria com `Gx`/`Gy` de sinal trocado, sem erro nenhum.
  Com `amplitude_factor: 0.3` no degrau não houve inversão, mas a área
  mínima caiu ~4× — `0.3` não é uma garantia formal, é um valor medido
  nesta malha.
- **`simulation.advection`**: `explicit` (padrão) trata a advecção de
  forma explícita, linearizada na velocidade do passo anterior —
  restrita a uma condição de estabilidade tipo CFL no `dt`.
  `semi_lagrangian` usa backtrace + interpolação no pé da
  característica (`femns.semi_lagrangian`) em vez disso, sem essa
  restrição de `dt`, ao custo de alguma difusão numérica adicional e
  de rodar mais devagar por passo (a busca do elemento ainda não é
  vetorizada — ver `docs/semi_lagrangian_strategy.pdf`).
- **`simulation.sl_boundary`**: o que fazer quando o pé da característica
  cai **fora da malha** (só tem efeito com `advection: semi_lagrangian`).
  `dirichlet` usa o valor de Dirichlet do contorno *do próprio nó
  de chegada*, e mantém o valor atual do nó se aquele contorno não
  prescreve aquele componente — o que "congela" a advecção nesse nó.
  `intercept` (padrão) reproduz o código de referência do professor
  (`referencia/clSemiLagrangian.py`, `computeIntercept`): calcula onde a
  característica de fato cruzou o contorno e interpola o campo atual nos
  dois nós daquela aresta. A interpolação na aresta é linear e isso é
  **exato**, não aproximação — sobre uma aresta uma coordenada baricêntrica
  é zero, a correção de bolha do MINI se anula e a base vira P1.
  Os dois só divergem quando nós **interiores** saem do domínio: com `dt`
  pequeno saem só nós do próprio contorno de entrada, onde as duas
  respostas coincidem (ver `docs/relatorio_sl_boundary.md`).
- **`simulation.element`**: `mini` (padrão) usa o elemento MINI
  (P1 + bolha / P1). `tri6` usa o elemento Tri6 (Taylor-Hood, P2/P1) —
  velocidade quadrática (vértices + nó de aresta), pressão linear só
  nos vértices. Os dois montam `Gx`/`Gy` na mesma forma fraca
  (`∫ φⱼ ∂Nᵢ/∂x`), então o sistema de sela fica simétrico (`C = Bᵀ`)
  nos dois casos — o Tri6 não é caso especial no solver. A referência
  em `referencia/ref tri6/` também define uma variante "slip" do
  gradiente que **não** é usada aqui: ela quebra essa simetria e faz o
  BiCGSTAB divergir (ver `femns.assembly.assemble_tri6`). Ainda não
  combina com `advection: semi_lagrangian` (a interpolação de
  `femns.semi_lagrangian` é específica do MINI).

Para testar rápido, copie o config e reduza `iterations` (ex.: 5) antes
de apontar `--config` para a cópia — os `.vtk` de uma malha grande levam
alguns segundos por iteração no solve denso atual (ver limitação abaixo).

## Testes

```bash
python -m pytest tests/ -v
```

Os testes cobrem conectividade de malha (`NToN`/`EToE`), montagem dos
elementos MINI e Tri6 contra valores analíticos (calculados à mão e/ou
verificados por integração simbólica exata com `sympy`) para um
triângulo de referência, e resolução de prioridade/condições de
contorno — nenhum depende de arquivo `.msh` nem de GPU.

## Lint

```bash
pip install -e ".[dev]"
ruff check src scripts tests
```

`line-length` está configurado em 200 no `pyproject.toml` porque
`assembly.py` tem fórmulas de elemento finito inerentemente longas —
quebrar essas linhas piora a leitura em vez de melhorar.

## Solver linear (BiCGSTAB esparso)

`solver.time_step` resolve o sistema global a cada passo de tempo com
BiCGSTAB (`femns.krylov.bicgstab`), 100% em PyTorch — sem depender de
nenhuma biblioteca de solver esparso do fabricante da GPU (funciona
igual em CUDA, ROCm ou CPU). Dois detalhes fazem esse solver convergir
de fato no sistema de sela (saddle-point) de Navier-Stokes:

- **Reescalonamento de pressão** (`solver.pressure_scale_factor`): o
  termo `1/dt` no bloco de velocidade deixa esse bloco ordens de
  grandeza maior que o bloco de acoplamento de pressão (`Gx`, `Gy`).
  Sem corrigir isso, o BiCGSTAB (com ou sem pré-condicionador de
  Jacobi) converge para uma solução completamente errada mesmo com
  resíduo aparentemente pequeno — mau condicionamento clássico de
  sistemas de sela. `build_system_matrix` aplica um fator `beta` nas
  colunas/linhas de pressão para equilibrar as escalas; a pressão é
  resolvida numa variável escalada `p_tilde = p / beta` e convertida de
  volta em `time_step`.
- **Warm-start entre iterações**: a solução do passo anterior é
  reaproveitada como chute inicial do próximo `bicgstab`, já que passos
  de tempo consecutivos (Euler implícito, `dt` pequeno) têm soluções
  próximas — isso reduz o número de iterações necessárias a cada passo.

Na malha do Poiseuille de exemplo (~19 mil incógnitas), isso roda a
~0.35s/iteração na GPU, contra ~6.5s/iteração do solve dense anterior
(`torch.linalg.solve(A.to_dense(), b)`) — a solução numérica é idêntica
nos dois casos, só a forma de resolver o sistema linear mudou.

## Benchmark entre simulações

Se o config tiver a chave `benchmark_xlsx` (ver seção "Config" acima),
`run_simulation.py` acrescenta, ao final, 1 linha de resumo nesse arquivo
`.xlsx` (`femns.report.salvar_resumo`) — cada run vira uma linha nova, sem
sobrescrever runs anteriores, pensado pra comparar `dt`/malha/Re/`advection`
lado a lado numa tabela só. Colunas registradas:

| Grupo | Colunas |
|---|---|
| Identificação | `timestamp`, `mesh`, `advection`, `sl_boundary`, `element`, `dt`, `reynolds`, `iterations`, `npoints`, `ne`, `device` |
| Desempenho | `tempo_total_s`, `tempo_assembly_s`, `tempo_medio_por_iter_s` |
| Convergência do BiCGSTAB | `bicg_iters_media`, `bicg_iters_max`, `bicg_residual_media`, `bicg_residual_max`, `passos_nao_convergidos` |
| Qualidade física da solução | `vel_l2_media`, `vel_l2_max`, `vel_linf_max`, `divergencia_l2_media`, `divergencia_l2_max`, `pressao_min`, `pressao_max`, `pressao_media` |

`divergencia_l2_*` mede `Dx@vx + Dy@vy` (`Dx = -Gx^T`, `Dy = -Gy^T`, sem o
fator `beta` — que é só escala numérica do solver, não faz parte da
restrição física de incompressibilidade) a cada passo. Deveria ficar
próximo de zero; um valor crescendo ao longo dos passos é sinal de
instabilidade numérica antes mesmo da velocidade "explodir" visivelmente.
Sem `benchmark_xlsx` no config, nada disso é calculado — o custo extra
(pequeno: 2 produtos matriz-vetor esparsos por passo) só é pago quando
pedido.

## Limitações conhecidas / próximos passos

- Sem suporte a outras geometrias além de malhas triangulares (MINI ou
  Tri6; sem elementos quadrilaterais).
- Tri6 ainda não tem gerador de malha para as demais combinações
  experimentais (`moving_point` não foi testado com Tri6; `advection:
  semi_lagrangian` não é suportado com `element: tri6`, ver seção
  "Config").
- `moving_mesh.py` move um único nó como experimento pontual; não é um
  esquema de malha móvel/ALE completo (não remonta a malha a partir do
  deslocamento).
- `tol`/`max_iter` do BiCGSTAB estão com valores padrão fixos em
  `time_step` (`1e-8`/`2000`), não expostos no config yaml — ajustar
  diretamente no código se uma malha maior precisar de outros valores.

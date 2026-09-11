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
│   ├── moving_mesh.py        # malha móvel: órbita de um nó (moving_point) ou oscilação de todos os nós interiores (mesh_motion) + velocidade da malha para a correção ALE
│   ├── solver.py             # montagem do sistema global e passo de tempo (Euler implícito)
│   ├── io.py                 # escrita dos resultados em VTK
│   ├── plotting.py           # imagens (PNG) da distribuição espacial das variáveis
│   └── webui/                # GUI web opcional (ver seção "Interface web") -- não usado pelo solver
├── scripts/
│   ├── run_simulation.py    # CLI: orquestra mesh → assembly → boundary → solver → io
│   ├── run_gui.py           # CLI: sobe a interface web opcional (ver seção "Interface web")
│   ├── plot_solution.py     # CLI: gera as imagens da última solução de uma simulação
│   └── plot_linha.py        # CLI: amostra um campo sobre uma linha e grava .json (formato plots/)
├── tests/                    # testes unitários (conectividade, valores analíticos dos elementos MINI/Tri6, contornos)
├── solucoes/                  # saída dos .vtk (gerada em runtime, ignorada pelo git)
├── femns-gui                  # executável: wrapper de shell pra scripts/run_gui.py (ver "Interface web")
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
| `moving_mesh.py` | move a malha: um nó em órbita (`moving_point`, com reinterpolação IDW da velocidade dele) ou **todos os nós interiores** oscilando ao longo de retas próprias (`mesh_motion`); também fornece a velocidade da malha (derivada analítica) para a correção ALE | não remonta o sistema (isso é do script, que remonta a cada passo quando `mesh_motion` está ligado) |
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
  advection: eulerian  # eulerian (padrao) ou semi_lagrangian
  sl_boundary: intercept  # intercept (padrao) ou dirichlet -- so vale com semi_lagrangian
  element: mini  # mini (padrao) ou tri6
  vtk_interval: 10  # grava 1 .vtk a cada N iteracoes (padrao 10); a ultima e' sempre gravada
  solver:                 # BiCGSTAB -- bloco opcional
    tol: 1.0e-8           # residuo relativo alvo por passo (padrao 1e-8)
    max_iter: 2000        # teto de iteracoes por passo (padrao 2000)
    beta_por_passo: false # recalcular a equilibracao de pressao a cada passo
                          #   de malha movel (padrao false)

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

  Cada valor pode ser um número (condição uniforme) ou um **perfil**:

  ```yaml
  inlet: {vx: {perfil: parabolico, vmax: 1.5}, vy: 0.0}
  ```

  `parabolico` dá `v(s) = 4·vmax·(s−s₀)(s₁−s)/(s₁−s₀)²` — zero nas pontas
  do contorno, `vmax` no meio. As pontas `s₀`, `s₁` saem da **extensão
  geométrica** do contorno (todos os nós dos segmentos com aquele nome),
  então o mesmo config vale para qualquer altura de canal; o eixo (`x` ou
  `y`) é detectado pela direção em que o contorno se estende. Como a média
  de uma parábola é 2/3 do pico, `vmax: 1.5` tem a mesma vazão de
  `vx: 1.0` uniforme.

  A distinção "extensão geométrica" vs "nós que recebem o valor" não é
  detalhe: as quinas da entrada pertencem à parede por `priority`
  (no-slip, correto), então saem do conjunto que recebe o perfil. Ancorar
  a parábola nos nós restantes a encolhe de `h` em cada ponta — a vazão
  cai ~2h/H (6,4% numa malha com h≈0,03) **com a vazão ainda conservada
  seção a seção**, o que faz o erro parecer discretização.

  Impor o perfil desenvolvido na entrada é o que torna o Poiseuille um
  caso de **validação exata**: a solução analítica `6y(1−y)` passa a
  valer no canal inteiro, sem região de desenvolvimento (ver
  `scripts/plot_poiseuille_validacao.py`).
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

  - `advection: eulerian` — a velocidade convectiva vira `v − w`, com `w`
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
- **`simulation.advection`**: `eulerian` (padrão) trata a advecção de
  forma explícita, linearizada na velocidade do passo anterior —
  restrita a uma condição de estabilidade tipo CFL no `dt`.
  `semi_lagrangian` usa backtrace + interpolação no pé da
  característica (`femns.semi_lagrangian`) em vez disso, sem essa
  restrição de `dt`, ao custo de alguma difusão numérica adicional e
  de rodar mais devagar por passo (ver `docs/semi_lagrangian_strategy.pdf`).
  A busca por elemento é vetorizada (caminhada em paralelo entre nós via
  máscara booleana); numa malha real de 8161 nós ela custa ~1,7 ms.
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
  BiCGSTAB divergir (ver `femns.assembly.assemble_tri6`). Combina com
  `advection: semi_lagrangian` — a interpolação no pé da característica
  usa a base P2 completa (`interpolar_tri6`), validada contra campo
  quadrático arbitrário (ver `docs/semi_lagrangian_tri6.md`).
- **`simulation.vtk_interval`**: grava um `solucao -N.vtk` a cada `N`
  iterações (padrão `10`). A **última** iteração é sempre gravada, então
  o estado final não depende de `iterations` ser múltiplo do intervalo,
  e `femns.plotting.ultimo_vtk` / o visualizador web continuam achando o
  campo final. `vtk_interval: 1` grava todo passo (comportamento
  anterior). Não afeta a numérica — só a densidade de frames no disco e
  o custo de I/O em runs longos.
- **`simulation.solver`** (opcional): `tol` (resíduo relativo alvo do
  BiCGSTAB por passo de tempo, padrão `1e-8`) e `max_iter` (teto de
  iterações por passo, padrão `2000`) — os valores que antes estavam
  fixos em `solver.time_step`. Uma malha maior ou um caso mais rígido
  pode precisar de outro `max_iter` (o degrau Tri6 a Re=800 com malha
  móvel precisa de ~6000 no transiente para atingir `1e-8`); afrouxar
  `tol` troca precisão da solução por velocidade. Ausente, cai nos
  padrões e nada muda.
  `beta_por_passo` (padrão `false`) só tem efeito com `mesh_motion`:
  recalcula `pressure_scale_factor` a cada passo (a equilibração
  `‖A_vv‖/‖(Gx;Gy)‖` afinada na geometria inicial degrada quando a malha
  deforma muito) e reescala o componente de pressão do warm-start.
- **`mesh_motion` e a remontagem**: com a malha móvel o sistema `A` é
  remontado a cada passo. A topologia não muda — só os valores —, então
  `femns.reassembly.Remontador` captura o padrão de esparsidade e a
  permutação do `coalesce` uma vez e nos passos seguintes só recomputa
  valores (pula o `torch.cat` dos 9 blocos, o `coalesce` grande e o
  `to_sparse_csr`). Verifica contra a montagem de referência no início
  (imprime `erro de verificacao`, ~`1e-17`).

Para testar rápido, copie o config e reduza `iterations` (ex.: 5) antes
de apontar `--config` para a cópia — cada iteração numa malha grande
custa alguns segundos (BiCGSTAB no sistema de sela + escrita do `.vtk`).

## Testes

```bash
python -m pytest tests/ -v
```

Os testes cobrem conectividade de malha (`NToN`/`EToE`), montagem dos
elementos MINI e Tri6 contra valores analíticos (calculados à mão e/ou
verificados por integração simbólica exata com `sympy`) para um
triângulo de referência e para dois elementos partilhando um nó
(acumulação da montagem vetorizada), e resolução de prioridade/condições
de contorno — nenhum depende de arquivo `.msh` nem de GPU.

`ruff check src scripts tests` roda o linter (config em `pyproject.toml`;
`referencia/` fica de fora, é código do professor verbatim).

`.github/workflows/ci.yml` roda `ruff` + `pytest` a cada `push`/PR no
GitHub (Ubuntu, Python 3.12, `torch` CPU-only) — a suíte inteira em
CPU, sem simulação de verdade.

## Interface web (GUI opcional)

```bash
./femns-gui                          # abre o navegador em http://127.0.0.1:8765/
./femns-gui --port 9000 --no-browser
```

Por padrão o servidor faz bind em `127.0.0.1` (só aceita conexão da
própria máquina) e **não tem autenticação**. Para acesso remoto o
caminho recomendado é um túnel SSH, que não exige mudar nada
(`ssh -N -L 8765:127.0.0.1:8765 usuario@servidor`, depois abrir
`http://127.0.0.1:8765/` no cliente). Ver `docs/planos/plano_remoto.md` para o passo
a passo, systemd e Tailscale.

Se ainda assim precisar de bind na rede, `--host 0.0.0.0` **exige** um
token no ambiente:

```bash
FEMNS_GUI_TOKEN=$(openssl rand -hex 16) ./femns-gui --host 0.0.0.0 --no-browser
```

Com `FEMNS_GUI_TOKEN` setado, toda rota `/api/*` passa a exigir o token
(via `Authorization: Bearer` ou `?token=…`); a URL que o servidor imprime
e abre no navegador já traz `?token=…`, e o frontend repassa o token em
todas as chamadas. Sem token, `--host` não-local é recusado com uma
mensagem explicando as opções — `--allow-no-auth` derruba essa trava
(inseguro; só para rede isolada). HTTPS continua por sua conta (proxy
reverso); o token vai em claro sem ele.

`femns-gui` (executável na raiz do projeto) é um wrapper de shell fino em
volta de `scripts/run_gui.py`: acha o Python certo (`.venv/` ao lado dele
por padrão, `FEMNS_VENV=/caminho/para/outro/.venv ./femns-gui` pra apontar
pra outro) e falha com uma mensagem clara se o pacote `femns` não estiver
instalado nele, em vez do traceback de `ModuleNotFoundError`. Equivalente
a rodar direto:

```bash
python scripts/run_gui.py --port 9000 --no-browser
```

Módulo à parte (`src/femns/webui/`), não importado por nenhum módulo do
solver nem por `run_simulation.py` — a GUI é só mais um cliente do
pacote `femns`, e o CLI continua funcionando sem ela instalada/rodando.
Sem dependências novas: o servidor é só biblioteca padrão do Python
(`http.server`), servindo um frontend estático (HTML/CSS/JS puro, sem
build step) em `src/femns/webui/static/`.

Duas telas:

- **Nova simulação**: escolhe a malha (lê os nomes de contorno reais do
  `.msh`, físical groups do Gmsh, via `femns.webui.meshes`), preenche
  `dt`/`iterations`/`reynolds`/`advection`/`element`/`sl_boundary`, o
  intervalo de gravação de `.vtk` (campo "Gravar .vtk a cada" na área de
  template, logo abaixo do rótulo — `simulation.vtk_interval`, padrão 10)
  e `tol`/`max_iter` do BiCGSTAB (seção "Solver linear",
  `simulation.solver`),
  monta `boundary.priority`/`boundary.conditions` num editor (reordenar
  prioridade, marcar `vx`/`vy`/`p` por contorno, e por componente escolher
  entre número, função de `y` ou perfil parabólico no seletor ao lado) —
  com um seletor de
  template que pré-preenche tudo a partir de um `configs/*.yaml`
  existente. "Salvar configuração" grava o formulário inteiro (nome
  escolhido na hora) em `configs/gui_saved/` via `femns.webui.saved_configs`
  (git-ignorado, local do usuário — distinto dos templates curados em
  `configs/*.yaml`); o seletor "Configuração salva" ao lado recarrega
  qualquer uma de volta no formulário (inclusive `moving_point`/
  `mesh_motion`), com botão de excluir. Salvar não valida nada (dá pra
  guardar um rascunho incompleto); só "Rodar simulação" valida de fato.
  "Rodar simulação" grava um config novo em `configs/gui/` e lança
  `scripts/run_simulation.py` como subprocesso
  (`femns.webui.jobs`); a tela acompanha progresso (quadros
  `solucao -N.vtk` escritos vs. os esperados, `⌈iterations/vtk_interval⌉`)
  e log em tempo real, com botão de cancelar.
  Saída em `solucoes/gui/<rótulo>-<timestamp>/` (git-ignorado, mesmo
  tratamento do resto de `solucoes/`). O editor de condição de contorno
  cobre as três formas que `boundary.valores_da_condicao` aceita: número
  uniforme, função de `y` (ex. `4*y*(1-y)`) e perfil parabólico (o campo
  de valor passa a ser o `vmax`) — o seletor `núm / ƒ(y) / parábola` ao
  lado de cada componente escolhe qual. Só uma forma de dict fica de fora
  (nenhuma outra existe hoje): um valor que o editor não reconheça é
  deixado sem marcar em vez de mostrar valor errado.
- **Resultados**: lista qualquer diretório sob `solucoes/` que tenha
  `solucao -N.vtk` (runs da GUI ou do CLI direto), carrega os quadros
  sob demanda (`femns.webui.results`, um `.vtk` por requisição, não tudo
  de uma vez) e desenha o campo escolhido (`vx`, `vy`, `p` ou `|v|`)
  num canvas 2D (colormap `jet`, sombreamento plano por triângulo).
  Play/pause, passo a passo, slider de quadro, toggle de malha.
  Traçando uma linha sobre o campo sai um gráfico de perfil ao longo
  dela, com curvas de referência carregadas de `.json` (ver `plots/`) e
  um gráfico de erro relativo ao lado; passar o mouse sobre um ponto de
  qualquer curva mostra `x`/`y`/`dist` e o valor.

Todo endpoint que recebe path do cliente (malha, run de resultado)
valida contra uma lista construída no servidor — o cliente nunca manda
um path de arquivo direto, só nomes/ids opacos (ver `jobs.validate_config`
e o índice `run_id → dir` em `server.py`). Isso barra leitura de arquivo
arbitrário, mas **não** controla *quem* chama as rotas — daí o bind local
por padrão e o token de `FEMNS_GUI_TOKEN` para bind na rede (`server.check_token`
protege `/api/*`; o shell estático continua aberto para o frontend poder
carregar e então mandar o token).

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

O termo convectivo Euleriano (`solver.time_step`, caminho
`advection: eulerian`) é aplicado direto como
`cx ⊙ (Gvx @ w) + cy ⊙ (Gvy @ w)`, sem formar `diag(cx)` densa `n×n` —
era `O(n²)` de VRAM por passo e limitava a malha a ~20 mil elementos;
agora escala como o resto do solver. Resultado algebricamente idêntico.

**Casos que não convergem em `max_iter=2000`:** o degrau Tri6 a Re=800
(estático precisa de ~6000 iterações; com `mesh_motion` o transiente
inicial fica em resíduo ~`1e-3`, e só depois que o escoamento assenta o
warm-start traz o resíduo para perto de `1e-8`). São o regime que
motiva um pré-condicionador de bloco (ver `docs/planos/plano_07set.md`
§6) — a equilibração escalar sozinha não basta a Re alto.

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
- `moving_point` (órbita de um nó) não foi testado com `element: tri6` —
  só com MINI. `advection: semi_lagrangian` está validado com os dois
  elementos; `mesh_motion` roda com os dois (nós extras recalculados dos
  vértices; a remontagem incremental de `femns.reassembly` é verificada
  contra a montagem do zero para MINI e Tri6).
- Elementos com orientação horária não são tratados de forma robusta
  (as malhas do projeto têm zero; `assembly` toma `abs` da área mas não
  dos coeficientes `bi`/`ci`). Com `mesh_motion` ligado isso deixa de
  ser hipotético — daí o aborto por inversão de elemento a cada passo.
- A oscilação de `mesh_motion` é sintética (fase/direção aleatórias por
  nó) — um experimento de robustez da malha móvel/ALE, não um caso de
  uso. Falta um esquema movido por deslocamento físico real (o projeto
  de mestrado descreve uma válvula fechando).
- Pré-condicionador de bloco / complemento de Schur ainda não
  implementado: hoje o BiCGSTAB roda com equilibração escalar de pressão
  e sem pré-condicionador (Jacobi piorou a convergência). Uma malha bem
  maior deve fazer o número de iterações crescer com `1/h` — ver
  `docs/planos/plano_07set.md` §6 para o levantamento de pré-condicionadores.

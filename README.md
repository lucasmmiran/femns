# femns

Solver de elementos finitos (método de Galerkin) para as equações de
Navier-Stokes incompressíveis, com montagem vetorizada em GPU (PyTorch).

Discretização espacial pelo elemento MINI (P1 + bolha para velocidade,
P1 para pressão) em malhas triangulares, com avanço temporal por Euler
implícito e termo advectivo linearizado na velocidade do passo anterior.

## Estrutura do projeto

```
.
├── configs/
│   └── poiseuille.yaml     # parâmetros de uma simulação (malha, dt, Re, contornos, ...)
├── meshes/
│   └── poiseuille.geo/.msh # malha gerada no Gmsh (.geo é a fonte, .msh o resultado)
├── src/femns/               # pacote com a física/numérica do solver
│   ├── mesh.py              # leitura da malha, conectividade (NToN/EToE), elemento MINI
│   ├── assembly.py          # montagem vetorizada das matrizes de elemento (K, M, Gx, Gy, Gvx, Gvy)
│   ├── boundary.py          # condições de contorno: nomes por nó, valores, aplicação na matriz
│   ├── moving_mesh.py        # deslocamento senoidal de um nó (experimento de malha móvel)
│   ├── solver.py             # montagem do sistema global e passo de tempo (Euler implícito)
│   └── io.py                 # escrita dos resultados em VTK
├── scripts/
│   └── run_simulation.py    # CLI: orquestra mesh → assembly → boundary → solver → io
├── tests/                    # testes unitários (conectividade, valores analíticos do elemento MINI, contornos)
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
| `mesh.py` | lê `.msh`, monta elemento MINI (nó de centroide), conectividade nó-a-nó (`NToN`) e elemento-a-elemento (`EToE`) | não sabe de condição de contorno nem de matrizes de rigidez |
| `assembly.py` | monta `K, M, Gx, Gy, Gvx, Gvy` (rigidez, massa, gradiente) por elemento, vetorizado em `torch` | não monta o sistema global nem aplica contorno |
| `boundary.py` | resolve qual nome de contorno cada nó tem (por prioridade) e monta os índices/valores de Dirichlet; zera linhas da matriz global e põe 1 na diagonal | não decide *quais* são os valores físicos — isso vem do config |
| `moving_mesh.py` | desloca um nó da malha em trajetória senoidal e reinterpola sua velocidade a partir dos vizinhos (IDW) | não atualiza a malha inteira, só o nó indicado |
| `solver.py` | monta a matriz de bloco global (vx, vy, p) e resolve um passo de tempo | não conhece a malha nem faz I/O |
| `io.py` | escreve pontos/células/campos em `.vtk` via `meshio` | não decide nomes de arquivo/diretório de saída (isso é do script) |

## Instalação

Requer Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

Isso instala o pacote `femns` em modo editável, mais as dependências de
runtime (`torch`, `meshio`, `numpy`, `scipy`, `tqdm`, `pyyaml`) e de teste
(`pytest`). Se houver GPU CUDA disponível, o `torch` a detecta e usa
automaticamente; caso contrário, roda em CPU.

## Como rodar uma simulação

```bash
source .venv/bin/activate
python scripts/run_simulation.py --config configs/poiseuille.yaml
```

O script lê o config, monta a malha e as matrizes, roda o número de
iterações configurado e escreve um `.vtk` por iteração em `output_dir`
(`solucoes/` por padrão), além de um `CondicaoDeContorno.vtk` inicial
com os valores de contorno aplicados.

### Config (`configs/*.yaml`)

```yaml
mesh: meshes/poiseuille.msh
output_dir: solucoes

simulation:
  dt: 0.001
  iterations: 1000
  reynolds: 1

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
```

- **`boundary.priority`**: quando um nó pertence a mais de um contorno
  (ex.: quina da malha), vale o nome que aparecer **por último** nesta
  lista que o contém. A lista vai da prioridade mais baixa para a mais
  alta.
- **`boundary.conditions`**: cada contorno lista só os componentes
  (`vx`, `vy`, `p`) que são condição de Dirichlet ali. Um componente
  ausente fica livre (natural) naquele contorno — é assim que o outlet
  fica livre em velocidade e as paredes/entrada ficam livres em pressão.
- **`moving_point`**: experimento de malha móvel/ALE — desloca o nó
  `node_index` em trajetória senoidal a cada iteração. Defina
  `enabled: false` para desativar.

Para testar rápido, copie o config e reduza `iterations` (ex.: 5) antes
de apontar `--config` para a cópia — os `.vtk` de uma malha grande levam
alguns segundos por iteração no solve denso atual (ver limitação abaixo).

## Testes

```bash
python -m pytest tests/ -v
```

Os testes cobrem conectividade de malha (`NToN`/`EToE`), montagem do
elemento MINI contra valores analíticos calculados à mão para um
triângulo de referência, e resolução de prioridade/condições de
contorno — nenhum depende de arquivo `.msh` nem de GPU.

## Limitações conhecidas / próximos passos

- **Solve denso a cada iteração**: `solver.time_step` converte a matriz
  esparsa para densa (`A.to_dense()`) antes de resolver. Isso funciona
  mas anula boa parte do ganho de usar uma matriz esparsa/GPU conforme a
  malha cresce. Fica como próximo passo revisitar um solver esparso
  (direto ou iterativo) que seja estável para este sistema de sela
  (saddle-point).
- Sem suporte a outras geometrias além de malhas triangulares com o
  elemento MINI (sem P2, sem elementos quadrilaterais).
- `moving_mesh.py` move um único nó como experimento pontual; não é um
  esquema de malha móvel/ALE completo (não remonta a malha a partir do
  deslocamento).

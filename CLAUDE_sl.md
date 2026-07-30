# Solver de Navier–Stokes por MEF — convenções do projeto

Solver 2D de escoamento incompressível, elementos finitos triangulares, termo
convectivo por método semi-lagrangeano. Código em Python + torch (tensores,
com suporte a GPU). Comunicação e comentários em português.

## Metodologia do termo convectivo

O semi-lagrangeano segue o Algoritmo 1 de 1ª ordem:

1. **Trajetória** — pé da característica por Euler explícito: `x_d = x - dt*v`
2. **Interpolação** — o operador Π é **materializado como matriz esparsa**
   (`conv`), não aplicado ponto a ponto. `phi_d = conv @ phi`
3. **Derivada material** — fica com o solver: `(M/dt + K) phi^{n+1} = (M/dt) phi_d + c.c.`

Materializar Π é deliberado: a mesma matriz é aplicada a `vx`, `vy` e a
qualquer outro escalar advectado no mesmo passo, amortizando a busca. Ela
depende da velocidade, logo é **remontada a cada passo de tempo**.

## Convenções que não podem ser violadas

- `IEN[e, 0:3]` são sempre os três vértices, em ordem **anti-horária**.
  Colunas seguintes são nós de aresta / bolha, conforme o elemento.
- `EToE[e, f]` é o vizinho pela aresta que liga os vértices locais `f` e
  `(f+1) % 3`; vale `-1` no contorno.
- Na busca, o vértice de baricêntrica mais negativa `m` escapa pela face
  `(m+1) % 3`. **Nunca** use `EToE[e, m]` direto — essa é a convenção do
  código de referência do professor (`oface`), incompatível com a nossa.
  Misturar as duas não gera erro: a caminhada apenas anda para o lado errado.
- Coordenadas baricêntricas vêm dos coeficientes afins pré-computados
  (`l_k = (a_k + b_k x + c_k y) / det`), montados uma vez por malha.
  **Não** escreva fórmula de área (shoelace) inline — já causou bug silencioso.

## Estrutura do código

Hierarquia de classes: a base `SemiLagrangianTri` contém todo o algoritmo
(backtrace, caminhada, montagem de Π, contorno, Dirichlet); cada subclasse
contém apenas a geometria do seu elemento.

**Para adicionar um elemento novo, declare quatro coisas e nada mais:**

- `NEN` — número de nós por elemento
- `EDGE_LOCAL` — para cada face `f`, os índices locais dos nós daquela aresta,
  na ordem `[nó em s=1, nó em s=0, ...internos...]`
- `_shape(lam)` — funções de forma no interior, a partir das baricêntricas
- `_edge_shape(s)` — funções de forma de linha na fronteira

Não duplique a lógica de busca na subclasse. Se algo do algoritmo precisar
mudar por elemento, discuta antes — provavelmente a abstração está errada.

## Estilo

- **Vetorizado, não laço Python.** A busca do elemento de partida usa
  "frente de onda": cada iteração dá um salto simultâneo em todos os pontos
  ativos, via máscara booleana. O laço externo roda `max_iter` vezes, não `n`.
- **Sem `multiprocessing`.** O paralelismo vem do backend do torch; com
  `device="cuda"` a busca inteira roda na GPU sem transferências.
- Pré-computo (geometria, índices de Dirichlet) vai no `__init__`, fora do
  loop temporal. Só o que depende da velocidade entra em `compute`.
- Métodos internos com `_` no nome; API pública enxuta.
- `float64` por padrão — o método é sensível a erro de interpolação.

## Protocolo de validação (obrigatório para código novo)

Todo elemento ou mudança no núcleo do SL precisa passar por **reprodução
polinomial exata**: um elemento de ordem `p` reproduz qualquer polinômio de
grau ≤ `p` até precisão de máquina (~1e-15). Advecte o campo com velocidade
uniforme e compare com a solução analítica nos nós de status 1.

Inclua sempre um **teste negativo** — por exemplo, Tri3 com campo quadrático
deve falhar na ordem de 1e-3. Sem ele, não há como saber se o teste positivo
é significativo.

Verifique também, a cada mudança:
- partição da unidade: `conv @ ones` deve dar 1 em todas as linhas
- linhas de fronteira: número correto de entradas, somando 1
- contagem de status: `dentro`, `fronteira`, `não-convergido`

## Armadilhas conhecidas

- Erro na convenção de face não gera exceção — só degrada a convergência.
  Ao mexer em `_locate` ou em `montar_EToE`, rode o teste polinomial.
- `_intercept` (projeção na fronteira) precisa tratar `det ≈ 0` (trajetória
  paralela à aresta) e limitar `s` a [0,1]. Sem isso: divisão por zero ou
  peso negativo.
- Nós que não convergem em `max_iter` recebem linha identidade, igual aos que
  saíram do domínio. Se a contagem de não-convergidos crescer, o passo de
  tempo está grande demais para `max_iter`.
- torch emite `UserWarning` sobre tensores esparsos (invariantes e CSR beta).
  São benignos; filtre no script principal para não poluir o log.

## Limitações atuais (candidatos a evolução)

- Backtrace de 1ª ordem (Euler) domina o erro temporal — RK2 elevaria a ordem
  sem mexer no resto da estrutura.
- `Tri10` está implementado mas **não validado**: confira a ordenação dos nós
  de aresta contra a malha antes de usar.
- Não há limitador de monotonicidade; elementos de ordem alta têm funções de
  forma negativas e podem oscilar em frentes abruptas.

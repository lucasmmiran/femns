# Estratégia de pontos móveis — malha de triângulos para válvula esfera 2D

Documento de referência para reaproveitar a estratégia de malha móvel (ALE) da válvula esfera em outro projeto. O código de referência é `valvula_malha_movel.py`.

---

## 1. Problema

Uma esfera com furo passante gira de um ângulo θ dentro de um tubo reto. O domínio fluido muda de forma: aparecem arcos de parede novos nas quatro junções entre o tubo e a esfera. A malha de triângulos precisa acompanhar essa mudança sem mudar o número de nós e sem gerar triângulos invertidos.

## 2. Geometria e notação

| Símbolo | Significado |
|---|---|
| H | meia-altura do tubo (paredes em y = ±H) |
| L1 | comprimento da parede do furo (corda) |
| Lp | meio-comprimento do domínio (entrada em x = −Lp, saída em x = +Lp) |
| r | raio da esfera, r² = H² + (L1/2)² |
| φ1 | asen(H/r) |
| φ2, φ3, φ4 | π − φ1, π + φ1, 2π − φ1 |
| θ | rotação da esfera, anti-horária |
| S | arco exposto, S = r·θ |
| P(α) | ponto do círculo, r·(cos α, sen α) |

As quinas do furo em θ = 0 são P(φ1) = (L1/2, H), P(φ2) = (−L1/2, H), P(φ3) = (−L1/2, −H) e P(φ4) = (L1/2, −H).

**Pareamento correto das quinas.** A parede de cima do furo liga φ2 a φ1 e a de baixo liga φ3 a φ4. As duas têm comprimento L1. As cordas φ1–φ4 e φ2–φ3 são as bocas do furo, com comprimento 2H.

**Rotação rígida (interior da esfera):**

```
x = xi·cosθ − yi·senθ
y = xi·senθ + yi·cosθ
```

**Domínio fluido em θ:**

```
Ω(θ) = (tubo fora da esfera)  ∪  (furo girado dentro da esfera)
     = { |y| < H, |p| ≥ r }   ∪  { |p| < r, |−x·senθ + y·cosθ| < H }
```

A região do tubo não muda de forma. Só a região do furo gira. Os arcos novos são as partes do círculo onde as duas regiões não se sobrepõem:

- arco φ2 → φ2+θ: superfície da esfera bloqueando o tubo de entrada (em cima);
- arco φ1 → φ1+θ: cavidade do corpo exposta pelo furo (em cima);
- arco φ3 → φ3+θ: cavidade do corpo (embaixo);
- arco φ4 → φ4+θ: superfície da esfera bloqueando o tubo de saída (embaixo).

**Fechamento total:** θ = 2·φ1. A fresta em cada lado mede 2r·sen(φ1 − θ/2).

## 3. Cada parede é um único caminho parametrizado por comprimento de arco

Da entrada para a saída:

```
cima : tubo (−Lp,H)→P(φ2) | arco φ2→φ2+θ | corda P(φ2+θ)→P(φ1+θ) | arco φ1+θ→φ1 | tubo P(φ1)→(Lp,H)
baixo: tubo (−Lp,−H)→P(φ3)| arco φ3→φ3+θ | corda P(φ3+θ)→P(φ4+θ) | arco φ4+θ→φ4 | tubo P(φ4)→(Lp,−H)
```

Comprimentos das peças: (Lp − L1/2), S, L1, S, (Lp − L1/2). O trecho em contato com a válvula passa de **L1 para L1 + 2S**, um arco S em cada ponta.

Para avaliar um ponto no caminho:

1. calcule as bordas acumuladas `borda = [0, cumsum(comprimentos)]`;
2. ache a peça k com `borda[k] ≤ s < borda[k+1]`;
3. use `u = (s − borda[k]) / comprimento[k]`;
4. se for arco, α = α_ini + u·(α_fim − α_ini) e ponto = P(α); se for segmento, ponto = (1 − u)·A + u·B.

As fórmulas do caderno, `x = r·cos(θ + φi)`, `y = r·sen(θ + φi)`, dão a **extremidade** de cada arco. O arco inteiro é P(φi + β) com β de 0 a θ.

## 4. Mapa de redistribuição s0 → s(θ)

Cada nó de parede guarda sua coordenada de arco de referência s0 (em θ = 0 a parede é reta e s0 = x + Lp). A posição em θ vem de um mapa linear por partes:

| Referência s0 | Posição s(θ) | Papel |
|---|---|---|
| 0 | 0 | entrada, fixo |
| A = Lp − L1/2 − b | A | início do buffer esquerdo |
| B = Lp − L1/2 | B + S | quina do furo φ2 (ou φ3) |
| C = Lp + L1/2 | C + S | quina do furo φ1 (ou φ4) |
| D = Lp + L1/2 + b | D + 2S | fim do buffer direito |
| 2Lp | 2Lp + 2S | saída, fixo |

Propriedades:

- **Entre B e C,** s = s0 + S. Os nós da corda caem exatamente na rotação rígida, então as quinas do furo têm nó exato.
- **Entre A e B e entre C e D,** o fator de esticamento é (b + S)/b. Os nós de uma faixa b do tubo escorregam e sobem nos arcos.
- **Fora do buffer,** o parâmetro muda mas a posição física não. O tubo longe da válvula fica parado.
- **Não incremental:** a posição do contorno depende só de θ. Fechar e reabrir volta à mesma posição.

**Por que não o esticamento uniforme Δx = (S + L1)/L1 · x?** Ele funciona como primeira ideia, mas desloca os nós da corda até S em relação à rotação rígida. O interior do furo gira rígido, então a primeira camada de triângulos junto da parede cisalha na mesma ordem de S. O mapa com buffer tira esse cisalhamento da corda e joga o esticamento no tubo, onde há espaço.

**Escolha de b:** o espaçamento nos arcos cresce por (b + S_max)/b. Com b ≈ 0,7·L1 a 1,0·L1 e fechamento de ~65°, fica entre 1,5 e 2,5 vezes h. Um b maior deixa o arco mais fino, mas movimenta mais nós do tubo.

## 5. Junções tubo/arco (P(φi) fixos)

Em θ = 0, cada arco tem comprimento zero e a junção coincide com a quina do furo. Nenhum nó consegue ficar travado na junção de forma contínua, porque ele teria que "nascer" da quina. Há duas opções:

1. **Não encaixar** (padrão): a junção é cortada por no máximo meio espaçamento. Dá a melhor qualidade de malha.
2. **Encaixar** (`--encaixar`): o nó mais próximo pula para a junção quando está a menos de meio espaçamento. A geometria fica exata, mas cria saltos de até h/2 (pico de velocidade de malha) e piora um pouco a qualidade.

Se a quina da sede for importante para o escoamento, a alternativa limpa é inserir um nó na junção quando S > h (muda o número de nós; ver seção 10).

## 6. Movimento do interior

A cada passo Δθ = θ − θ_ant, o deslocamento é incremental:

1. **Contorno (Dirichlet):** d = posição do mapa em θ − posição atual.
2. **Núcleo rígido (Dirichlet):** nós internos com |p| < f·r (f ≈ 0,75) giram de Δθ com a equação do interior da válvula.
3. **Resto:** laplaciano do deslocamento, K·d = 0, com pesos de aresta w_ij = 1/|x_i − x_j|. É um sistema esparso por componente.

O núcleo rígido preserva a malha do furo intacta. O laplaciano distribui o cisalhamento entre a esfera que gira e o tubo parado.

Se um nó interno sair do domínio ou chegar a menos de `folga_min·h` da parede, o passo dele é recuado (1/2, 1/4, 1/8, 0). Se nem a posição antiga servir, porque a parede passou por cima, ele vai para a média dos vizinhos.

## 7. Checagens de qualidade

Por triângulo:

- **área com sinal** A = ½·[(b − a) × (c − a)]; A ≤ 0 indica triângulo invertido;
- **qualidade** q = 4√3·A / (l1² + l2² + l3²); q = 1 é equilátero, q → 0 é degenerado e q < 0 é invertido.

Por nó interno:

- **folga** até o polígono de contorno, positiva dentro e negativa fora, calculada com KD-tree nos vértices de contorno.

Gatilho de reparo: `q_min < 0.30` ou algum nó fora ou colado na parede.

## 8. Reparo: retriangulação com os mesmos nós

Quando o gatilho dispara:

1. **Retriangular** todos os nós atuais.
   - Com o pacote `triangle` (`pip install triangle`), use Delaunay restrita no anel de contorno com a opção `'pY'`. As arestas de contorno são respeitadas e nenhum nó novo é criado. **É o recomendado.**
   - Sem ele, use Delaunay comum e descarte os triângulos com centróide fora do polígono. Nos testes, perdeu 2 a 6 arestas de contorno perto dos arcos, então sirva-se dele apenas como fallback.
2. **Suavização laplaciana protegida** dos nós internos fora do núcleo, 20 iterações de Jacobi com relaxação 0,5. O movimento é desfeito se o nó se aproxima da parede abaixo da folga ou se algum triângulo vizinho piora abaixo de 0,5.
3. **Retriangular de novo** para a conectividade final.

O número de nós é constante durante todo o movimento. Só a conectividade muda, e a velocidade de malha de cada nó fica sempre bem definida.

## 9. Integração com o solver CFD (ALE)

- **Velocidade de malha:** w_i = (x_i^{n+1} − x_i^n)/Δt. Use os fluxos convectivos relativos (u − w)·n.
- **Lei de conservação geométrica (GCL):** a variação de área de cada volume de controle tem que bater com o fluxo de w nas faces. Calcule os fluxos de malha a partir das áreas varridas pelas faces, não de w avaliado no centro.
- **Passos com retriangulação:**
  - Solver nodal (FEM, CVFEM, vertex-centered): os valores nos nós continuam válidos. Só reconstrua matrizes e volumes duais.
  - Solver cell-centered: é preciso remapear conservativamente as médias de célula da malha velha para a nova (interseção de triângulos ou sobreposição via supermesh). Limite o número de retriangulações se isso for caro.
- **Relação entre Δθ e Δt:** o deslocamento máximo por passo é r·Δθ. Mantenha r·Δθ < 0,2·h para o laplaciano não inverter elementos antes do reparo e para o CFL de malha ficar controlado.
- **Condições de contorno:** nos arcos do tipo esfera (φ2 e φ4) a parede se move com velocidade tangencial ω·r (u = ω × p). Nos arcos de cavidade (φ1 e φ3) a parede é o corpo, parado. Na corda, a parede gira: u = ω × p.

## 10. Resultados de teste e limites

Com H = 0,5, L1 = 1,4, Lp = 3,0, h = 0,05, 2541 nós e 120 passos até 90% do fechamento (64°):

- nenhum triângulo invertido, nenhum nó fora do domínio e nenhuma aresta de contorno perdida (modo `triangle`);
- q_min ≥ 0,20 em todos os passos, com q médio em torno de 0,93;
- 35 retriangulações em 120 passos;
- fechar e reabrir volta a uma malha válida (q_min final 0,56, q médio 0,93).

Com H = 0,5, L1 = 1,0 e h = 0,03, q_min ficou entre 0,19 e 0,25, dependendo do núcleo e da junção.

Limites conhecidos:

- **Perto do fechamento** (fresta < ~10h) os nós se acumulam na fresta e aparecem elementos muito pequenos. A solução é remover nós na fresta e inserir nos arcos (malha adaptativa, número de nós variável). Outra opção é parar a simulação num ângulo mínimo de abertura.
- **Arcos grossos:** o espaçamento nos arcos é (b + S)/b vezes h. Para resolver camada limite na sede, insira nós nos arcos conforme S cresce.
- **Alternativas para rotações grandes:** malha deslizante (interface não conforme no círculo, com as duas partes rígidas) ou malha sobreposta (overset). Os arcos da seção 2 são justamente as partes não sobrepostas da interface deslizante.

## 11. Checklist para reaproveitar em outro projeto

1. Escrever cada parede móvel como caminho de peças (segmento e arco), com comprimento e avaliação por s.
2. Guardar s0 de referência para cada nó de parede.
3. Definir o mapa s0 → s(θ) com nós nas quinas que giram rígidas e faixas buffer que absorvem o crescimento.
4. Marcar os nós do núcleo rígido e resolver o laplaciano do deslocamento no resto.
5. Checar área com sinal, qualidade e folga ao contorno a cada passo.
6. Retriangular com Delaunay restrita (mesmos nós) e suavizar com proteção quando o gatilho disparar.
7. Passar ao solver as posições, a conectividade, a velocidade de malha e a flag "conectividade mudou".

## 12. Referência rápida do script

| Função | O que faz |
|---|---|
| `Geometria(H, L1, Lp)` | r, φ1 a φ4, fechamento, `pecas(parede, θ)`, `avaliar`, `dentro` |
| `malha_inicial(geo, h)` | retângulo estruturado em θ = 0, com nós em x = ±L1/2 |
| `mapa_arco(geo, s0, θ, b)` | mapa s0 → s(θ) da seção 4 |
| `nos_parede(geo, M, parede, θ, b)` | posições dos nós de uma parede em θ |
| `mover_interior(...)` | núcleo rígido e laplaciano incremental |
| `qualidade(xy, tri)` | q com sinal |
| `retriangular(M, xy)` | Delaunay restrita (ou fallback) com os mesmos nós |
| `suavizar(...)` | laplaciano protegido |
| `passo(geo, M, θ_ant, θ, cfg)` | um passo completo, devolve o diagnóstico |
| `simular(geo, cfg, θ_max, n)` | laço de θ = 0 até θ_max |
| `exportar(M, arquivo)` | salva xy, tri e marcadores em .npz |

Linha de comando:

```
python valvula_malha_movel.py --H 0.5 --L1 1.4 --Lp 3.0 --h 0.05 --frac 0.9 --passos 120
    [--buffer 1.0] [--nucleo 0.75] [--qmin 0.3] [--encaixar] [--sem-triangle] [--exportar]
```

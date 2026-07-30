# Planilha de benchmark (`femns.report`)

Se o config tiver a chave `benchmark_xlsx`, `scripts/run_simulation.py`
acrescenta, ao final da simulação, **1 linha** nesse arquivo `.xlsx`
(`femns.report.salvar_resumo`) — sem sobrescrever runs anteriores. A ideia é
comparar `dt`/malha/`Re`/`advection` lado a lado numa tabela só, não analisar
um único run em detalhe (não há série temporal por iteração, só valores
agregados — ver seção final sobre essa limitação).

Cada coluna é explicada abaixo, agrupada pelo que ela mede.

## Identificação do run

Não são métricas — servem só pra saber o que gerou aquela linha.

| Coluna | O que é |
|---|---|
| `timestamp` | Data/hora (local) em que a simulação terminou e a linha foi salva. |
| `mesh` | Caminho do `.msh` usado (`cfg["mesh"]`). |
| `advection` | `explicit` ou `semi_lagrangian` (`cfg["simulation"]["advection"]`) — qual formulação do termo convectivo foi usada nesse run. |
| `dt` | Passo de tempo. |
| `reynolds` | Número de Reynolds da simulação. |
| `iterations` | Quantas iterações de tempo foram rodadas. |
| `npoints` | Número de nós "reais" da malha (vértices, sem contar o centroide do elemento MINI). |
| `ne` | Número de elementos (triângulos) da malha. |
| `device` | `cpu` ou `cuda` — em qual dispositivo rodou. |

## Desempenho

| Coluna | O que é |
|---|---|
| `tempo_total_s` | Tempo de ponta a ponta (leitura de malha + assembly + todas as iterações). |
| `tempo_assembly_s` | Tempo até a matriz global `A` estar pronta (leitura de malha, montagem de `K,M,Gx,Gy,Gvx,Gvy`, `build_system_matrix`) — fora do loop de tempo, pago uma vez só. |
| `tempo_medio_por_iter_s` | `(tempo_total_s - tempo_assembly_s) / iterations` — custo médio de 1 passo de tempo (advecção + solve do BiCGSTAB + escrita do `.vtk`). |

## Convergência do BiCGSTAB

`solver.time_step` resolve o sistema global a cada passo com
`femns.krylov.bicgstab` (`tol=1e-8`, `max_iter=2000`, ambos fixos por
padrão — ver limitação no README). Essas colunas resumem como esse solve se
comportou ao longo dos `iterations` passos.

| Coluna | O que é |
|---|---|
| `bicg_iters_media` | Média de `info["iters"]` (iterações do BiCGSTAB) entre todos os passos. |
| `bicg_iters_max` | Máximo de `info["iters"]` num único passo — tende a ser bem maior nos primeiros passos (antes do chute inicial "esquentar", ver README, seção "Warm-start entre iterações") e cair conforme a simulação avança. |
| `bicg_residual_media` | Média do resíduo relativo final (`info["residual"]`, `‖b - A x‖ / ‖b‖`) entre os passos. |
| `bicg_residual_max` | Maior resíduo final entre os passos — se muito acima de `1e-8`, aponta pra `passos_nao_convergidos` (coluna seguinte). |
| `passos_nao_convergidos` | Quantos passos terminaram com resíduo acima de `1e-8` (o `tol` padrão de `solver.time_step`). **Importante**: isso não significa necessariamente que o BiCGSTAB rodou os `2000` iterações completas — ele também para mais cedo se disparar uma das 3 condições de "breakdown" do algoritmo (`rho_new`, `r_hat_v` ou `t_dot_t` chegando a zero, ver `src/femns/krylov.py:85,94,108`), devolvendo o que tinha até ali mesmo sem atingir `tol`. A planilha não distingue os dois casos (falta de iteração vs. breakdown) — só sinaliza que aquele passo não bateu a tolerância pedida. |

## Qualidade física da solução

Calculadas a cada passo (a partir de `vx, vy, p` já com contorno aplicado) e
agregadas por média/máximo ao final.

| Coluna | O que é |
|---|---|
| `vel_l2_media` | Média (entre os passos) da norma L2 do campo de velocidade nos nós de vértice: `sqrt(mean(vx² + vy²))`. |
| `vel_l2_max` | Maior norma L2 de velocidade observada num único passo. |
| `vel_linf_max` | Maior valor pontual de `sqrt(vx² + vy²)` (norma L∞) entre todos os nós e todos os passos — um pico aqui bem acima da escala física esperada (ex.: `vx_inlet`) é sinal de instabilidade/blow-up. |
| `divergencia_l2_media` | Média (entre os passos) da norma L2 de `Dx@vx + Dy@vy` (`Dx = -Gx^T`, `Dy = -Gy^T`, **sem** o fator `beta` — que é só escala numérica do solver, ver `solver.pressure_scale_factor`; a divergência física não deve levar `beta`). Como o escoamento é incompressível, isso deveria ficar bem próximo de zero; é o indicador mais direto de erro numérico real introduzido pela advecção (distinto de difusão numérica, que não quebra a incompressibilidade). |
| `divergencia_l2_max` | Maior norma L2 de divergência observada num único passo — uma tendência de crescimento ao longo da simulação (mesmo que a média pareça pequena) é sinal de instabilidade se acumulando. |
| `pressao_min` | Pressão mínima nos nós, **no último passo** (não agregada — é só o estado final). |
| `pressao_max` | Pressão máxima nos nós, no último passo. |
| `pressao_media` | Pressão média nos nós, no último passo. |

## Limitações desse resumo

- **Só agregado, sem série temporal**: se `bicg_iters_max`/`divergencia_l2_max`
  chamar atenção, a planilha não diz *em qual* passo aconteceu — pra isso,
  seria preciso adicionar uma aba por-iteração (não implementada; ver
  conversa de design original, decisão foi começar só com o resumo).
- **`passos_nao_convergidos` não diz o motivo** (falta de iteração vs.
  breakdown do BiCGSTAB) — só que o resíduo daquele passo ficou acima de
  `tol`.
- **`pressao_*` é só do último passo**, não agregada como as outras métricas
  físicas — útil pra ver o estado final, não a evolução.
- Calcular essas métricas tem um custo extra pequeno (2 produtos
  matriz-vetor esparsos por passo, pra divergência) — só é pago quando
  `benchmark_xlsx` está presente no config (ver `scripts/run_simulation.py`).

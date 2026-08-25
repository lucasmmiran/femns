"use strict";

// ---------------------------------------------------------------------
// tema (claro/escuro) -- o atributo data-theme ja foi aplicado inline no
// <head> (evita flash do tema errado); aqui so sincroniza o rotulo do
// botao e liga o clique.
// ---------------------------------------------------------------------

(function initTheme() {
  const KEY = "femns-theme";
  const btn = document.getElementById("theme-toggle");

  function apply(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    if (btn) btn.textContent = theme === "light" ? "☾ escuro" : "☀ claro";
  }

  apply(document.documentElement.getAttribute("data-theme") || "dark");

  btn?.addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
    localStorage.setItem(KEY, next);
    apply(next);
  });
})();

// ---------------------------------------------------------------------
// utilidades
// ---------------------------------------------------------------------

async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children || []) node.appendChild(child);
  return node;
}

// Exporta como download do navegador -- nao ha endpoint de servidor pra
// plotagens salvas, o arquivo .json em si e' o "banco de dados" (o usuario
// guarda onde quiser e recarrega depois via <input type=file>).
function downloadJSON(filename, obj) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function slugify(s) {
  return (
    s
      .toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/(^-|-$)/g, "") || "plot"
  );
}

// Paleta fixa (nao depende de contagem de contornos da malha) usada tanto
// no preview de malha (contorno colorido por nome) quanto na legenda --
// mesma cor em qualquer canvas que desenhe o mesmo nome.
const BOUNDARY_PALETTE = ["#ff6b6b", "#4f8cff", "#4caf7d", "#f5a623", "#c77dff", "#2dd4bf", "#f472b6", "#a3e635"];

function boundaryColor(nome, nomesOrdenados) {
  return BOUNDARY_PALETTE[nomesOrdenados.indexOf(nome) % BOUNDARY_PALETTE.length];
}

// Desenha o wireframe de uma malha (triangulos + arestas de contorno
// coloridas por cima) num canvas -- usado tanto no preview pequeno quanto
// no modal ampliado, mesma funcao pros dois. Retorna os nomes de contorno
// (ordenados, mesma ordem = mesma cor) pra montar a legenda em seguida.
function renderMeshWireframe(canvas, geometry) {
  const ctx = canvas.getContext("2d");
  const xs = geometry.points.map((p) => p[0]);
  const ys = geometry.points.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const margin = 16;
  const w = canvas.width - 2 * margin;
  const h = canvas.height - 2 * margin;
  const scale = Math.min(w / (x1 - x0 || 1), h / (y1 - y0 || 1));
  const offX = margin + (w - scale * (x1 - x0)) / 2;
  const offY = margin + (h - scale * (y1 - y0)) / 2;
  const toPx = (x, y) => [offX + (x - x0) * scale, canvas.height - (offY + (y - y0) * scale)];

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(255,255,255,0.25)";
  ctx.lineWidth = 0.6;
  for (const tri of geometry.triangles) {
    const [a, b, c] = tri;
    const [ax, ay] = toPx(...geometry.points[a]);
    const [bx, by] = toPx(...geometry.points[b]);
    const [cx, cy] = toPx(...geometry.points[c]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.lineTo(cx, cy);
    ctx.closePath();
    ctx.stroke();
  }

  const nomes = Object.keys(geometry.boundaries).sort();
  ctx.lineWidth = 3;
  for (const nome of nomes) {
    ctx.strokeStyle = boundaryColor(nome, nomes);
    for (const [i, j] of geometry.boundaries[nome]) {
      const [ax, ay] = toPx(...geometry.points[i]);
      const [bx, by] = toPx(...geometry.points[j]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
  }

  return nomes;
}

function renderMeshLegend(container, nomes) {
  container.innerHTML = "";
  for (const nome of nomes) {
    container.appendChild(el("span", { class: "legend-item" }, [
      el("span", { class: "legend-swatch", style: `background:${boundaryColor(nome, nomes)}` }),
      el("span", { text: nome }),
    ]));
  }
}

// ---------------------------------------------------------------------
// abas
// ---------------------------------------------------------------------

for (const btn of document.querySelectorAll(".tab-btn")) {
  btn.addEventListener("click", () => {
    for (const b of document.querySelectorAll(".tab-btn")) b.classList.remove("active");
    for (const s of document.querySelectorAll(".screen")) s.classList.remove("active");
    btn.classList.add("active");
    document.getElementById(`screen-${btn.dataset.screen}`).classList.add("active");
    if (btn.dataset.screen === "resultados") Resultados.onShow();
  });
}

// ===========================================================================
// TELA: NOVA SIMULAÇÃO
// ===========================================================================

const NovaSimulacao = (() => {
  let meshes = [];
  let templateItems = []; // {kind: "curated"|"saved", name, slug?, content}
  let boundaryOrder = []; // nomes na ordem de prioridade (baixa -> alta)
  let jobsTimer = null;

  const meshGeometryCache = new Map(); // nome da malha -> geometria (points/triangles/boundaries)
  let currentMeshGeometry = null;

  function currentMesh() {
    const name = document.getElementById("f-mesh").value;
    return meshes.find((m) => m.name === name);
  }

  // -- preview da malha (canvas pequeno + botao de ampliar) -------------

  async function loadMeshGeometry(meshName) {
    if (meshGeometryCache.has(meshName)) return meshGeometryCache.get(meshName);
    const geometry = await api(`/api/meshes/${encodeURIComponent(meshName)}/geometry`);
    meshGeometryCache.set(meshName, geometry);
    return geometry;
  }

  async function refreshMeshPreview() {
    const mesh = currentMesh();
    if (!mesh) return;
    currentMeshGeometry = await loadMeshGeometry(mesh.name);
    const nomes = renderMeshWireframe(document.getElementById("mesh-preview-canvas"), currentMeshGeometry);
    renderMeshLegend(document.getElementById("mesh-preview-legend"), nomes);
  }

  function closeMeshModal() {
    document.getElementById("mesh-modal").hidden = true;
  }

  function bindMeshPreviewControls() {
    document.getElementById("mesh-expand-btn").addEventListener("click", () => {
      if (!currentMeshGeometry) return;
      document.getElementById("mesh-modal").hidden = false;
      const nomes = renderMeshWireframe(document.getElementById("mesh-modal-canvas"), currentMeshGeometry);
      renderMeshLegend(document.getElementById("mesh-modal-legend"), nomes);
    });
    document.getElementById("mesh-modal-close").addEventListener("click", closeMeshModal);
    document.getElementById("mesh-modal").addEventListener("click", (ev) => {
      if (ev.target.id === "mesh-modal") closeMeshModal(); // clique no fundo, nao na caixa
    });
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") closeMeshModal();
    });
  }

  // -- condicoes de contorno ---------------------------------------------

  function renderBoundaryList() {
    const ul = document.getElementById("boundary-list");
    ul.innerHTML = "";
    boundaryOrder.forEach((name, idx) => {
      const row = el("li", { class: "boundary-row", "data-name": name }, [
        el("span", { class: "name", text: name }),
        el("div", { class: "order-btns" }, [
          el("button", { type: "button", title: "subir prioridade", onclick: () => moveBoundary(idx, -1), text: "▲" }),
          el("button", { type: "button", title: "descer prioridade", onclick: () => moveBoundary(idx, 1), text: "▼" }),
        ]),
        comp(name, "vx"), comp(name, "vy"), comp(name, "p"),
      ]);
      ul.appendChild(row);
    });
  }

  // Cada componente (vx/vy/p) de um contorno pode ser um valor constante
  // (input numerico normal) ou uma expressao Python de y absoluto (ex.
  // "4*y*(1-y)", ver boundary.funcao_de_y) -- o botao fn-toggle alterna
  // entre os dois. O modo atual fica em `label.dataset.mode` ("const",
  // default, ou "func"); `.comp-value` identifica o input de valor
  // independente do `type` atual (que muda entre number/text).
  function comp(boundaryName, campo) {
    const cb = el("input", { type: "checkbox", "data-comp": campo });
    const num = el("input", {
      type: "number", step: "any", value: "0", disabled: "disabled", class: "comp-value",
    });
    const fnBtn = el("button", {
      type: "button", class: "fn-toggle", disabled: "disabled",
      title: "Alternar entre valor constante e função de y", text: "ƒ(y)",
    });
    const label = el("label", { class: "comp" }, [cb, el("span", { text: campo }), num, fnBtn]);
    cb.addEventListener("change", () => {
      num.disabled = !cb.checked;
      fnBtn.disabled = !cb.checked;
    });
    fnBtn.addEventListener("click", () => setCompMode(label, num, label.dataset.mode !== "func"));
    return label;
  }

  function setCompMode(label, num, isFunc) {
    label.dataset.mode = isFunc ? "func" : "const";
    if (isFunc) {
      num.type = "text";
      num.placeholder = "ex.: 4*y*(1-y)";
      if (num.value === "0") num.value = "";
    } else {
      num.type = "number";
      num.placeholder = "";
      if (num.value.trim() === "") num.value = "0";
    }
  }

  function moveBoundary(idx, delta) {
    const j = idx + delta;
    if (j < 0 || j >= boundaryOrder.length) return;
    [boundaryOrder[idx], boundaryOrder[j]] = [boundaryOrder[j], boundaryOrder[idx]];
    renderBoundaryList();
  }

  function applyBoundaryConditions(conditions, priority) {
    if (priority && priority.length) {
      // reordena boundaryOrder pra bater com a prioridade do template, mantendo
      // qualquer contorno da malha que o template nao cite no fim da lista.
      const resto = boundaryOrder.filter((n) => !priority.includes(n));
      boundaryOrder = [...priority.filter((n) => boundaryOrder.includes(n)), ...resto];
      renderBoundaryList();
    }
    for (const [nome, valores] of Object.entries(conditions || {})) {
      const row = document.querySelector(`.boundary-row[data-name="${CSS.escape(nome)}"]`);
      if (!row) continue;
      for (const campo of ["vx", "vy", "p"]) {
        const valor = valores[campo];
        if (valor == null) continue;
        const label = [...row.querySelectorAll(".comp")].find((l) => l.querySelector("span").textContent === campo);
        const cb = label.querySelector("input[type=checkbox]");
        const fnBtn = label.querySelector(".fn-toggle");
        const num = label.querySelector(".comp-value");
        if (typeof valor === "number") {
          setCompMode(label, num, false);
          num.value = valor;
        } else if (valor && typeof valor === "object" && typeof valor.funcao === "string") {
          setCompMode(label, num, true);
          num.value = valor.funcao;
        } else {
          // outra forma de dict (ex. {perfil: parabolico, vmax: ...}) nao
          // tem editor aqui ainda -- pula em vez de jogar "[object Object]"
          // no campo de valor.
          continue;
        }
        cb.checked = true;
        num.disabled = false;
        fnBtn.disabled = false;
      }
    }
  }

  function collectBoundaryConditions() {
    const conditions = {};
    for (const row of document.querySelectorAll(".boundary-row")) {
      const nome = row.dataset.name;
      const valores = {};
      for (const label of row.querySelectorAll(".comp")) {
        const campo = label.querySelector("span").textContent;
        const cb = label.querySelector("input[type=checkbox]");
        const num = label.querySelector(".comp-value");
        if (!cb.checked) continue;
        if (label.dataset.mode === "func") {
          const expr = num.value.trim();
          if (expr) valores[campo] = { funcao: expr };
        } else {
          valores[campo] = parseFloat(num.value);
        }
      }
      if (Object.keys(valores).length) conditions[nome] = valores;
    }
    return conditions;
  }

  // -- malha selecionada: contorno + preview + template automatico ------

  // So o essencial de reagir a malha selecionada (contorno + preview) --
  // sem o auto-apply de template abaixo. Usado quando a troca de malha e'
  // efeito colateral de carregar um template (`switchMeshIfKnown`), onde
  // reaplicar um template *diferente* do que o usuario acabou de escolher
  // seria surpreendente (bug corrigido: selecionar uma configuracao salva
  // cujo mesh bate com o de um template do projeto sobrescrevia a escolha
  // do usuario de volta pro template do projeto).
  function refreshMeshDependentUI() {
    const mesh = currentMesh();
    boundaryOrder = mesh ? [...mesh.boundary_names] : [];
    renderBoundaryList();
    refreshMeshPreview().catch((e) => console.error("erro ao carregar visualização da malha:", e));
  }

  // Disparado pelo <select> de malha em si (interacao direta do usuario) --
  // aqui sim faz sentido sugerir automaticamente o template do projeto pra
  // essa malha.
  function onMeshChange() {
    refreshMeshDependentUI();
    const mesh = currentMesh();
    // aplica automaticamente o primeiro template *do projeto* cujo mesh bate
    // (configuracoes salvas pelo usuario nao entram nessa selecao automatica)
    const tpl = templateItems.find((t) => t.kind === "curated" && t.content.mesh && t.content.mesh.endsWith("/" + mesh?.name));
    if (tpl) {
      applyConfig(tpl.content);
      document.getElementById("f-template").value = templateValue(tpl);
      updateTemplateDeleteButton();
    }
  }

  // Preenche o formulario inteiro a partir de um config (template do
  // projeto ou configuracao salva, mesmo formato de `buildConfig()`) --
  // sempre "load" completo, nao merge: campos ausentes no config (ex.
  // moving_point/mesh_motion quando o preset nao usa) voltam pro estado
  // desligado em vez de manter o que estava preenchido antes.
  function applyConfig(c) {
    const sim = c.simulation || {};
    if (sim.dt != null) document.getElementById("f-dt").value = sim.dt;
    if (sim.iterations != null) document.getElementById("f-iterations").value = sim.iterations;
    if (sim.reynolds != null) document.getElementById("f-reynolds").value = sim.reynolds;
    // "explicit" e o nome antigo de "eulerian" (ver run_simulation.py) --
    // normaliza aqui pra templates salvos antes da renomeacao continuarem
    // carregando certo em vez de deixar o select sem nenhuma opcao valida.
    document.getElementById("f-advection").value = sim.advection === "explicit" ? "eulerian" : sim.advection || "eulerian";
    document.getElementById("f-element").value = sim.element || "mini";
    document.getElementById("f-sl-boundary").value = sim.sl_boundary || "intercept";
    updateAdvectionUI();

    const boundary = c.boundary || {};
    applyBoundaryConditions(boundary.conditions, boundary.priority);

    const mp = c.moving_point || {};
    document.getElementById("mp-enabled").checked = !!mp.enabled;
    document.getElementById("mp-node").value = mp.node_index ?? 0;
    document.getElementById("mp-amp").value = mp.amplitude_factor ?? 0.1;
    document.getElementById("mp-omega").value = mp.omega ?? 628.3185307179587;

    const mm = c.mesh_motion || {};
    document.getElementById("mm-enabled").checked = !!mm.enabled;
    document.getElementById("mm-amp").value = mm.amplitude_factor ?? 0.3;
    document.getElementById("mm-omega").value = mm.omega ?? 31.41593;
    document.getElementById("mm-seed").value = mm.seed ?? 0;
  }

  // Troca a malha selecionada se `meshPath` (ex. "meshes/degrau.msh") for
  // conhecida -- compartilhado entre "carregar template" e "carregar
  // configuracao salva".
  function switchMeshIfKnown(meshPath) {
    const meshName = (meshPath || "").split("/").pop();
    if (meshName && meshes.some((m) => m.name === meshName)) {
      document.getElementById("f-mesh").value = meshName;
      refreshMeshDependentUI();
    }
  }

  function updateAdvectionUI() {
    const advection = document.getElementById("f-advection").value;
    document.getElementById("f-sl-boundary-wrap").style.display = advection === "semi_lagrangian" ? "" : "none";
  }

  // -- template (unifica templates do projeto + configuracoes salvas) ---

  function templateValue(item) {
    return item.kind === "curated" ? `curated:${item.name}` : `saved:${item.slug}`;
  }

  function findTemplate(value) {
    return templateItems.find((t) => templateValue(t) === value);
  }

  function updateTemplateDeleteButton() {
    const sel = document.getElementById("f-template");
    const item = findTemplate(sel.value);
    document.getElementById("f-template-delete").disabled = !item || item.kind !== "saved";
  }

  async function loadTemplates() {
    const [curated, saved] = await Promise.all([api("/api/configs"), api("/api/saved-configs")]);
    templateItems = [
      ...curated.map((t) => ({ kind: "curated", name: t.name, content: t.content })),
      ...saved.map((s) => ({ kind: "saved", name: s.name, slug: s.slug, content: s.content })),
    ];

    const sel = document.getElementById("f-template");
    const prev = sel.value;
    sel.innerHTML = "";
    sel.appendChild(el("option", { value: "", text: "— nenhum —" }));

    const grupoProjeto = el("optgroup", { label: "Templates do projeto" });
    const grupoSalvos = el("optgroup", { label: "Configurações salvas" });
    for (const item of templateItems) {
      const opt = el("option", { value: templateValue(item), text: item.name });
      (item.kind === "curated" ? grupoProjeto : grupoSalvos).appendChild(opt);
    }
    if (grupoProjeto.children.length) sel.appendChild(grupoProjeto);
    if (grupoSalvos.children.length) sel.appendChild(grupoSalvos);

    if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
    updateTemplateDeleteButton();
  }

  function bindTemplateControls() {
    const sel = document.getElementById("f-template");
    sel.addEventListener("change", () => {
      updateTemplateDeleteButton();
      const item = findTemplate(sel.value);
      if (!item) return;
      switchMeshIfKnown(item.content.mesh);
      document.getElementById("f-label").value = item.name;
      applyConfig(item.content);
    });

    document.getElementById("f-template-delete").addEventListener("click", async () => {
      const item = findTemplate(sel.value);
      if (!item || item.kind !== "saved") return;
      if (!window.confirm(`Excluir o template salvo "${item.name}"?`)) return;
      await api(`/api/saved-configs/${item.slug}`, { method: "DELETE" });
      sel.value = "";
      await loadTemplates();
    });

    document.getElementById("f-save-config").addEventListener("click", async () => {
      const errorBox = document.getElementById("form-error");
      const okBox = document.getElementById("save-ok");
      errorBox.hidden = true;
      okBox.hidden = true;
      const nome = document.getElementById("f-label").value.trim();
      if (!nome) {
        errorBox.textContent = "Preencha o Rótulo antes de salvar.";
        errorBox.hidden = false;
        return;
      }
      try {
        const res = await api("/api/saved-configs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: nome, config: buildConfig() }),
        });
        await loadTemplates();
        sel.value = `saved:${res.slug}`;
        updateTemplateDeleteButton();
        okBox.textContent = `Template "${nome}" salvo.`;
        okBox.hidden = false;
      } catch (e) {
        errorBox.textContent = e.message;
        errorBox.hidden = false;
      }
    });
  }

  // -- montagem do config / envio ----------------------------------------

  function buildConfig() {
    const mesh = currentMesh();
    const cfg = {
      mesh: mesh.name,
      simulation: {
        dt: parseFloat(document.getElementById("f-dt").value),
        iterations: parseInt(document.getElementById("f-iterations").value, 10),
        reynolds: parseFloat(document.getElementById("f-reynolds").value),
        advection: document.getElementById("f-advection").value,
        element: document.getElementById("f-element").value,
        sl_boundary: document.getElementById("f-sl-boundary").value,
      },
      boundary: {
        priority: boundaryOrder,
        conditions: collectBoundaryConditions(),
      },
    };

    if (document.getElementById("mp-enabled").checked) {
      cfg.moving_point = {
        enabled: true,
        node_index: parseInt(document.getElementById("mp-node").value, 10),
        amplitude_factor: parseFloat(document.getElementById("mp-amp").value),
        omega: parseFloat(document.getElementById("mp-omega").value),
      };
    }
    if (document.getElementById("mm-enabled").checked) {
      cfg.mesh_motion = {
        enabled: true,
        amplitude_factor: parseFloat(document.getElementById("mm-amp").value),
        omega: parseFloat(document.getElementById("mm-omega").value),
        seed: parseInt(document.getElementById("mm-seed").value, 10),
      };
    }
    return cfg;
  }

  async function onSubmit(ev) {
    ev.preventDefault();
    const errorBox = document.getElementById("form-error");
    errorBox.hidden = true;
    document.getElementById("save-ok").hidden = true;
    try {
      const config = buildConfig();
      const label = document.getElementById("f-label").value.trim();
      await api("/api/simulations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config, label: label || undefined }),
      });
      refreshJobs();
    } catch (e) {
      errorBox.textContent = e.message;
      errorBox.hidden = false;
    }
  }

  // -- lista de simulacoes lancadas ---------------------------------------

  function jobCard(job) {
    const pct = job.iterations ? Math.min(100, Math.round((100 * job.frames_done) / job.iterations)) : 0;
    const card = el("div", { class: "job-card" }, [
      el("div", { class: "job-head" }, [
        el("span", { class: "job-title", text: job.label }),
        el("span", { class: `status ${job.status}`, text: job.status }),
      ]),
      el("div", { class: "bar" }, [el("div", { class: "bar-fill", style: `width:${pct}%` })]),
      el("div", { text: `${job.frames_done} / ${job.iterations} quadros` }),
      el("pre", { text: job.log_tail || "" }),
    ]);
    if (job.status === "running") {
      const cancelBtn = el("button", {
        text: "Cancelar",
        onclick: async () => { await api(`/api/simulations/${job.id}/cancel`, { method: "POST" }); refreshJobs(); },
      });
      card.appendChild(el("div", { class: "job-actions" }, [cancelBtn]));
    }
    return card;
  }

  async function refreshJobs() {
    const jobs = await api("/api/simulations");
    const list = document.getElementById("jobs-list");
    list.innerHTML = "";
    if (!jobs.length) {
      list.appendChild(el("p", { class: "hint", text: "Nenhuma simulação lançada ainda." }));
      return;
    }
    for (const job of jobs) list.appendChild(jobCard(job));
  }

  // -- inicializacao -------------------------------------------------------

  async function loadMeshes() {
    meshes = await api("/api/meshes");
    const meshSel = document.getElementById("f-mesh");
    meshSel.innerHTML = "";
    for (const m of meshes) meshSel.appendChild(el("option", { value: m.name, text: `${m.name} (${m.npoints} pts)` }));
    meshSel.addEventListener("change", onMeshChange);
  }

  async function loadAll() {
    await Promise.all([loadMeshes(), loadTemplates()]);
    onMeshChange();
  }

  function init() {
    document.getElementById("f-advection").addEventListener("change", updateAdvectionUI);
    document.getElementById("f-element").addEventListener("change", updateAdvectionUI);
    document.getElementById("form-nova").addEventListener("submit", onSubmit);
    bindTemplateControls();
    bindMeshPreviewControls();
    loadAll().then(updateAdvectionUI);
    refreshJobs();
    jobsTimer = setInterval(refreshJobs, 2000);
  }

  return { init };
})();

// ===========================================================================
// TELA: RESULTADOS
// ===========================================================================

const Resultados = (() => {
  const canvas = document.getElementById("r-canvas");
  const ctx = canvas.getContext("2d");
  const frameCache = new Map(); // n -> frame data
  const MAX_CACHE = 60;

  let runs = [];
  let currentRun = null; // {id, meta}
  let frameNumbers = [];
  let frameIdx = 0; // indice dentro de frameNumbers
  let playing = false;
  let playTimer = null;
  let loadedForRun = null;

  // -- grafico sobre linha (plot over line) -------------------------------
  let line = null; // {x1,y1,x2,y2} em coordenadas fisicas, ou null
  let drawingLine = false;
  let drawFirstPoint = null; // primeiro clique, enquanto aguarda o segundo
  let currentTransform = null; // {x0,y0,scale,offX,offY,zoom,panX,panY} do ultimo render(), pra converter clique -> mundo
  // zoom (scroll) / pan (arrastar com o botao direito) sobre o campo --
  // aplicado por cima do ajuste automatico de bbox, nao no lugar dele.
  let viewState = { zoom: 1, panX: 0, panY: 0 };
  let savedOverlays = []; // plotagens carregadas de arquivo: {name, campo, color, dash, samples, length}
  const OVERLAY_PALETTE = ["#f5a623", "#4caf7d", "#c77dff", "#f472b6", "#2dd4bf", "#a3e635", "#ff6b6b"];
  // estilo (cor + tracado) da linha "ao vivo" (a que segue o quadro atual,
  // em oposicao as plotagens salvas carregadas de arquivo em savedOverlays).
  let liveLineStyle = { color: "#4f8cff", dash: "solid" };
  const LINE_DASH_STYLES = {
    solid: { label: "sólida", pattern: [] },
    dashed: { label: "tracejada", pattern: [8, 4] },
    dotted: { label: "pontilhada", pattern: [2, 3] },
    dashdot: { label: "traço-ponto", pattern: [8, 3, 2, 3] },
  };

  function jetColor(t) {
    t = Math.min(1, Math.max(0, t));
    const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)));
    const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)));
    const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)));
    return [Math.round(255 * r), Math.round(255 * g), Math.round(255 * b)];
  }

  // Aproximacoes por poucos pontos de controle (interpolacao linear em RGB) --
  // suficiente pra uma legenda de GUI, nao pretende reproduzir bit-a-bit os
  // LUTs originais do ParaView/matplotlib.
  const COOLWARM_STOPS = [
    [0.00, 59, 76, 192],
    [0.25, 124, 159, 249],
    [0.50, 221, 221, 221],
    [0.75, 220, 131, 116],
    [1.00, 180, 4, 38],
  ];
  const VIRIDIS_STOPS = [
    [0.00, 68, 1, 84],
    [0.13, 71, 44, 122],
    [0.25, 59, 81, 139],
    [0.38, 44, 113, 142],
    [0.50, 33, 144, 141],
    [0.63, 39, 173, 129],
    [0.75, 92, 200, 99],
    [0.88, 170, 220, 50],
    [1.00, 253, 231, 37],
  ];

  function interpStops(stops, t) {
    t = Math.min(1, Math.max(0, t));
    for (let i = 0; i < stops.length - 1; i++) {
      const [t0, r0, g0, b0] = stops[i];
      const [t1, r1, g1, b1] = stops[i + 1];
      if (t <= t1 || i === stops.length - 2) {
        const local = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
        return [
          Math.round(r0 + (r1 - r0) * local),
          Math.round(g0 + (g1 - g0) * local),
          Math.round(b0 + (b1 - b0) * local),
        ];
      }
    }
    return stops[stops.length - 1].slice(1);
  }

  function colormapColor(preset, t) {
    if (preset === "coolwarm") return interpStops(COOLWARM_STOPS, t);
    if (preset === "viridis") return interpStops(VIRIDIS_STOPS, t);
    return jetColor(t);
  }

  // Discretiza t em `ndiv` faixas (mesma ideia de "jet em N faixas" ja usada
  // em scripts/plot_comparison.py) -- devolve o t representativo (centro da
  // faixa) em vez de um gradiente continuo.
  function quantize(t, ndiv) {
    if (!ndiv || ndiv < 1) return t;
    const level = Math.min(ndiv - 1, Math.floor(Math.min(1, Math.max(0, t)) * ndiv));
    return (level + 0.5) / ndiv;
  }

  function currentColorConfig() {
    const ndiv = Math.max(1, parseInt(document.getElementById("r-colorbar-divisions").value, 10) || 1);
    return { preset: document.getElementById("r-colorbar-preset").value, ndiv };
  }

  function fieldValues(frame, campo) {
    if (campo === "mag") {
      const vx = frame.vx, vy = frame.vy;
      return vx.map((v, i) => Math.hypot(v, vy[i]));
    }
    return frame[campo];
  }

  function updateColorbar(vmin, vmax) {
    const { preset, ndiv } = currentColorConfig();
    const stops = [];
    for (let i = 0; i < ndiv; i++) {
      const [r, g, b] = colormapColor(preset, (i + 0.5) / ndiv);
      const color = `rgb(${r},${g},${b})`;
      stops.push(`${color} ${(i / ndiv) * 100}%`, `${color} ${((i + 1) / ndiv) * 100}%`);
    }
    // "to top": posicao 0% = base (min), 100% = topo (max) -- bate com os
    // rotulos (max em cima, min embaixo, ver .colorbar-labels). Os rotulos
    // intermediarios (1/4, 1/2, 3/4) usam a mesma ordem, de cima pra baixo
    // -- ".colorbar-labels" e' flex coluna com space-between, entao 5
    // rotulos caem sozinhos em 100/75/50/25/0% da barra.
    document.getElementById("r-colorbar-grad").style.background = `linear-gradient(to top, ${stops.join(",")})`;
    const passo = (vmax - vmin) / 4;
    document.getElementById("r-cb-max").textContent = vmax.toFixed(5);
    document.getElementById("r-cb-q3").textContent = (vmin + 3 * passo).toFixed(2);
    document.getElementById("r-cb-q2").textContent = (vmin + 2 * passo).toFixed(2);
    document.getElementById("r-cb-q1").textContent = (vmin + 1 * passo).toFixed(2);
    document.getElementById("r-cb-min").textContent = vmin.toFixed(5);
  }

  // Passo "redondo" (1/2/5 * 10^n) pra grade de coordenadas ter ~`target`
  // linhas cobrindo `range`.
  function niceStep(range, target) {
    if (!(range > 0)) return 1;
    const raw = range / target;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
    return step * mag;
  }

  function drawGeometricGrid(x0, x1, y0, y1, toPx) {
    const stepX = niceStep(x1 - x0, 6);
    const stepY = niceStep(y1 - y0, 6);
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 1;
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillStyle = "rgba(230,233,239,0.6)";
    for (let x = Math.ceil(x0 / stepX) * stepX; x <= x1 + 1e-9; x += stepX) {
      const [px] = toPx(x, y0);
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, canvas.height);
      ctx.stroke();
      ctx.fillText(x.toPrecision(3), px + 2, canvas.height - 4);
    }
    for (let y = Math.ceil(y0 / stepY) * stepY; y <= y1 + 1e-9; y += stepY) {
      const [, py] = toPx(x0, y);
      ctx.beginPath();
      ctx.moveTo(0, py);
      ctx.lineTo(canvas.width, py);
      ctx.stroke();
      ctx.fillText(y.toPrecision(3), 2, py - 2);
    }
    ctx.restore();
  }

  // -- preenchimento do campo: plano (1 cor por triangulo) vs suave --------

  function renderFieldFlat(frame, valores, vmin, range, toPx, preset, ndiv) {
    for (const tri of frame.triangles) {
      const [a, b, c] = tri;
      const val = (valores[a] + valores[b] + valores[c]) / 3;
      const [r, g, bch] = colormapColor(preset, quantize((val - vmin) / range, ndiv));
      const [ax, ay] = toPx(frame.points[a][0], frame.points[a][1]);
      const [bx, by] = toPx(frame.points[b][0], frame.points[b][1]);
      const [cx, cy] = toPx(frame.points[c][0], frame.points[c][1]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.lineTo(cx, cy);
      ctx.closePath();
      ctx.fillStyle = `rgb(${r},${g},${bch})`;
      ctx.fill();
    }
  }

  // buffer de pixels reaproveitado entre quadros (evita realocar 4 bytes por
  // pixel a cada frame durante o playback)
  let smoothBuffer = null;
  let smoothBufferW = 0, smoothBufferH = 0;

  // Rasteriza cada triangulo por pixel, interpolando o VALOR do campo por
  // coordenadas baricentricas (nao a cor -- interpolar a cor pos-colormap
  // distorceria o mapeamento, sobretudo em presets divergentes como
  // coolwarm) e so entao aplicando o colormap/quantizacao. E' o analogo
  // "Gouraud" da interpolacao P1 que a propria malha ja representa -- os
  // vertices compartilhados entre triangulos garantem continuidade nas
  // arestas internas, sem costura visivel.
  function renderFieldSmooth(frame, valores, vmin, range, toPx, preset, ndiv) {
    const W = canvas.width, H = canvas.height;
    if (!smoothBuffer || smoothBufferW !== W || smoothBufferH !== H) {
      smoothBuffer = new Uint8ClampedArray(W * H * 4);
      smoothBufferW = W;
      smoothBufferH = H;
    } else {
      smoothBuffer.fill(0);
    }

    // `putImageData` (no fim desta funcao) escreve pixels crus direto no
    // canvas -- ao contrario de fill()/stroke(), ela IGNORA por completo a
    // transformacao ativa (ctx.translate/scale do zoom/pan em render()).
    // Por isso o zoom/pan tem que ser aplicado aqui manualmente, em cima
    // das coordenadas de tela ja calculadas por `toPx` (o ajuste de
    // bbox-pro-canvas de sempre) -- senao o campo fica "colado" na tela
    // enquanto malha/grid/linha (desenhados com fill/stroke, que respeitam
    // a transformacao) zoom/pan normalmente.
    const zx = (x) => viewState.panX + viewState.zoom * x;
    const zy = (y) => viewState.panY + viewState.zoom * y;

    for (const tri of frame.triangles) {
      const [a, b, c] = tri;
      const va = valores[a], vb = valores[b], vc = valores[c];
      const [ax0, ay0] = toPx(frame.points[a][0], frame.points[a][1]);
      const [bx0, by0] = toPx(frame.points[b][0], frame.points[b][1]);
      const [cx0, cy0] = toPx(frame.points[c][0], frame.points[c][1]);
      const ax = zx(ax0), ay = zy(ay0);
      const bx = zx(bx0), by = zy(by0);
      const cx = zx(cx0), cy = zy(cy0);

      const minX = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
      const maxX = Math.min(W - 1, Math.ceil(Math.max(ax, bx, cx)));
      const minY = Math.max(0, Math.floor(Math.min(ay, by, cy)));
      const maxY = Math.min(H - 1, Math.ceil(Math.max(ay, by, cy)));
      if (minX > maxX || minY > maxY) continue;

      const denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
      if (denom === 0) continue;

      for (let py = minY; py <= maxY; py++) {
        const pyc = py + 0.5;
        for (let px = minX; px <= maxX; px++) {
          const pxc = px + 0.5;
          const wa = ((by - cy) * (pxc - cx) + (cx - bx) * (pyc - cy)) / denom;
          const wb = ((cy - ay) * (pxc - cx) + (ax - cx) * (pyc - cy)) / denom;
          const wc = 1 - wa - wb;
          if (wa < -1e-6 || wb < -1e-6 || wc < -1e-6) continue;
          const val = wa * va + wb * vb + wc * vc;
          const [r, g, bch] = colormapColor(preset, quantize((val - vmin) / range, ndiv));
          const idx = (py * W + px) * 4;
          smoothBuffer[idx] = r;
          smoothBuffer[idx + 1] = g;
          smoothBuffer[idx + 2] = bch;
          smoothBuffer[idx + 3] = 255;
        }
      }
    }

    ctx.putImageData(new ImageData(smoothBuffer, W, H), 0, 0);
  }

  // -- grafico sobre linha: interpolacao + desenho -------------------------

  // Localiza o triangulo que contem (x,y) por busca linear (nao ha
  // estrutura de adjacencia no lado do cliente) e interpola o valor nodal
  // por coordenadas baricentricas -- mesma logica de renderFieldSmooth, so
  // que avaliada num ponto arbitrario em vez de rasterizada em toda a tela.
  // O tamanho tipico das malhas do projeto (milhares de triangulos) x poucas
  // centenas de amostras por linha fica bem dentro do orcamento de um clique.
  function interpolateFieldAt(x, y, frame, valores) {
    for (const tri of frame.triangles) {
      const [a, b, c] = tri;
      const pa = frame.points[a], pb = frame.points[b], pc = frame.points[c];
      if (x < Math.min(pa[0], pb[0], pc[0]) || x > Math.max(pa[0], pb[0], pc[0])) continue;
      if (y < Math.min(pa[1], pb[1], pc[1]) || y > Math.max(pa[1], pb[1], pc[1])) continue;
      const denom = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1]);
      if (denom === 0) continue;
      const wa = ((pb[1] - pc[1]) * (x - pc[0]) + (pc[0] - pb[0]) * (y - pc[1])) / denom;
      const wb = ((pc[1] - pa[1]) * (x - pc[0]) + (pa[0] - pc[0]) * (y - pc[1])) / denom;
      const wc = 1 - wa - wb;
      if (wa < -1e-9 || wb < -1e-9 || wc < -1e-9) continue;
      return wa * valores[a] + wb * valores[b] + wc * valores[c];
    }
    return null; // fora do dominio da malha
  }

  function sampleLine(frame, valores, ln, nSamples) {
    const { x1, y1, x2, y2 } = ln;
    const length = Math.hypot(x2 - x1, y2 - y1);
    const samples = [];
    for (let i = 0; i < nSamples; i++) {
      const t = nSamples === 1 ? 0 : i / (nSamples - 1);
      const x = x1 + (x2 - x1) * t;
      const y = y1 + (y2 - y1) * t;
      samples.push({ dist: t * length, x, y, value: interpolateFieldAt(x, y, frame, valores) });
    }
    return { samples, length };
  }

  function drawLineOverlay(toPx) {
    if (drawingLine && drawFirstPoint) {
      const [px, py] = toPx(drawFirstPoint[0], drawFirstPoint[1]);
      ctx.save();
      ctx.fillStyle = "#ffd23f";
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    if (!line || !document.getElementById("r-line-toggle").checked) return;
    const [ax, ay] = toPx(line.x1, line.y1);
    const [bx, by] = toPx(line.x2, line.y2);
    ctx.save();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#ffffff";
    for (const [px, py] of [[ax, ay], [bx, by]]) {
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function drawLinePlotGrid(lctx, xMin, xMax, yMin, yMax, toX, toY, padLeft, padTop, w, h) {
    const stepX = niceStep(xMax - xMin, 6);
    const stepY = niceStep(yMax - yMin, 6);
    lctx.save();
    lctx.strokeStyle = "rgba(0,0,0,0.14)";
    lctx.lineWidth = 1;
    for (let x = Math.ceil(xMin / stepX) * stepX; x <= xMax + 1e-9; x += stepX) {
      const px = toX(x);
      lctx.beginPath();
      lctx.moveTo(px, padTop);
      lctx.lineTo(px, padTop + h);
      lctx.stroke();
    }
    for (let y = Math.ceil(yMin / stepY) * stepY; y <= yMax + 1e-9; y += stepY) {
      const py = toY(y);
      lctx.beginPath();
      lctx.moveTo(padLeft, py);
      lctx.lineTo(padLeft + w, py);
      lctx.stroke();
    }
    lctx.restore();
  }

  // Interpola linearmente o valor de `samples` (ordenadas por `coordOf`
  // crescente, como `sampleLine` sempre gera) no ponto `target` da mesma
  // coordenada. `null` se `target` cair fora do trecho coberto pela
  // referencia ou entre dois pontos invalidos (fora do dominio da malha).
  function interpolarEm(samples, coordOf, target) {
    for (let i = 0; i < samples.length - 1; i++) {
      const c0 = coordOf(samples[i]), c1 = coordOf(samples[i + 1]);
      if ((target >= c0 && target <= c1) || (target <= c0 && target >= c1)) {
        const v0 = samples[i].value, v1 = samples[i + 1].value;
        if (v0 === null || v1 === null) return null;
        const t = c1 === c0 ? 0 : (target - c0) / (c1 - c0);
        return v0 + (v1 - v0) * t;
      }
    }
    return null;
  }

  // Magnitude do erro relativo |(atual - referencia) / referencia|, ponto a
  // ponto na mesma coordenada da linha atual -- a referencia (plotagem
  // carregada) e' reamostrada por interpolacao linear pra alinhar com os
  // pontos da linha atual, que podem ter numero/posicao de amostras
  // diferentes. Plotada num grafico separado (ver renderLinePlot), nao
  // misturada com a serie de valores.
  function construirSerieErro(atual, referencia, coordKey) {
    const coordOf = (p) => (coordKey === "dist" ? p.dist : (p[coordKey] ?? p.dist));
    const samples = atual.samples.map((p) => {
      const c = coordOf(p);
      const vRef = interpolarEm(referencia.samples, coordOf, c);
      let value = null;
      if (p.value !== null && vRef !== null && Math.abs(vRef) > 1e-12) {
        value = Math.abs((p.value - vRef) / vRef);
      }
      return { dist: p.dist, x: p.x, y: p.y, value };
    });
    return {
      name: `erro relativo: ${atual.name} vs ${referencia.name}`,
      campo: "|erro rel.|",
      color: "#d6304a",
      dash: "dashdot",
      samples,
      length: atual.length,
    };
  }

  // `series`: [{name, campo, color, dash, samples: [{dist,value|null}], length}]
  // Desenha em `canvasId` -- usado tanto pro grafico principal
  // (r-line-canvas: linha atual + plotagens carregadas) quanto pro grafico
  // de erro relativo, num canvas separado ao lado (r-error-canvas), que so'
  // tem a serie de erro. `opts.forceAuto` ignora "eixos automaticos" (usado
  // pelo grafico de erro, cuja escala de magnitude nao tem relacao com os
  // limites manuais configurados pra variavel do grafico principal).
  function renderLinePlot(canvasId, series, opts = {}) {
    const lc = document.getElementById(canvasId);
    const lctx = lc.getContext("2d");

    // fundo sempre branco, pintado dentro do proprio canvas (nao so' via
    // CSS do wrapper) -- assim "salvar imagem" do canvas sai com fundo em
    // vez de transparente.
    lctx.fillStyle = "#ffffff";
    lctx.fillRect(0, 0, lc.width, lc.height);

    const allValid = series.flatMap((s) => s.samples.filter((p) => p.value !== null));
    if (!allValid.length) {
      lctx.fillStyle = "rgba(0,0,0,0.45)";
      lctx.font = "12px system-ui, sans-serif";
      lctx.fillText("Linha fora do domínio da malha.", 10, lc.height / 2);
      return;
    }

    // coordenada (posicao ao longo da linha): distancia (padrao), x ou y do
    // proprio ponto amostrado -- pontos de arquivos salvos antes dessa opcao
    // existir nao tem x/y gravado, caem de volta pra distancia. Por padrao
    // vai no eixo horizontal e a variavel (r-line-y-axis, ja' resolvida em
    // p.value na amostragem) no vertical; "eixo invertido" troca os dois --
    // vale igual pro grafico de erro, que usa a mesma coordenada/orientacao
    // do grafico principal (so' a "variavel" dele e' a magnitude do erro).
    const coordKey = document.getElementById("r-line-x-axis").value;
    const coordOf = (p) => (coordKey === "dist" ? p.dist : (p[coordKey] ?? p.dist));
    const varOf = (p) => p.value;
    const invert = document.getElementById("r-line-axes-invert").checked;
    const horizOf = invert ? varOf : coordOf;
    const vertOf = invert ? coordOf : varOf;

    const autoAxes = opts.forceAuto || document.getElementById("r-line-axes-auto").checked;
    let xMin, xMax, yMin, yMax;
    if (autoAxes) {
      const hs = allValid.map(horizOf);
      const vs = allValid.map(vertOf);
      xMin = Math.min(...hs);
      xMax = Math.max(...hs);
      yMin = Math.min(...vs);
      yMax = Math.max(...vs);
    } else {
      xMin = parseFloat(document.getElementById("r-line-xmin").value);
      xMax = parseFloat(document.getElementById("r-line-xmax").value);
      yMin = parseFloat(document.getElementById("r-line-ymin").value);
      yMax = parseFloat(document.getElementById("r-line-ymax").value);
      if (!Number.isFinite(xMin)) xMin = 0;
      if (!Number.isFinite(xMax) || xMax <= xMin) xMax = xMin + 1;
      if (!Number.isFinite(yMin)) yMin = 0;
      if (!Number.isFinite(yMax) || yMax <= yMin) yMax = yMin + 1;
    }
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;

    // legenda agora fica abaixo do grafico (nao mais por cima, no topo) --
    // padTop soh precisa de folga pro rotulo do maximo do eixo vertical;
    // padBottom cobre os rotulos do eixo horizontal + a linha de legenda.
    const padTop = 18, padBottom = 46, padLeft = 56, padRight = 14;
    const w = lc.width - padLeft - padRight;
    const h = lc.height - padTop - padBottom;

    const toX = (d) => padLeft + ((d - xMin) / xRange) * w;
    const toY = (v) => padTop + h - ((v - yMin) / yRange) * h;

    if (document.getElementById("r-line-grid-toggle").checked) {
      drawLinePlotGrid(lctx, xMin, xMax, yMin, yMax, toX, toY, padLeft, padTop, w, h);
    }

    lctx.strokeStyle = "rgba(0,0,0,0.55)";
    lctx.lineWidth = 1;
    lctx.beginPath();
    lctx.moveTo(padLeft, padTop);
    lctx.lineTo(padLeft, padTop + h);
    lctx.lineTo(padLeft + w, padTop + h);
    lctx.stroke();

    lctx.fillStyle = "rgba(0,0,0,0.8)";
    lctx.font = "10px system-ui, sans-serif";
    lctx.textAlign = "right";
    lctx.fillText(yMax.toPrecision(4), padLeft - 6, padTop + 8);
    lctx.fillText(yMin.toPrecision(4), padLeft - 6, padTop + h);
    lctx.textAlign = "center";
    lctx.fillText(xMin.toPrecision(4), padLeft, padTop + h + 14);
    lctx.fillText(xMax.toPrecision(4), padLeft + w, padTop + h + 14);

    // recorta pro retangulo do grafico -- com eixos manuais mais estreitos
    // que os dados, a linha nao pode vazar por cima dos rotulos/legenda.
    lctx.save();
    lctx.beginPath();
    lctx.rect(padLeft, padTop, w, h);
    lctx.clip();
    for (const s of series) {
      lctx.strokeStyle = s.color;
      lctx.lineWidth = 2;
      lctx.setLineDash((LINE_DASH_STYLES[s.dash] || LINE_DASH_STYLES.solid).pattern);
      lctx.beginPath();
      let started = false;
      for (const p of s.samples) {
        if (p.value === null) { started = false; continue; }
        const px = toX(horizOf(p)), py = toY(vertOf(p));
        if (!started) { lctx.moveTo(px, py); started = true; }
        else lctx.lineTo(px, py);
      }
      lctx.stroke();
    }
    lctx.setLineDash([]);
    lctx.restore();

    // legenda: amostra da linha (cor + tracado) + "nome (campo)" por serie,
    // numa linha abaixo dos rotulos do eixo horizontal.
    let lx = padLeft;
    const ly = padTop + h + 34;
    lctx.font = "10px system-ui, sans-serif";
    lctx.textAlign = "left";
    for (const s of series) {
      const label = `${s.name} (${s.campo})`;
      lctx.strokeStyle = s.color;
      lctx.lineWidth = 2;
      lctx.setLineDash((LINE_DASH_STYLES[s.dash] || LINE_DASH_STYLES.solid).pattern);
      lctx.beginPath();
      lctx.moveTo(lx, ly - 4);
      lctx.lineTo(lx + 14, ly - 4);
      lctx.stroke();
      lctx.setLineDash([]);
      lctx.fillStyle = "rgba(0,0,0,0.85)";
      lctx.fillText(label, lx + 18, ly);
      lx += 18 + lctx.measureText(label).width + 16;
      if (lx > lc.width - padRight - 40) { lx = padLeft; break; } // sem quebra de linha -- so evita estourar o canvas
    }
  }

  function updateLinePlot(frame) {
    const hint = document.getElementById("r-line-hint");
    const errorWrap = document.getElementById("r-error-canvas-wrap");
    const series = [];
    let liveSeries = null;

    if (line) {
      const nSamples = Math.max(2, parseInt(document.getElementById("r-line-samples").value, 10) || 200);
      // eixo vertical do grafico sobre a linha e' independente do campo
      // mostrado no canvas 2D (r-field) -- assim da pra, por exemplo, olhar
      // a pressao no campo e o vx ao longo da linha ao mesmo tempo.
      const campo = document.getElementById("r-line-y-axis").value;
      const yValores = fieldValues(frame, campo);
      const { samples, length } = sampleLine(frame, yValores, line, nSamples);
      const name = document.getElementById("r-line-name").value.trim() || "atual";
      liveSeries = {
        name, campo: campo === "mag" ? "|v|" : campo,
        color: liveLineStyle.color, dash: liveLineStyle.dash, samples, length,
      };
      series.push(liveSeries);
    }
    series.push(...savedOverlays);

    // erro relativo: grafico separado, ao lado do principal (nao misturado
    // na mesma serie/eixo de valor, ver renderLinePlot).
    let errorSeries = null;
    const errorOn = document.getElementById("r-line-error-toggle").checked;
    if (errorOn && liveSeries && savedOverlays.length) {
      const refIdx = parseInt(document.getElementById("r-line-error-ref").value, 10);
      const referencia = savedOverlays[refIdx];
      if (referencia) {
        const coordKey = document.getElementById("r-line-x-axis").value;
        errorSeries = construirSerieErro(liveSeries, referencia, coordKey);
      }
    }
    errorWrap.hidden = !errorSeries;
    if (errorSeries) renderLinePlot("r-error-canvas", [errorSeries], { forceAuto: true });

    if (!series.length) {
      hint.hidden = false;
      hint.textContent = "Nenhuma linha definida.";
      const lc = document.getElementById("r-line-canvas");
      lc.getContext("2d").fillStyle = "#ffffff";
      lc.getContext("2d").fillRect(0, 0, lc.width, lc.height);
      return;
    }
    hint.hidden = true;
    renderLinePlot("r-line-canvas", series);
  }

  function renderOverlayList() {
    const ul = document.getElementById("r-line-overlays");
    ul.innerHTML = "";
    savedOverlays.forEach((ov, idx) => {
      const colorInput = el("input", {
        type: "color", value: ov.color, class: "overlay-color", title: "Cor da linha",
      });
      colorInput.addEventListener("input", () => { ov.color = colorInput.value; rerender(); });

      const dashSelect = el(
        "select",
        { class: "overlay-dash", title: "Estilo da linha" },
        Object.entries(LINE_DASH_STYLES).map(([value, { label }]) => el("option", { value, text: label })),
      );
      dashSelect.value = ov.dash || "solid";
      dashSelect.addEventListener("change", () => { ov.dash = dashSelect.value; rerender(); });

      ul.appendChild(el("li", { class: "overlay-chip" }, [
        colorInput,
        dashSelect,
        el("span", { class: "name", text: `${ov.name} (${ov.campo})` }),
        el("button", {
          type: "button",
          title: "Remover plotagem carregada",
          onclick: () => { savedOverlays.splice(idx, 1); renderOverlayList(); rerender(); },
          text: "✕",
        }),
      ]));
    });

    // seletor de referencia do erro relativo -- so' faz sentido com pelo
    // menos uma plotagem carregada; tenta manter a mesma referencia
    // selecionada (por nome) quando a lista muda de tamanho/ordem.
    const refSelect = document.getElementById("r-line-error-ref");
    const nomePrevio = savedOverlays[parseInt(refSelect.value, 10)]?.name;
    refSelect.innerHTML = "";
    savedOverlays.forEach((ov, idx) => {
      refSelect.appendChild(el("option", { value: String(idx), text: `${ov.name} (${ov.campo})` }));
    });
    const hasOverlays = savedOverlays.length > 0;
    document.getElementById("r-line-error-toggle").disabled = !hasOverlays;
    refSelect.disabled = !hasOverlays;
    if (!hasOverlays) document.getElementById("r-line-error-toggle").checked = false;
    if (hasOverlays) {
      const mantidoIdx = savedOverlays.findIndex((o) => o.name === nomePrevio);
      refSelect.value = String(mantidoIdx >= 0 ? mantidoIdx : 0);
    }
  }

  function setLine(x1, y1, x2, y2) {
    line = { x1, y1, x2, y2 };
    document.getElementById("r-line-x1").value = x1.toPrecision(6);
    document.getElementById("r-line-y1").value = y1.toPrecision(6);
    document.getElementById("r-line-x2").value = x2.toPrecision(6);
    document.getElementById("r-line-y2").value = y2.toPrecision(6);
    document.getElementById("r-line-toggle").checked = true;
    document.getElementById("r-line-toggle").disabled = false;
    document.getElementById("r-line-clear").disabled = false;
    document.getElementById("r-line-save").disabled = false;
  }

  function toWorld(px, py) {
    const { x0, y0, scale, offX, offY, zoom, panX, panY } = currentTransform;
    const bx = (px - panX) / zoom;
    const by = (py - panY) / zoom;
    return [(bx - offX) / scale + x0, (canvas.height - by - offY) / scale + y0];
  }

  function eventToCanvasPx(ev) {
    const rect = canvas.getBoundingClientRect();
    return [(ev.clientX - rect.left) * (canvas.width / rect.width), (ev.clientY - rect.top) * (canvas.height / rect.height)];
  }

  function stopDrawingLine() {
    drawingLine = false;
    drawFirstPoint = null;
    canvas.style.cursor = "";
    document.getElementById("r-line-draw").textContent = "Desenhar linha";
  }

  // Limpa a linha e os campos x1/y1/x2/y2 -- usado tanto pelo botao
  // "remover linha" quanto ao trocar de simulacao (r-run), onde uma linha
  // desenhada sobre a geometria antiga nao faz sentido pra nova.
  function clearLine() {
    line = null;
    document.getElementById("r-line-toggle").checked = false;
    document.getElementById("r-line-toggle").disabled = true;
    document.getElementById("r-line-clear").disabled = true;
    document.getElementById("r-line-save").disabled = true;
    for (const id of ["r-line-x1", "r-line-y1", "r-line-x2", "r-line-y2"]) document.getElementById(id).value = "";
  }

  function bindLineControls() {
    document.getElementById("r-line-draw").addEventListener("click", () => {
      if (drawingLine) { stopDrawingLine(); rerender(); return; }

      // se x1/y1/x2/y2 ja' estiverem preenchidos (o usuario digitou os
      // valores em vez de clicar 2 pontos), desenha direto com eles em vez
      // de entrar no modo "clique 2 pontos" -- a linha resultante pode ser
      // ajustada depois do mesmo jeito de sempre (reclicar/editar os campos).
      const coordIds = ["r-line-x1", "r-line-y1", "r-line-x2", "r-line-y2"];
      const vals = coordIds.map((id) => parseFloat(document.getElementById(id).value));
      if (vals.every((v) => Number.isFinite(v))) {
        setLine(vals[0], vals[1], vals[2], vals[3]);
        rerender();
        return;
      }

      drawingLine = true;
      drawFirstPoint = null;
      canvas.style.cursor = "crosshair";
      document.getElementById("r-line-draw").textContent = "Clique 2 pontos… (cancelar)";
    });

    canvas.addEventListener("click", (ev) => {
      if (!drawingLine || !currentTransform) return;
      const [px, py] = eventToCanvasPx(ev);
      const [wx, wy] = toWorld(px, py);
      if (!drawFirstPoint) {
        drawFirstPoint = [wx, wy];
        rerender();
      } else {
        setLine(drawFirstPoint[0], drawFirstPoint[1], wx, wy);
        stopDrawingLine();
        rerender();
      }
    });

    document.getElementById("r-line-clear").addEventListener("click", () => {
      clearLine();
      rerender();
    });

    const coordIds = ["r-line-x1", "r-line-y1", "r-line-x2", "r-line-y2"];
    for (const id of coordIds) {
      document.getElementById(id).addEventListener("change", () => {
        const vals = coordIds.map((i) => parseFloat(document.getElementById(i).value));
        if (vals.every((v) => Number.isFinite(v))) {
          setLine(vals[0], vals[1], vals[2], vals[3]);
          rerender();
        }
      });
    }

    document.getElementById("r-line-toggle").addEventListener("change", rerender);
    document.getElementById("r-line-samples").addEventListener("input", rerender);

    // -- aparencia do grafico sobre a linha: grid, eixos manuais,
    // cor/tracado da linha "ao vivo" (fundo e' sempre branco, sem opcao de
    // escolher -- ver renderLinePlot) ------------------------------------
    document.getElementById("r-line-grid-toggle").addEventListener("change", rerender);
    const axesManual = document.getElementById("r-line-axes-manual");
    axesManual.style.display = "none"; // estado inicial = "eixos automaticos" marcado
    document.getElementById("r-line-axes-auto").addEventListener("change", async (e) => {
      const manual = !e.target.checked;
      axesManual.style.display = manual ? "" : "none";
      // pre-preenche os campos manuais com o range atual (em vez de comecar
      // vazio, que cairia no 0-1 generico do fallback em renderLinePlot) --
      // assim o usuario so ajusta a partir de algo plausivel.
      if (manual && currentRun) {
        const frame = await getFrame(frameNumbers[frameIdx]);
        const campo = document.getElementById("r-line-y-axis").value;
        const valores = fieldValues(frame, campo);
        const series = [...savedOverlays];
        if (line) {
          const nSamples = Math.max(2, parseInt(document.getElementById("r-line-samples").value, 10) || 200);
          series.push(sampleLine(frame, valores, line, nSamples));
        }
        const allValid = series.flatMap((s) => s.samples.filter((p) => p.value !== null));
        if (allValid.length) {
          const coordKey = document.getElementById("r-line-x-axis").value;
          const coordOf = (p) => (coordKey === "dist" ? p.dist : (p[coordKey] ?? p.dist));
          const varOf = (p) => p.value;
          const invert = document.getElementById("r-line-axes-invert").checked;
          const horizOf = invert ? varOf : coordOf;
          const vertOf = invert ? coordOf : varOf;
          document.getElementById("r-line-xmin").value = Math.min(...allValid.map(horizOf)).toPrecision(4);
          document.getElementById("r-line-xmax").value = Math.max(...allValid.map(horizOf)).toPrecision(4);
          document.getElementById("r-line-ymin").value = Math.min(...allValid.map(vertOf)).toPrecision(4);
          document.getElementById("r-line-ymax").value = Math.max(...allValid.map(vertOf)).toPrecision(4);
        }
      }
      rerender();
    });
    for (const id of ["r-line-xmin", "r-line-xmax", "r-line-ymin", "r-line-ymax"]) {
      document.getElementById(id).addEventListener("input", rerender);
    }
    document.getElementById("r-line-x-axis").addEventListener("change", rerender);
    document.getElementById("r-line-y-axis").addEventListener("change", rerender);
    document.getElementById("r-line-axes-invert").addEventListener("change", rerender);
    document.getElementById("r-line-error-toggle").addEventListener("change", rerender);
    document.getElementById("r-line-error-ref").addEventListener("change", rerender);
    document.getElementById("r-line-color").addEventListener("input", (e) => {
      liveLineStyle.color = e.target.value;
      rerender();
    });
    document.getElementById("r-line-dash").addEventListener("change", (e) => {
      liveLineStyle.dash = e.target.value;
      rerender();
    });

    // -- salvar/carregar plotagens em arquivo .json --------------------
    document.getElementById("r-line-save").addEventListener("click", async () => {
      if (!line || !currentRun) return;
      const frame = await getFrame(frameNumbers[frameIdx]);
      const campo = document.getElementById("r-field").value;
      const valores = fieldValues(frame, campo);
      const nSamples = Math.max(2, parseInt(document.getElementById("r-line-samples").value, 10) || 200);
      const { samples, length } = sampleLine(frame, valores, line, nSamples);
      const runLabel = document.getElementById("r-run").selectedOptions[0]?.textContent || String(currentRun.id);
      const name = document.getElementById("r-line-name").value.trim() || `${campo}-${frameNumbers[frameIdx]}`;
      downloadJSON(`femns-line-${slugify(name)}.json`, {
        name,
        campo: campo === "mag" ? "|v|" : campo,
        color: liveLineStyle.color,
        dash: liveLineStyle.dash,
        line: { ...line },
        length,
        samples,
        run_label: runLabel,
        frame: frameNumbers[frameIdx],
        saved_at: new Date().toISOString(),
      });
    });

    document.getElementById("r-line-load").addEventListener("click", () => {
      document.getElementById("r-line-file").click();
    });

    document.getElementById("r-line-file").addEventListener("change", async (ev) => {
      const file = ev.target.files[0];
      ev.target.value = ""; // permite recarregar o mesmo arquivo de novo depois
      if (!file) return;
      try {
        const data = JSON.parse(await file.text());
        if (!Array.isArray(data.samples)) throw new Error("arquivo sem 'samples' -- não é uma plotagem salva pelo femns");
        // cor/estilo do proprio arquivo, se foi salvo por uma versao que ja
        // gravava isso -- senao cai no palete automatico + solida de sempre.
        const color = data.color || OVERLAY_PALETTE[savedOverlays.length % OVERLAY_PALETTE.length];
        const dash = data.dash && LINE_DASH_STYLES[data.dash] ? data.dash : "solid";
        const length = data.length ?? (data.samples.length ? data.samples[data.samples.length - 1].dist : 0);
        savedOverlays.push({
          name: data.name || file.name.replace(/\.json$/i, ""),
          campo: data.campo || "?",
          color,
          dash,
          samples: data.samples,
          length,
        });
        renderOverlayList();
        rerender();
      } catch (e) {
        window.alert(`Erro ao carregar plotagem: ${e.message}`);
      }
    });

    const wrap = document.getElementById("r-line-canvas-wrap");
    const lineCanvas = document.getElementById("r-line-canvas");
    new ResizeObserver(() => {
      const w = Math.max(1, Math.round(wrap.clientWidth));
      const h = Math.max(1, Math.round(wrap.clientHeight));
      if (lineCanvas.width !== w || lineCanvas.height !== h) {
        lineCanvas.width = w;
        lineCanvas.height = h;
        rerender();
      }
    }).observe(wrap);

    const errorWrap = document.getElementById("r-error-canvas-wrap");
    const errorCanvas = document.getElementById("r-error-canvas");
    new ResizeObserver(() => {
      const w = Math.max(1, Math.round(errorWrap.clientWidth));
      const h = Math.max(1, Math.round(errorWrap.clientHeight));
      if (errorCanvas.width !== w || errorCanvas.height !== h) {
        errorCanvas.width = w;
        errorCanvas.height = h;
        rerender();
      }
    }).observe(errorWrap);
  }

  function render(frame) {
    const campo = document.getElementById("r-field").value;
    const valores = fieldValues(frame, campo);
    const vmin = Math.min(...valores);
    const vmax = Math.max(...valores);
    updateColorbar(vmin, vmax);

    // bbox do proprio quadro (nao de `currentRun.meta`, calculado so no
    // primeiro frame) -- com malha movel (mesh_motion/moving_point) os
    // pontos deslocam um pouco a cada passo, e um bbox fixo cortaria a
    // visualizacao nos ultimos quadros.
    const xs = frame.points.map((p) => p[0]);
    const ys = frame.points.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const margin = 20;
    const w = canvas.width - 2 * margin;
    const h = canvas.height - 2 * margin;
    const scale = Math.min(w / (x1 - x0 || 1), h / (y1 - y0 || 1));
    const offX = margin + (w - scale * (x1 - x0)) / 2;
    const offY = margin + (h - scale * (y1 - y0)) / 2;
    const toPx = (x, y) => [offX + (x - x0) * scale, canvas.height - (offY + (y - y0) * scale)];
    currentTransform = { x0, y0, scale, offX, offY, zoom: viewState.zoom, panX: viewState.panX, panY: viewState.panY };

    // limpa em coordenadas de tela (identidade), antes de aplicar zoom/pan --
    // senao clearRect precisaria da area em coordenadas de "mundo do pixel
    // base", que muda a cada zoom.
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    ctx.translate(viewState.panX, viewState.panY);
    ctx.scale(viewState.zoom, viewState.zoom);

    const showMesh = document.getElementById("r-mesh-toggle").checked;
    const showGrid = document.getElementById("r-grid-toggle").checked;
    const smooth = document.getElementById("r-smooth-toggle").checked;
    const { preset, ndiv } = currentColorConfig();
    const range = vmax - vmin || 1;

    if (smooth) renderFieldSmooth(frame, valores, vmin, range, toPx, preset, ndiv);
    else renderFieldFlat(frame, valores, vmin, range, toPx, preset, ndiv);

    if (showMesh) {
      ctx.strokeStyle = "rgba(255,255,255,0.15)";
      ctx.lineWidth = 0.5 / viewState.zoom;
      for (const tri of frame.triangles) {
        const [a, b, c] = tri;
        const [ax, ay] = toPx(frame.points[a][0], frame.points[a][1]);
        const [bx, by] = toPx(frame.points[b][0], frame.points[b][1]);
        const [cx, cy] = toPx(frame.points[c][0], frame.points[c][1]);
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.lineTo(cx, cy);
        ctx.closePath();
        ctx.stroke();
      }
    }

    if (showGrid) drawGeometricGrid(x0, x1, y0, y1, toPx);

    drawLineOverlay(toPx);
    ctx.restore();

    updateLinePlot(frame);
  }

  // -- zoom (scroll) / pan (arrastar com botao direito) sobre o campo -----

  function bindViewportControls() {
    canvas.addEventListener("contextmenu", (ev) => ev.preventDefault());

    canvas.addEventListener("wheel", (ev) => {
      if (!currentTransform) return;
      ev.preventDefault();
      const [mx, my] = eventToCanvasPx(ev);
      const fator = Math.exp(-ev.deltaY * 0.0015);
      const novoZoom = Math.min(30, Math.max(0.2, viewState.zoom * fator));
      // mantem o ponto do campo sob o cursor fixo na tela ao mudar o zoom
      const baseX = (mx - viewState.panX) / viewState.zoom;
      const baseY = (my - viewState.panY) / viewState.zoom;
      viewState.panX = mx - novoZoom * baseX;
      viewState.panY = my - novoZoom * baseY;
      viewState.zoom = novoZoom;
      rerender();
    }, { passive: false });

    let panning = false;
    let lastPx = null;
    canvas.addEventListener("mousedown", (ev) => {
      if (ev.button !== 2) return; // so botao direito
      panning = true;
      lastPx = eventToCanvasPx(ev);
      canvas.style.cursor = "grabbing";
      ev.preventDefault();
    });
    window.addEventListener("mousemove", (ev) => {
      if (!panning) return;
      const [px, py] = eventToCanvasPx(ev);
      viewState.panX += px - lastPx[0];
      viewState.panY += py - lastPx[1];
      lastPx = [px, py];
      rerender();
    });
    window.addEventListener("mouseup", (ev) => {
      if (ev.button !== 2 || !panning) return;
      panning = false;
      canvas.style.cursor = drawingLine ? "crosshair" : "";
    });

    // atalho pra sair do zoom/pan sem precisar caçar a proporcao certa
    canvas.addEventListener("dblclick", () => {
      if (drawingLine) return; // nao interfere no fluxo de desenhar linha
      viewState = { zoom: 1, panX: 0, panY: 0 };
      rerender();
    });
  }

  async function getFrame(n) {
    if (frameCache.has(n)) return frameCache.get(n);
    const frame = await api(`/api/results/${currentRun.id}/frame/${n}`);
    if (frameCache.size >= MAX_CACHE) frameCache.delete(frameCache.keys().next().value);
    frameCache.set(n, frame);
    return frame;
  }

  async function showFrameAt(idx) {
    idx = Math.min(Math.max(idx, 0), frameNumbers.length - 1);
    frameIdx = idx;
    const n = frameNumbers[idx];
    document.getElementById("r-slider").value = idx;
    document.getElementById("r-frame-label").textContent = `quadro ${n} (${idx + 1}/${frameNumbers.length})`;
    const frame = await getFrame(n);
    render(frame);
  }

  function stopPlay() {
    playing = false;
    document.getElementById("r-play").textContent = "▶";
    if (playTimer) clearTimeout(playTimer);
  }

  function startPlay() {
    playing = true;
    document.getElementById("r-play").textContent = "⏸";
    const step = async () => {
      if (!playing) return;
      if (frameIdx >= frameNumbers.length - 1) { stopPlay(); return; }
      await showFrameAt(frameIdx + 1);
      const fps = Math.max(1, parseFloat(document.getElementById("r-fps").value) || 12);
      playTimer = setTimeout(step, 1000 / fps);
    };
    step();
  }

  async function loadRun(runId) {
    stopPlay();
    viewState = { zoom: 1, panX: 0, panY: 0 }; // simulacao nova = geometria nova, comeca sem zoom/pan
    stopDrawingLine();
    clearLine(); // linha desenhada sobre a geometria antiga nao faz sentido pra nova simulacao
    currentRun = { id: runId, meta: await api(`/api/results/${runId}/meta`) };
    frameNumbers = [];
    for (let n = currentRun.meta.first_frame; n <= currentRun.meta.last_frame; n++) frameNumbers.push(n);
    frameCache.clear();
    document.getElementById("r-slider").max = frameNumbers.length - 1;
    loadedForRun = runId;
    await showFrameAt(0);
  }

  async function refreshRuns() {
    runs = await api("/api/results");
    const sel = document.getElementById("r-run");
    const prev = sel.value;
    sel.innerHTML = "";
    for (const r of runs) sel.appendChild(el("option", { value: r.id, text: `${r.label} (${r.nframes})` }));
    if (runs.some((r) => r.id === prev)) sel.value = prev;
    else if (runs.length) sel.value = runs[0].id;
  }

  function rerender() {
    if (currentRun) getFrame(frameNumbers[frameIdx]).then(render);
  }

  function bindControls() {
    document.getElementById("r-refresh").addEventListener("click", refreshRuns);
    document.getElementById("r-run").addEventListener("change", (e) => loadRun(e.target.value));
    document.getElementById("r-field").addEventListener("change", rerender);
    document.getElementById("r-mesh-toggle").addEventListener("change", rerender);
    document.getElementById("r-smooth-toggle").addEventListener("change", rerender);
    document.getElementById("r-grid-toggle").addEventListener("change", rerender);
    document.getElementById("r-colorbar-preset").addEventListener("change", rerender);
    document.getElementById("r-colorbar-divisions").addEventListener("input", rerender);
    document.getElementById("r-bg-color").addEventListener("input", (e) => {
      document.getElementById("r-canvas-wrap").style.background = e.target.value;
    });
    document.getElementById("r-first").addEventListener("click", () => { stopPlay(); showFrameAt(0); });
    document.getElementById("r-last").addEventListener("click", () => { stopPlay(); showFrameAt(frameNumbers.length - 1); });
    document.getElementById("r-step-back").addEventListener("click", () => { stopPlay(); showFrameAt(frameIdx - 1); });
    document.getElementById("r-step-fwd").addEventListener("click", () => { stopPlay(); showFrameAt(frameIdx + 1); });
    document.getElementById("r-slider").addEventListener("input", (e) => { stopPlay(); showFrameAt(parseInt(e.target.value, 10)); });
    document.getElementById("r-play").addEventListener("click", () => (playing ? stopPlay() : startPlay()));

    // canvas reescalonavel (CSS resize no wrapper) -- ajusta a resolucao
    // interna do canvas pro novo tamanho e redesenha o quadro atual.
    const wrap = document.getElementById("r-canvas-wrap");
    new ResizeObserver(() => {
      const w = Math.max(1, Math.round(wrap.clientWidth));
      const h = Math.max(1, Math.round(wrap.clientHeight));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        rerender();
      }
    }).observe(wrap);
  }

  let initialized = false;
  async function onShow() {
    if (!initialized) { bindControls(); bindLineControls(); bindViewportControls(); initialized = true; }
    await refreshRuns();
    const sel = document.getElementById("r-run");
    if (sel.value && sel.value !== loadedForRun) await loadRun(sel.value);
  }

  return { onShow };
})();

NovaSimulacao.init();

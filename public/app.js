const $ = (s) => document.querySelector(s);
const post = async (u, b) => (await fetch(u, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(b) }));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
let ds = null, run = null, status = null, proposed = {}, explanation = '';

async function init() {
  status = await (await fetch('/api/status')).json();
  const l = $('#llmState');
  l.className = 'llm' + (status.keyConfigured ? ' on' : '');
  l.textContent = status.keyConfigured ? `Nemotron on Token Factory: ${status.model.split('/')[1]}` : 'Nemotron off: set NEBIUS_API_KEY';
  $('#picks').insertAdjacentHTML('beforeend', Object.entries(status.controls).map(([id, c]) => `<label data-c="${id}" title="${esc(c.question)}"><input type="checkbox" name="ctl" value="${id}" checked> ${esc(c.title)}</label>`).join(''));
  const sc = await (await fetch('/api/scenarios')).json();
  $('#scenarios').innerHTML = sc.map((s, i) => `<button role="tab" type="button" data-id="${s.id}" aria-selected="${i === 0}" title="${esc(s.blurb)}">${esc(s.name)}</button>`).join('');
  $('#scenarios').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) loadScenario(b.dataset.id); });
  loadScenario(sc[0].id);
}

async function setDataset(body) {
  const r = await post('/api/dataset', body); const j = await r.json();
  if (!r.ok) { alert(j.error); return; }
  ds = j; run = null; proposed = {}; explanation = '';
  $('#out').hidden = true; $('#hyp').innerHTML = ''; $('#askMsg').textContent = ''; $('#exMsg').textContent = ''; $('#explainOut').hidden = true; $('#explain').disabled = true;
  document.querySelectorAll('#picks label').forEach((l) => l.classList.remove('rec'));
  drawBeds();
}
function loadScenario(id) {
  document.querySelectorAll('#scenarios button').forEach((b) => b.setAttribute('aria-selected', b.dataset.id === id));
  return setDataset({ scenario: id });
}

function drawBeds() {
  $('#plateCap').textContent = `${ds.name}: ${ds.n} samples, ${ds.features} features. Each column is a batch. Filled dot = label 1, ring = label 0.`;
  $('#beds').innerHTML = ds.design.map((b) => `<div class="bed ${b.mixed ? 'mixed' : ''}" style="--n:${b.n}"><div class="dots" role="img" aria-label="Batch ${esc(b.batch)}: ${b.pos} label 1, ${b.neg} label 0">${b.samples.map((s) => `<i class="d ${s.y ? 'p' : 'n'}"></i>`).join('')}</div><div class="bed-l"><b>Batch ${esc(b.batch)}</b>${b.pos} / ${b.neg} <span class="tag">${b.mixed ? 'mixed' : 'pure'}</span></div></div>`).join('');
}

$('#run').addEventListener('click', async () => {
  const controls = [...document.querySelectorAll('input[name=ctl]:checked')].map((i) => i.value);
  const btn = $('#run'); btn.disabled = true; btn.textContent = 'Running...';
  try {
    const r = await post('/api/run', { datasetId: ds.datasetId, controls }); run = await r.json();
    if (!r.ok) { alert(run.error); return; }
    drawOut(); $('#explain').disabled = !status.keyConfigured; explanation = ''; $('#explainOut').hidden = true;
  } finally { btn.disabled = false; btn.textContent = 'Run controls'; }
});

const lo = 0.2, hi = 1.0, pos = (v) => Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
function drawOut() {
  $('#out').hidden = false;
  const v = run.verdict;
  $('#verdict').className = 'verdict ' + v.id;
  $('#verdict').innerHTML = `<b>${esc(v.label)}</b><span>${esc(v.text)}</span>`;
  const rs = Object.values(run.results);
  $('#ladder').style.setProperty('--naive-pos', pos(run.results.naive.auc));
  $('#ladder').innerHTML = rs.map((r) => {
    const band = r.nullLo !== undefined ? `<span class="band" style="left:${pos(r.nullLo) * 100}%;width:${(pos(r.nullHi) - pos(r.nullLo)) * 100}%" title="95% of shuffled-label runs fall here"></span>` : '';
    const pt = r.auc === null ? '' : `<span class="pt ${r.id === 'naive' ? 'naive' : r.status || ''}" style="left:${pos(r.auc) * 100}%"></span>`;
    return `<div class="row"><div class="nm">${esc(r.title)}</div><div class="trk" style="--half:${pos(0.5) * 100}%" role="img" aria-label="${esc(r.title)} AUC ${r.auc === null ? 'not available' : r.auc.toFixed(2)}">${band}${pt}</div><div class="val">${r.auc === null ? 'n/a' : r.auc.toFixed(2)}</div></div>`;
  }).join('') + `<div class="axis"><span></span><div><span>${lo.toFixed(1)}</span><span>0.5 chance</span><span>${hi.toFixed(1)} AUC</span></div><span></span></div>`;
  $('#findings').innerHTML = rs.filter((r) => r.finding).map((r) => `<li><b>${esc(r.title)}.</b> <span>${esc(r.finding)}</span></li>`).join('');
  markHypotheses();
}

$('#ask').addEventListener('click', async () => {
  const m = $('#askMsg'); m.className = 'msg'; m.textContent = 'Asking Nemotron...'; $('#ask').disabled = true;
  try {
    const j = await (await post('/api/propose', { datasetId: ds.datasetId, note: $('#note').value })).json();
    if (!j.ok) { m.className = 'msg err'; m.textContent = j.message || j.error; return; }
    m.textContent = `${j.model.split('/')[1]}, ${(j.ms / 1000).toFixed(1)} s. Controls are tested by the app, not trusted from the model.`;
    proposed = {}; j.hypotheses.forEach((h, i) => { proposed[i] = h; });
    const rec = new Set(j.hypotheses.map((h) => h.control));
    document.querySelectorAll('#picks label').forEach((l) => { l.classList.toggle('rec', rec.has(l.dataset.c)); l.querySelector('input').checked = true; });
    drawHyp();
  } catch (e) { m.className = 'msg err'; m.textContent = 'Request failed.'; } finally { $('#ask').disabled = false; }
});
function drawHyp() {
  $('#hyp').innerHTML = Object.values(proposed).map((h) => `<li data-c="${h.control}"><div class="s">${esc(h.shortcut)}</div><div class="w">${esc(h.why)}</div><div class="chip">tests: ${esc(status.controls[h.control].title)} (not run yet)</div></li>`).join('');
  if (run) markHypotheses();
}
function markHypotheses() {
  document.querySelectorAll('#hyp li').forEach((li) => {
    const r = run.results[li.dataset.c]; const chip = li.querySelector('.chip');
    if (!r) { chip.className = 'chip'; chip.textContent = `tests: ${status.controls[li.dataset.c].title} (not selected)`; return; }
    chip.className = 'chip ' + r.status;
    chip.textContent = r.status === 'shortcut' ? `Supported: ${r.title} flags it (AUC ${r.auc.toFixed(2)})` : r.status === 'signal' ? `Not supported: ${r.title} found signal (AUC ${r.auc.toFixed(2)})` : `Not supported: ${r.title} (AUC ${r.auc === null ? 'n/a' : r.auc.toFixed(2)})`;
  });
}

$('#explain').addEventListener('click', async () => {
  const m = $('#exMsg'); m.className = 'msg'; m.textContent = 'Asking Nemotron...'; $('#explain').disabled = true;
  try {
    const controls = Object.keys(run.results).filter((k) => k !== 'naive');
    const j = await (await post('/api/explain', { datasetId: ds.datasetId, controls })).json();
    if (!j.ok) { m.className = 'msg err'; m.textContent = j.message || j.error; return; }
    explanation = j.text; m.textContent = '';
    const c = j.check, ok = c.unmatched.length === 0;
    $('#explainOut').hidden = false;
    $('#explainOut').innerHTML = `${esc(j.text)}<small class="${ok ? '' : 'bad'}">${j.model.split('/')[1]}, ${(j.ms / 1000).toFixed(1)} s. Figures checked against the control table: ${c.matched} of ${c.total} match${ok ? '' : `; unmatched: ${c.unmatched.join(', ')}`}.</small>`;
  } catch (e) { m.className = 'msg err'; m.textContent = 'Request failed.'; } finally { $('#explain').disabled = false; }
});

$('#dl').addEventListener('click', async () => {
  const controls = Object.keys(run.results).filter((k) => k !== 'naive');
  const r = await post('/api/report', { datasetId: ds.datasetId, controls, narrative: explanation });
  const a = document.createElement('a'); a.href = URL.createObjectURL(await r.blob()); a.download = 'nullgarden-report.md'; a.click();
});

$('#csvFile').addEventListener('change', async (e) => {
  const f = e.target.files[0]; if (!f) return;
  document.querySelectorAll('#scenarios button').forEach((b) => b.setAttribute('aria-selected', 'false'));
  await setDataset({ csv: await f.text(), name: f.name }); e.target.value = '';
});
init();

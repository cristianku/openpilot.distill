// distill-web: the page. Vanilla JS, no libraries; the DOM is built with textContent only.
(function () {
  'use strict';

  const TABS = ['stato', 'dati', 'training', 'valutazione', 'confronto', 'pubblica', 'comma', 'job'];
  const JOB_PILL = {running: 'run', ok: 'ok', failed: 'bad', stopped: 'todo', unknown: 'muted'};
  const RUN_STATE = {running: ['in corso', 'run'], finished: ['finito', 'ok'], interrupted: ['interrotto', 'todo']};
  const SEGMENT_STATE = {localized: ['localizzato', 'ok'], failed: ['fallito', 'bad'], todo: ['da localizzare', 'todo']};
  const CHART_TITLES = {'loss': 'loss', 'gt': 'gt (la tua guida)', 'distill': 'distill (il teacher)',
    'gt/plan': 'gt/plan (traiettoria)', 'grad_norm': 'grad_norm', 'step_time': 'tempo per step (s)'};
  // How often (ms) the data of each tab is read again while the tab is open.
  const TAB_PERIOD = {dati: 15000, training: 10000, valutazione: 30000, confronto: 30000, pubblica: 15000,
    comma: 0, job: 0, stato: 0};
  const STATUS_PERIOD = 4000;
  const SMOOTHING = 0.85;

  let status = null;
  let firstCode = null;
  let currentTab = 'stato';
  const data = {segments: null, runs: null, evalResults: null, videos: null, hf: null, metrics: null};
  const loadedAt = {};
  let selectedRun = null;
  let previousRunning = '';
  let log = {id: null, offset: 0, timer: null, follow: true};

  // -- small helpers ---------------------------------------------------------------------------
  const byId = (id) => document.getElementById(id);
  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = text;
    return element;
  }
  function pill(label, kind) { return node('span', 'pill ' + kind, label); }
  function setBar(id, percent, kind) {
    const bar = byId(id);
    const value = Math.max(0, Math.min(100, percent || 0));
    bar.style.width = value + '%';
    bar.parentElement.className = 'bar' + (kind ? ' ' + kind : '');
  }
  function barNode(percent, kind) {
    const bar = node('div', 'bar' + (kind ? ' ' + kind : ''));
    const fill = node('span');
    fill.style.width = Math.max(0, Math.min(100, percent || 0)) + '%';
    bar.appendChild(fill);
    return bar;
  }
  function fmt(value) {
    if (value === null || value === undefined || !isFinite(value)) return '–';
    const size = Math.abs(value);
    if (size >= 1000) return value.toFixed(0);
    if (size >= 100) return value.toFixed(1);
    if (size >= 10) return value.toFixed(2);
    if (size >= 1) return value.toFixed(3);
    if (size === 0) return '0';
    return value.toPrecision(3);
  }
  function fmtTime(unix) {
    if (unix === null || unix === undefined) return '–';
    let date = null;
    if (typeof unix === 'string') date = new Date(unix);
    else date = new Date(unix * 1000);
    if (isNaN(date.getTime())) return String(unix);
    return date.toLocaleString('it-IT', {day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit'});
  }
  function fmtDuration(seconds) {
    if (seconds === null || seconds === undefined || !isFinite(seconds)) return '–';
    const total = Math.round(seconds);
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    if (hours > 0) return hours + ' h ' + minutes + ' min';
    if (minutes > 0) return minutes + ' min';
    return total + ' s';
  }
  function fmtBytes(bytes) {
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + ' GB';
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
    return Math.round(bytes / 1e3) + ' kB';
  }
  function same(key, value) {
    // True when `value` is what was rendered last time under `key`: rebuilding the same DOM would
    // lose a click landing at that moment and the keyboard focus.
    const text = JSON.stringify(value);
    if (same.cache[key] === text) return true;
    same.cache[key] = text;
    return false;
  }
  same.cache = {};

  async function api(path, options) {
    const response = await fetch(path, Object.assign({credentials: 'same-origin', cache: 'no-store'}, options || {}));
    let body = null;
    try { body = await response.json(); } catch (error) { body = {message: 'Risposta non valida (HTTP ' + response.status + ').'}; }
    if (!response.ok) throw new Error(body.message || ('HTTP ' + response.status));
    return body;
  }
  function post(path, payload) {
    return api(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload || {})});
  }

  let toastTimer = null;
  function toast(message, error) {
    const element = byId('toast');
    element.textContent = message;
    element.className = 'toast show' + (error ? ' error' : '');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { element.className = 'toast'; }, error ? 8000 : 3500);
  }

  // Replaces the options of a select only when they changed, keeping the chosen value when it is
  // still there. options: [{value, label, group?}]
  function fillSelect(select, options, preferred) {
    const key = 'select:' + select.id;
    if (same(key, options)) return;
    const previous = select.value;
    select.replaceChildren();
    const groups = {};
    for (const option of options) {
      const element = node('option', '', option.label);
      element.value = option.value;
      if (option.group) {
        if (!groups[option.group]) {
          groups[option.group] = document.createElement('optgroup');
          groups[option.group].label = option.group;
          select.appendChild(groups[option.group]);
        }
        groups[option.group].appendChild(element);
      } else {
        select.appendChild(element);
      }
    }
    const values = options.map((option) => option.value);
    if (previous && values.indexOf(previous) >= 0) select.value = previous;
    else if (preferred && values.indexOf(preferred) >= 0) select.value = preferred;
    else if (values.length) select.value = values[0];
  }

  // -- tabs ------------------------------------------------------------------------------------
  function showTab(name) {
    if (TABS.indexOf(name) < 0) name = 'stato';
    currentTab = name;
    for (const tab of TABS) {
      const button = byId('tab-' + tab);
      const selected = tab === name;
      button.setAttribute('aria-selected', String(selected));
      button.tabIndex = selected ? 0 : -1;
      byId('panel-' + tab).hidden = !selected;
    }
    if (location.hash !== '#' + name) history.replaceState(null, '', '#' + name);
    loadTab(true);
  }
  for (const tab of TABS) {
    byId('tab-' + tab).addEventListener('click', () => showTab(tab));
    byId('tab-' + tab).addEventListener('keydown', (event) => {
      let index = TABS.indexOf(tab);
      if (event.key === 'ArrowRight') index = (index + 1) % TABS.length;
      else if (event.key === 'ArrowLeft') index = (index + TABS.length - 1) % TABS.length;
      else return;
      event.preventDefault();
      showTab(TABS[index]);
      byId('tab-' + TABS[index]).focus();
    });
  }
  window.addEventListener('hashchange', () => showTab(location.hash.slice(1)));

  // Which data each tab needs.
  function needs(tab) {
    if (tab === 'dati') return ['segments'];
    if (tab === 'training') return ['runs', 'hf'];
    if (tab === 'valutazione') return ['runs', 'hf', 'evalResults'];
    if (tab === 'confronto') return ['segments', 'runs', 'hf', 'videos'];
    if (tab === 'pubblica') return ['runs', 'hf'];
    if (tab === 'comma') return ['runs'];
    return [];
  }

  async function load(what) {
    try {
      if (what === 'segments') data.segments = await api('/api/data');
      else if (what === 'runs') data.runs = (await api('/api/runs')).runs;
      else if (what === 'evalResults') data.evalResults = (await api('/api/eval')).results;
      else if (what === 'videos') data.videos = (await api('/api/compare')).videos;
      else if (what === 'hf') data.hf = await api('/api/hf');
      loadedAt[what] = Date.now();
    } catch (error) {
      toast(error.message, true);
    }
  }

  async function loadTab(force) {
    const tab = currentTab;
    const period = TAB_PERIOD[tab] || 0;
    const wanted = needs(tab);
    const pending = [];
    for (const what of wanted) {
      const age = Date.now() - (loadedAt[what] || 0);
      // The HF list is cached on the server for 5 minutes: no need to ask more often.
      const limit = what === 'hf' ? 60000 : period;
      if (force || !loadedAt[what] || (limit && age >= limit)) pending.push(load(what));
    }
    await Promise.all(pending);
    if (tab === 'training') await loadMetrics(false);
    renderTab();
  }

  function renderTab() {
    if (currentTab === 'dati') renderData();
    else if (currentTab === 'training') renderTraining();
    else if (currentTab === 'valutazione') renderEval();
    else if (currentTab === 'confronto') renderCompare();
    else if (currentTab === 'pubblica') renderPublish();
    else if (currentTab === 'comma') renderCommaTab();
    updateButtons();
  }

  // -- status ----------------------------------------------------------------------------------
  async function refreshStatus() {
    try {
      status = await api('/api/status');
      byId('live-dot').className = 'dot';
      byId('live-text').textContent = 'aggiornato ' + new Date().toLocaleTimeString('it-IT');
    } catch (error) {
      byId('live-dot').className = 'dot off';
      byId('live-text').textContent = 'non raggiungibile: ' + error.message;
      return;
    }
    if (firstCode === null) firstCode = status.code;
    byId('code-banner').hidden = !(status.code && firstCode && status.code !== firstCode);
    byId('code-version').textContent = status.code || 'sconosciuta';
    byId('selftest-box').hidden = !status.test_actions;
    renderMachine();
    renderRunning();
    renderComma('comma-kv-stato');
    renderTrainingNow();
    renderJobs();
    // A job just ended: what it produced shows up in the open tab at once.
    const running = status.jobs.filter((job) => job.state === 'running').map((job) => job.id).join(',');
    if (previousRunning && running !== previousRunning) loadTab(true);
    else renderTab();
    previousRunning = running;
    if (log.id) {
      const job = status.jobs.find((item) => item.id === log.id);
      byId('term-state').hidden = !(job && job.state === 'running');
    }
  }

  function renderMachine() {
    const system = status.system;
    const gpu = system.gpu;
    if (gpu) {
      const util = gpu.utilization === null ? 0 : gpu.utilization;
      byId('gpu-value').textContent = util + '%';
      setBar('gpu-bar', util, util > 85 ? 'warn' : '');
      let text = gpu.name;
      if (gpu.memory_total_mb) text += ' · ' + (gpu.memory_used_mb / 1024).toFixed(1) + ' / ' + (gpu.memory_total_mb / 1024).toFixed(0) + ' GB';
      if (gpu.temperature !== null) text += ' · ' + gpu.temperature + ' °C';
      if (gpu.power_w !== null) text += ' · ' + Math.round(gpu.power_w) + ' W';
      text += gpu.processes.length ? ' · ' + gpu.processes.length + ' processi' : ' · libera';
      byId('gpu-sub').textContent = text;
    } else {
      byId('gpu-value').textContent = 'n/d';
      setBar('gpu-bar', 0, '');
      byId('gpu-sub').textContent = 'nvidia-smi non disponibile';
    }
    if (system.cpus) {
      const load = system.load ? system.load[0] : 0;
      byId('cpu-value').textContent = system.cpus + ' core';
      setBar('cpu-bar', load * 100 / system.cpus, load > system.cpus ? 'warn' : '');
      byId('cpu-sub').textContent = system.load ? 'carico ' + system.load.join(' · ') : 'carico n/d';
    }
    const memory = system.memory;
    if (memory) {
      byId('ram-value').textContent = memory.available_gb + ' GB liberi';
      setBar('ram-bar', memory.used_percent, memory.used_percent > 90 ? 'bad' : (memory.used_percent > 75 ? 'warn' : ''));
      byId('ram-sub').textContent = 'su ' + memory.total_gb + ' GB · usata ' + memory.used_percent + '%';
    } else {
      byId('ram-value').textContent = 'n/d';
    }
    const disk = system.disk;
    if (disk) {
      const used = 100 - disk.free_gb * 100 / disk.total_gb;
      byId('disk-value').textContent = disk.free_gb + ' GB liberi';
      setBar('disk-bar', used, disk.free_gb < 10 ? 'bad' : (used > 80 ? 'warn' : ''));
      byId('disk-sub').textContent = 'su ' + disk.total_gb + ' GB';
    }
    if (same('gpu-procs', gpu ? gpu.processes : [])) return;
    const body = byId('gpu-procs');
    body.replaceChildren();
    const processes = gpu ? gpu.processes : [];
    byId('gpu-procs-empty').hidden = processes.length > 0;
    for (const process of processes) {
      const row = node('tr');
      row.appendChild(node('td', 'mono', process.pid === null ? '?' : String(process.pid)));
      row.appendChild(node('td', '', process.name));
      row.appendChild(node('td', 'num', process.memory_mb === null ? '–' : Math.round(process.memory_mb) + ' MiB'));
      row.appendChild(node('td', 'mono', process.command || '–'));
      body.appendChild(row);
    }
  }

  function jobItem(job, withButtons) {
    const item = node('li', 'item');
    item.appendChild(pill(job.state_label, JOB_PILL[job.state] || 'muted'));
    const text = node('div', 'grow');
    text.appendChild(node('div', 'what', job.label));
    let sub = job.summary || '';
    sub += (sub ? ' · ' : '') + fmtTime(job.started);
    if (job.state === 'running') sub += ' · da ' + fmtDuration(Date.now() / 1000 - job.started);
    else if (job.finished) sub += ' · durata ' + fmtDuration(job.finished - job.started);
    if (job.exit_code !== null && job.exit_code !== undefined) sub += ' · codice ' + job.exit_code;
    text.appendChild(node('div', 'sub', sub));
    item.appendChild(text);
    if (withButtons) {
      if (job.run) {
        const charts = node('button', 'btn small', 'Grafici');
        charts.type = 'button';
        charts.addEventListener('click', () => { selectedRun = job.run; showTab('training'); });
        item.appendChild(charts);
      }
      if (job.video) {
        const play = node('button', 'btn small', 'Guarda');
        play.type = 'button';
        play.addEventListener('click', () => { showTab('confronto'); playVideo(job.video); });
        item.appendChild(play);
      }
      const view = node('button', 'btn small', 'Log');
      view.type = 'button';
      view.addEventListener('click', () => openLog(job.id, job.label));
      item.appendChild(view);
      if (job.stoppable) {
        const stop = node('button', 'btn small danger', 'Ferma');
        stop.type = 'button';
        stop.addEventListener('click', () => stopJob(job));
        item.appendChild(stop);
      }
    }
    if (job.progress) {
      const progress = node('div', 'progress');
      progress.appendChild(barNode(job.progress.percent, 'run'));
      progress.appendChild(node('span', 'sub', job.progress.done + ' / ' + job.progress.total));
      item.appendChild(progress);
    }
    return item;
  }

  function renderRunning() {
    const running = status.jobs.filter((job) => job.state === 'running');
    const count = byId('running-count');
    count.hidden = running.length === 0;
    count.textContent = String(running.length);
    // The duration text changes every second: rebuilt at every status.
    const list = byId('running-jobs');
    list.replaceChildren();
    byId('running-empty').hidden = running.length > 0;
    for (const job of running) list.appendChild(jobItem(job, true));

    if (same('activities', status.activities)) return;
    const activities = byId('activities');
    activities.replaceChildren();
    byId('activities-empty').hidden = status.activities.length > 0;
    for (const activity of status.activities) {
      const item = node('li', 'item');
      item.appendChild(pill(activity.label, activity.group === 'other' ? 'imp' : 'run'));
      item.appendChild(node('span', 'sub', 'pid ' + activity.pid + (activity.uses_gpu ? ' · GPU' : '')));
      item.appendChild(node('div', 'cmd', activity.command));
      activities.appendChild(item);
    }
  }

  function renderComma(target) {
    const comma = status.comma;
    const list = byId(target);
    list.replaceChildren();
    function row(label, value, kind) {
      list.appendChild(node('dt', '', label));
      const cell = node('dd');
      if (kind) cell.appendChild(pill(value, kind));
      else cell.textContent = value;
      list.appendChild(cell);
    }
    row('Host', comma.host);
    if (comma.reachable === null) row('Raggiungibile', 'non ancora controllato', 'muted');
    else if (comma.reachable) row('Raggiungibile', 'sì', 'ok');
    else row('Raggiungibile', 'no', 'bad');
    if (comma.reachable) {
      if (comma.onroad === true) row('Stato', 'onroad (in marcia)', 'todo');
      else row('Stato', 'offroad (fermo)', 'ok');
      if (comma.bundle_default === false) row('Modello scelto nel menu Modelli', comma.bundle, 'imp');
      else row('Menu Modelli', 'Default (il modello di openpilot)');
      row('Copia dell\'originale', comma.backup ? 'presente (/data/driving_supercombo.onnx.orig)' : 'nessuna');
    } else if (comma.error) {
      row('Errore', comma.error);
    }
    row('Controllato', comma.checked ? fmtTime(comma.checked) + ' (ogni 30 s mentre la pagina è aperta)' : '–');
  }

  function renderTrainingNow() {
    const card = byId('training-card');
    const box = byId('training-now');
    card.hidden = status.training.length === 0;
    box.replaceChildren();
    for (const run of status.training) {
      const item = node('div', 'item');
      item.appendChild(pill('in corso', 'run'));
      const text = node('div', 'grow');
      text.appendChild(node('div', 'what', run.name));
      let sub = 'step ' + run.steps_done + (run.steps_total ? ' / ' + run.steps_total : '');
      if (run.eta_seconds !== null) sub += ' · mancano ' + fmtDuration(run.eta_seconds);
      if (run.avg_step_time !== null) sub += ' · ' + run.avg_step_time.toFixed(2) + ' s/step';
      if (run.last && run.last.loss !== undefined) sub += ' · loss ' + fmt(run.last.loss);
      text.appendChild(node('div', 'sub', sub));
      item.appendChild(text);
      const charts = node('button', 'btn small', 'Grafici');
      charts.type = 'button';
      charts.addEventListener('click', () => { selectedRun = run.name; showTab('training'); });
      item.appendChild(charts);
      const progress = node('div', 'progress');
      progress.appendChild(barNode(run.percent, 'run'));
      progress.appendChild(node('span', 'sub', (run.percent === null ? '?' : run.percent) + '%'));
      item.appendChild(progress);
      box.appendChild(item);
    }
  }

  function renderJobs() {
    const list = byId('jobs');
    list.replaceChildren();
    byId('jobs-empty').hidden = status.jobs.length > 0;
    for (const job of status.jobs) list.appendChild(jobItem(job, true));
  }

  // -- action buttons --------------------------------------------------------------------------
  function actionReason(name) {
    if (!status || !status.actions[name]) return 'stato del server non ancora letto';
    const state = status.actions[name];
    return state.allowed ? '' : state.reason;
  }
  // extra: a reason of the form (e.g. confirmation not typed), shown when the server allows the action.
  function setAction(buttonId, name, extra) {
    const reason = actionReason(name) || extra || '';
    byId(buttonId).disabled = reason !== '';
    const element = byId('reason-' + name);
    if (element) element.textContent = reason;
  }

  async function runAction(name, params, question) {
    if (question && !window.confirm(question)) return null;
    try {
      const answer = await post('/api/run/' + name, params);
      toast('Partito: ' + answer.job.label);
      openLog(answer.job.id, answer.job.label);
      await refreshStatus();
      return answer.job;
    } catch (error) {
      toast(error.message, true);
      refreshStatus();
      return null;
    }
  }

  async function stopJob(job) {
    if (!window.confirm('Fermo "' + job.label + '"? Il job riceve SIGTERM, dopo 20 s SIGKILL.')) return;
    try {
      await post('/api/jobs/' + job.id + '/stop', {});
      toast('Fermo: ' + job.label);
    } catch (error) {
      toast(error.message, true);
    }
    refreshStatus();
  }

  function updateButtons() {
    if (!status) return;
    setAction('btn-fetch', 'fetch');
    setAction('btn-localize', 'localize');
    updateTrainButtons();
    setAction('btn-eval', 'eval', byId('eval-model').value ? '' : 'nessun modello da valutare');
    setAction('btn-compare', 'compare', compareFormReason());
    setAction('btn-export', 'export', byId('export-run').value ? '' : 'nessun run finito');
    updatePublishButtons();
    const installRun = byId('install-run').value;
    let installReason = '';
    if (!installRun) installReason = 'nessun run finito';
    else if (byId('install-confirm').value !== 'INSTALLA') installReason = 'scrivi INSTALLA per confermare';
    setAction('btn-install', 'install', installReason);
    setAction('btn-restore', 'restore', byId('restore-confirm').value !== 'RIPRISTINA' ? 'scrivi RIPRISTINA per confermare' : '');
    setAction('btn-update', 'update');
    if (status.test_actions) setAction('btn-selftest', 'selftest');
  }

  // -- Dati ------------------------------------------------------------------------------------
  function renderData() {
    const segments = data.segments;
    if (!segments || same('data', segments)) return;
    const counts = segments.counts;
    const tiles = byId('data-tiles');
    tiles.replaceChildren();
    const items = [[counts.total, 'segmenti in data/'], [counts.localized, 'localizzati'], [counts.failed, 'falliti'],
      [counts.todo, 'da localizzare'], [counts.heldout_localized, 'hold-out pronti'], [counts.discarded, 'scartati']];
    for (const [value, label] of items) {
      const tile = node('div', 'tile');
      tile.appendChild(node('div', 'big', String(value)));
      tile.appendChild(node('div', 'what', label));
      tiles.appendChild(tile);
    }
    const routes = byId('routes');
    routes.replaceChildren();
    byId('routes-empty').hidden = segments.routes.length > 0;
    for (const route of segments.routes) {
      const row = node('tr');
      row.appendChild(node('td', 'mono', route.route));
      for (const value of [route.segments, route.localized, route.failed, route.todo, route.heldout]) {
        row.appendChild(node('td', 'num', String(value)));
      }
      routes.appendChild(row);
    }
    const discarded = byId('discarded');
    discarded.replaceChildren();
    byId('discarded-empty').hidden = segments.discarded_reasons.length > 0;
    for (const reason of segments.discarded_reasons) {
      const item = node('li', 'item');
      item.appendChild(node('span', 'what', String(reason.count)));
      item.appendChild(node('span', 'grow', reason.reason));
      discarded.appendChild(item);
    }
    const body = byId('segments');
    body.replaceChildren();
    for (const segment of segments.segments) {
      const row = node('tr');
      row.appendChild(node('td', 'mono', segment.name));
      const state = SEGMENT_STATE[segment.state] || [segment.state, 'muted'];
      const cell = node('td');
      cell.appendChild(pill(state[0], state[1]));
      if (segment.heldout) {
        cell.appendChild(document.createTextNode(' '));
        cell.appendChild(pill('hold-out', 'imp'));
      }
      row.appendChild(cell);
      row.appendChild(node('td', 'mono', segment.map || '–'));
      let note = segment.reason || '';
      if (segment.broken_video) note = 'video mancante nella cartella comune. ' + note;
      row.appendChild(node('td', 'note', note));
      body.appendChild(row);
    }
  }
  byId('btn-fetch').addEventListener('click', () => {
    const since = byId('fetch-since').value;
    const params = since ? {since: since} : {};
    runAction('fetch', params);
  });
  byId('btn-localize').addEventListener('click', () => runAction('localize', {}));

  // -- Training --------------------------------------------------------------------------------
  function runsWith(filter) { return (data.runs || []).filter(filter); }

  function trainMode() {
    const checked = document.querySelector('input[name="train-mode"]:checked');
    return checked ? checked.value : 'fresh';
  }

  function updateTrainButtons() {
    const mode = trainMode();
    byId('train-steps').disabled = mode === 'resume';
    byId('train-resume-run').disabled = mode !== 'resume';
    let extra = '';
    if (mode === 'resume' && !byId('train-resume-run').value) extra = 'nessun run da riprendere';
    if (mode === 'continue' && data.hf && data.hf.ok && data.hf.stable === false) extra = 'su Hugging Face non c\'è training/stable.json';
    setAction('btn-train', 'train', extra);
    setAction('btn-train-test', 'train_test');
  }

  function numberOrText(value) {
    // A number when the field holds one; otherwise the text itself, so the server says what is wrong.
    if (value === '') return '';
    const number = Number(value);
    return isFinite(number) ? number : value;
  }

  byId('btn-train').addEventListener('click', () => {
    const mode = trainMode();
    const params = {mode: mode, batch_size: numberOrText(byId('train-batch').value),
      workers: numberOrText(byId('train-workers').value), wandb: byId('train-wandb').value};
    let question = 'Training da zero di ' + byId('train-steps').value + ' step: la GPU resta occupata per ore. Avvio?';
    if (mode === 'resume') {
      params.run = byId('train-resume-run').value;
      question = 'Riprendo il run ' + params.run + ' dall\'ultimo checkpoint. Avvio?';
    } else {
      params.steps = numberOrText(byId('train-steps').value);
      if (mode === 'continue') question = 'Training di ' + params.steps + ' step dai pesi stable: la GPU resta occupata per ore. Avvio?';
    }
    runAction('train', params, question);
  });
  byId('btn-train-test').addEventListener('click', () => runAction('train_test', {}));
  for (const radio of document.querySelectorAll('input[name="train-mode"]')) radio.addEventListener('change', updateButtons);
  byId('train-resume-run').addEventListener('change', () => {
    const run = (data.runs || []).find((item) => item.name === byId('train-resume-run').value);
    if (run && run.config) {
      if (run.config.batch_size) byId('train-batch').value = run.config.batch_size;
      if (run.config.workers !== undefined) byId('train-workers').value = run.config.workers;
    }
    updateButtons();
  });

  function renderTraining() {
    const resumable = runsWith((run) => run.resumable);
    const options = resumable.map((run) => ({value: run.name, label: run.name + ' (' + run.steps_done + (run.steps_total ? '/' + run.steps_total : '') + ' step)'}));
    fillSelect(byId('train-resume-run'), options);
    byId('mode-resume').disabled = resumable.length === 0;
    let note = '';
    if (data.hf && data.hf.ok && data.hf.stable === false) note = 'Su Hugging Face non c\'è ancora training/stable.json: pubblica prima un modello con «Stable».';
    else if (data.hf && !data.hf.ok) note = 'Non so se c\'è training/stable.json (' + data.hf.error + ').';
    byId('continue-note').textContent = note;

    const runs = data.runs || [];
    if (!selectedRun || !runs.find((run) => run.name === selectedRun)) {
      const running = runs.find((run) => run.status === 'running');
      selectedRun = running ? running.name : (runs.length ? runs[0].name : null);
    }
    if (!same('runs-table', [runs, selectedRun])) {
      const body = byId('runs');
      body.replaceChildren();
      byId('runs-empty').hidden = runs.length > 0;
      for (const run of runs) {
        const row = node('tr', run.name === selectedRun ? 'selected' : '');
        row.appendChild(node('td', 'mono', run.name));
        const state = RUN_STATE[run.status] || [run.status, 'muted'];
        const stateCell = node('td');
        stateCell.appendChild(pill(state[0], state[1]));
        row.appendChild(stateCell);
        const stepsCell = node('td');
        const steps = node('div', 'steps-cell');
        steps.appendChild(barNode(run.percent, run.status === 'running' ? 'run' : (run.status === 'finished' ? '' : 'warn')));
        steps.appendChild(node('span', 'note', run.steps_done + (run.steps_total ? ' / ' + run.steps_total : '')));
        stepsCell.appendChild(steps);
        row.appendChild(stepsCell);
        row.appendChild(node('td', 'num', run.status === 'running' ? fmtDuration(run.eta_seconds) : '–'));
        for (const key of ['loss', 'gt', 'distill', 'gt/plan']) {
          row.appendChild(node('td', 'num', run.last ? fmt(run.last[key]) : '–'));
        }
        const files = [];
        if (run.has_pt) files.push('pt');
        if (run.has_onnx) files.push('onnx');
        if (run.has_training_pt && !run.has_pt) files.push('checkpoint');
        row.appendChild(node('td', 'note', files.join(' · ') || '–'));
        const buttonCell = node('td');
        const button = node('button', 'btn small', 'Grafici');
        button.type = 'button';
        button.addEventListener('click', () => { selectedRun = run.name; same.cache['runs-table'] = ''; renderTraining(); loadMetrics(true); });
        buttonCell.appendChild(button);
        row.appendChild(buttonCell);
        body.appendChild(row);
      }
    }
    renderCharts();
  }

  async function loadMetrics(force) {
    if (!selectedRun) {
      const runs = data.runs || [];
      const running = runs.find((run) => run.status === 'running');
      selectedRun = running ? running.name : (runs.length ? runs[0].name : null);
    }
    if (!selectedRun) return;
    const run = (data.runs || []).find((item) => item.name === selectedRun);
    const stale = !data.metrics || data.metrics.name !== selectedRun;
    // A finished run does not change: read it once; a running one at every tab refresh.
    if (!force && !stale && !(run && run.status === 'running')) return;
    try {
      const width = Math.max(200, Math.min(1500, Math.round(window.innerWidth)));
      data.metrics = await api('/api/runs/' + encodeURIComponent(selectedRun) + '/metrics?points=' + width);
    } catch (error) {
      toast(error.message, true);
    }
    renderCharts();
  }

  byId('chart-smooth').addEventListener('change', () => { same.cache.charts = ''; renderCharts(); });
  byId('chart-log').addEventListener('change', () => { same.cache.charts = ''; renderCharts(); });

  function renderCharts() {
    byId('chart-run').textContent = selectedRun || '–';
    const metrics = data.metrics;
    const smooth = byId('chart-smooth').checked;
    const logScale = byId('chart-log').checked;
    if (!metrics || metrics.name !== selectedRun) return;
    if (same('charts', [metrics, smooth, logScale])) return;
    const box = byId('charts');
    box.replaceChildren();
    byId('charts-empty').hidden = metrics.steps.length > 0;
    let note = 'fonte: ' + (metrics.source || '–');
    if (metrics.bucket > 1) note += ' · ogni punto è la media di ' + metrics.bucket + ' step';
    byId('chart-note').textContent = note;
    if (!metrics.steps.length) return;
    for (const key of metrics.available) {
      box.appendChild(chart(CHART_TITLES[key] || key, metrics.steps, metrics.series[key], smooth, logScale));
    }
  }

  // One line chart (one series): faint raw values, the smoothed line on top when asked,
  // a crosshair with the value under the pointer (or the arrow keys).
  function chart(title, steps, values, smooth, logScale) {
    const SVG = 'http://www.w3.org/2000/svg';
    const width = 480;
    const height = 220;
    const left = 52;
    const right = 12;
    const top = 10;
    const bottom = 26;
    const box = node('div', 'chart');
    const head = node('div', 'chart-title');
    head.appendChild(node('span', 'name', title));
    const lastText = node('span', 'last');
    head.appendChild(lastText);
    box.appendChild(head);

    const points = [];
    for (let index = 0; index < steps.length; index++) {
      if (values[index] !== null && values[index] !== undefined) points.push([steps[index], values[index]]);
    }
    if (!points.length) {
      box.appendChild(node('p', 'empty', 'nessun valore'));
      return box;
    }
    let useLog = logScale;
    for (const point of points) if (point[1] <= 0) useLog = false;
    let smoothed = null;
    if (smooth) {
      smoothed = [];
      let running = points[0][1];
      for (const point of points) {
        running = SMOOTHING * running + (1 - SMOOTHING) * point[1];
        smoothed.push([point[0], running]);
      }
    }
    const shown = smoothed || points;
    lastText.textContent = 'ultimo ' + fmt(shown[shown.length - 1][1]);

    const transform = (value) => (useLog ? Math.log10(value) : value);
    let low = Infinity;
    let high = -Infinity;
    for (const point of points) { low = Math.min(low, transform(point[1])); high = Math.max(high, transform(point[1])); }
    if (low === high) { low -= 1; high += 1; }
    const pad = (high - low) * 0.05;
    low -= pad;
    high += pad;
    const firstStep = points[0][0];
    const lastStep = points[points.length - 1][0];
    const spanSteps = Math.max(1, lastStep - firstStep);
    const xOf = (step) => left + (step - firstStep) * (width - left - right) / spanSteps;
    const yOf = (value) => top + (high - transform(value)) * (height - top - bottom) / (high - low);

    const svg = document.createElementNS(SVG, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', title + ', ultimo valore ' + fmt(shown[shown.length - 1][1]));
    svg.setAttribute('tabindex', '0');
    function element(tag, attributes, className) {
      const item = document.createElementNS(SVG, tag);
      for (const name in attributes) item.setAttribute(name, attributes[name]);
      if (className) item.setAttribute('class', className);
      svg.appendChild(item);
      return item;
    }
    // y ticks: 4 levels evenly spaced on the (log) scale, labelled with the real value.
    for (let tick = 0; tick <= 3; tick++) {
      const level = low + (high - low) * tick / 3;
      const value = useLog ? Math.pow(10, level) : level;
      const y = yOf(value);
      element('line', {x1: left, x2: width - right, y1: y, y2: y}, 'gridline');
      const label = element('text', {x: left - 6, y: y + 3.5, 'text-anchor': 'end'}, 'axis-label');
      label.textContent = fmt(value);
    }
    element('line', {x1: left, x2: width - right, y1: height - bottom, y2: height - bottom}, 'baseline');
    for (let tick = 0; tick <= 3; tick++) {
      const step = Math.round(firstStep + spanSteps * tick / 3);
      const label = element('text', {x: xOf(step), y: height - 8, 'text-anchor': tick === 0 ? 'start' : (tick === 3 ? 'end' : 'middle')}, 'axis-label');
      label.textContent = String(step);
    }
    function pathOf(list) {
      let text = '';
      for (let index = 0; index < list.length; index++) {
        text += (index === 0 ? 'M' : 'L') + xOf(list[index][0]).toFixed(1) + ' ' + yOf(list[index][1]).toFixed(1);
      }
      return text;
    }
    element('path', {d: pathOf(points)}, smoothed ? 'line raw' : 'line');
    if (smoothed) element('path', {d: pathOf(smoothed)}, 'line');

    const cross = element('line', {x1: 0, x2: 0, y1: top, y2: height - bottom}, 'cross');
    const marker = element('circle', {cx: 0, cy: 0, r: 4.5}, 'marker');
    cross.setAttribute('visibility', 'hidden');
    marker.setAttribute('visibility', 'hidden');
    const hit = element('rect', {x: left, y: top, width: width - left - right, height: height - top - bottom}, 'hit');
    const tip = node('div', 'tooltip');
    tip.hidden = true;
    box.appendChild(svg);
    box.appendChild(tip);

    let current = -1;
    function show(index) {
      current = Math.max(0, Math.min(shown.length - 1, index));
      const point = shown[current];
      const x = xOf(point[0]);
      cross.setAttribute('x1', x);
      cross.setAttribute('x2', x);
      marker.setAttribute('cx', x);
      marker.setAttribute('cy', yOf(point[1]));
      cross.setAttribute('visibility', 'visible');
      marker.setAttribute('visibility', 'visible');
      tip.replaceChildren(node('strong', '', fmt(point[1])), node('span', '', 'step ' + point[0]));
      tip.hidden = false;
      const fraction = (x / width);
      const pixels = fraction * svg.getBoundingClientRect().width;
      tip.style.left = Math.max(4, Math.min(box.clientWidth - tip.offsetWidth - 4, pixels + 12)) + 'px';
    }
    function hide() {
      cross.setAttribute('visibility', 'hidden');
      marker.setAttribute('visibility', 'hidden');
      tip.hidden = true;
    }
    function nearest(step) {
      let lo = 0;
      let hi = shown.length - 1;
      while (hi - lo > 1) {
        const middle = (lo + hi) >> 1;
        if (shown[middle][0] < step) lo = middle; else hi = middle;
      }
      return Math.abs(shown[lo][0] - step) <= Math.abs(shown[hi][0] - step) ? lo : hi;
    }
    hit.addEventListener('pointermove', (event) => {
      const rect = svg.getBoundingClientRect();
      const x = (event.clientX - rect.left) * width / rect.width;
      const step = firstStep + (x - left) * spanSteps / (width - left - right);
      show(nearest(step));
    });
    hit.addEventListener('pointerleave', hide);
    svg.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowRight') { event.preventDefault(); show(current < 0 ? shown.length - 1 : current + 1); }
      else if (event.key === 'ArrowLeft') { event.preventDefault(); show(current < 0 ? shown.length - 1 : current - 1); }
      else if (event.key === 'Escape') hide();
    });
    svg.addEventListener('blur', hide);
    return box;
  }

  // -- model lists (evaluation, compare) -----------------------------------------------------------
  function hfModelOptions() {
    const options = [];
    if (data.hf && data.hf.ok) {
      for (const name of data.hf.models) {
        if (/^[a-z0-9_]+$/.test(name)) options.push({value: 'hf:' + name, label: 'HF ' + name, group: 'Pubblicati su Hugging Face'});
      }
    }
    return options;
  }
  function runModelOptions() {
    const options = [];
    for (const run of runsWith((item) => item.has_pt || item.has_training_pt)) {
      options.push({value: 'run:' + run.name, label: 'run ' + run.name + (run.has_pt ? '' : ' (checkpoint, non finito)'), group: 'Run locali'});
    }
    return options;
  }

  // -- Valutazione -------------------------------------------------------------------------------
  function renderEval() {
    const options = [{value: 'stock', label: 'Stock CD210 (il modello del comma)', group: 'Riferimenti'},
      {value: 'teacher', label: 'Teacher (il modello grande)', group: 'Riferimenti'},
      {value: 'logged', label: 'Logged (quello che ha guidato nel giro)', group: 'Riferimenti'}]
      .concat(runModelOptions(), hfModelOptions());
    const firstRun = runModelOptions();
    fillSelect(byId('eval-model'), options, firstRun.length ? firstRun[0].value : 'stock');
    const results = data.evalResults || [];
    if (same('leaderboard', results)) return;
    const body = byId('leaderboard');
    body.replaceChildren();
    byId('leaderboard-empty').hidden = results.length > 0;
    results.forEach((result, index) => {
      const row = node('tr', index === 0 && result.avg_miss_m !== null ? 'best' : '');
      row.appendChild(node('td', 'num', String(index + 1)));
      const modelCell = node('td');
      modelCell.appendChild(node('div', 'mono', result.model));
      let note = result.label && result.label !== result.model ? result.label : '';
      if (result.selection && result.selection !== 'heldout') note += (note ? ' · ' : '') + 'segmenti: ' + result.selection;
      if (result.rate_hz && result.rate_hz !== 5) note += (note ? ' · ' : '') + result.rate_hz + ' Hz';
      if (result.skipped) note += (note ? ' · ' : '') + result.skipped + ' saltati';
      if (note) modelCell.appendChild(node('div', 'note', note));
      row.appendChild(modelCell);
      row.appendChild(node('td', 'num', result.avg_miss_m === null ? '–' : result.avg_miss_m.toFixed(2) + ' m'));
      row.appendChild(node('td', 'num', result.lateral_miss_m === null ? '–' : result.lateral_miss_m.toFixed(2) + ' m'));
      row.appendChild(node('td', 'num', result.segments === null ? '–' : String(result.segments)));
      row.appendChild(node('td', 'note', fmtTime(result.date)));
      body.appendChild(row);
    });
  }
  byId('btn-eval').addEventListener('click', () => {
    const model = byId('eval-model').value;
    runAction('eval', {model: model}, 'Valuto ' + model + ' sui segmenti hold-out (usa la GPU). Avvio?');
  });

  // -- Confronto video ---------------------------------------------------------------------------
  function compareFormReason() {
    if (!byId('cmp-segment').value) return 'nessun segmento localizzato';
    if (!byId('cmp-student').value) return 'nessun modello studente (run o HF)';
    return '';
  }

  function renderCompare() {
    const segments = (data.segments ? data.segments.segments : []).filter((segment) => segment.state === 'localized');
    const options = [];
    let firstHeldout = null;
    for (const segment of segments) {
      if (segment.heldout) {
        options.push({value: segment.name, label: '★ ' + segment.name + ' (hold-out)', group: 'Hold-out: mai visti nel training'});
        if (!firstHeldout) firstHeldout = segment.name;
      }
    }
    for (const segment of segments) {
      if (!segment.heldout) options.push({value: segment.name, label: segment.name, group: 'Usati nel training'});
    }
    fillSelect(byId('cmp-segment'), options, firstHeldout);
    const students = runModelOptions().concat(hfModelOptions());
    fillSelect(byId('cmp-student'), students);

    const videos = data.videos || [];
    if (same('videos', videos)) return;
    const list = byId('videos');
    list.replaceChildren();
    byId('videos-empty').hidden = videos.length > 0;
    for (const video of videos) {
      const item = node('li', 'item');
      const text = node('div', 'grow');
      const who = (video.student.label || video.student.model || '?') + ' contro '
        + (video.reference.label || video.reference.model || '?');
      text.appendChild(node('div', 'what', video.has_json ? who : video.name));
      let sub = (video.segment || video.name) + ' · ' + fmtTime(video.modified) + ' · ' + fmtBytes(video.size);
      if (video.student.avg_miss_m !== null) sub += ' · studente ' + video.student.avg_miss_m.toFixed(2) + ' m';
      if (video.reference.avg_miss_m !== null) sub += ' · riferimento ' + video.reference.avg_miss_m.toFixed(2) + ' m';
      text.appendChild(node('div', 'sub', sub));
      item.appendChild(text);
      const play = node('button', 'btn small', 'Guarda');
      play.type = 'button';
      play.addEventListener('click', () => playVideo(video.name));
      item.appendChild(play);
      list.appendChild(item);
    }
  }

  function playVideo(name) {
    const video = (data.videos || []).find((item) => item.name === name);
    const card = byId('player-card');
    card.hidden = false;
    const player = byId('player');
    const source = '/video/' + encodeURIComponent(name);
    if (player.getAttribute('src') !== source) player.setAttribute('src', source);
    byId('player-title').textContent = 'Video · ' + name;
    const miss = byId('player-miss');
    miss.replaceChildren();
    function box(who, side) {
      const element = node('div', 'box');
      const name = side.label || side.model;
      element.appendChild(node('div', 'who', who + (name ? ': ' + name : '')));
      element.appendChild(node('div', 'num', side.avg_miss_m === null ? '–' : side.avg_miss_m.toFixed(2) + ' m'));
      let note = 'avg miss su questo tratto';
      if (side.lateral_miss_m !== null) note += ' · laterale ' + side.lateral_miss_m.toFixed(2) + ' m';
      element.appendChild(node('div', 'who', note));
      return element;
    }
    if (video) {
      miss.appendChild(box('Riferimento (sopra)', video.reference));
      miss.appendChild(box('Studente (sotto)', video.student));
      let sub = video.segment ? 'segmento ' + video.segment : '';
      if (video.start !== null) sub += ' · da ' + video.start + ' s';
      if (video.duration !== null) sub += ' · ' + video.duration + ' s';
      if (video.speed !== null) sub += ' · ' + video.speed + '×';
      miss.appendChild(node('div', 'note', sub));
    } else if (!data.videos) {
      load('videos').then(() => { renderCompare(); playVideo(name); });
    }
    card.scrollIntoView({behavior: 'smooth', block: 'start'});
  }

  byId('btn-compare').addEventListener('click', () => {
    const params = {segment: byId('cmp-segment').value, reference: byId('cmp-reference').value,
      student: byId('cmp-student').value, start: numberOrText(byId('cmp-start').value),
      duration: numberOrText(byId('cmp-duration').value), speed: Number(byId('cmp-speed').value),
      inset: byId('cmp-inset').checked, show_gt: byId('cmp-gt').checked};
    runAction('compare', params);
  });
  for (const id of ['cmp-segment', 'cmp-student']) byId(id).addEventListener('change', updateButtons);

  // -- Pubblica --------------------------------------------------------------------------------
  function slugOf(name) { return name.toLowerCase().replace(/[^a-z0-9]/g, '_'); }
  function publishParams() {
    return {run: byId('publish-run').value, name: byId('publish-name').value, stable: byId('publish-stable').checked};
  }
  function sameParams(a, b) {
    return a && b && a.run === b.run && a.name === b.name && Boolean(a.stable) === Boolean(b.stable);
  }
  function lastDryRun() {
    if (!status) return null;
    const params = publishParams();
    for (const job of status.jobs) {
      if (job.action === 'publish_dry' && sameParams(job.params, params)) return job;
    }
    return null;
  }
  function nameOnHf(name) {
    if (!data.hf || !data.hf.ok || !name) return false;
    const slug = slugOf(name);
    return data.hf.models.indexOf(slug) >= 0 || data.hf.releases.indexOf(slug) >= 0;
  }

  let lastPublishRun = '';
  function renderPublish() {
    const finished = runsWith((run) => run.has_pt).map((run) => ({value: run.name, label: run.name + (run.has_onnx ? ' (onnx fatto)' : '')}));
    fillSelect(byId('export-run'), finished);
    fillSelect(byId('publish-run'), finished);
    // The name follows the run until the user writes another one.
    const run = byId('publish-run').value;
    const nameField = byId('publish-name');
    if (run !== lastPublishRun) {
      if (!nameField.value || nameField.value === lastPublishRun) nameField.value = run;
      lastPublishRun = run;
    }
    const hf = data.hf;
    if (hf && !same('hf', hf)) {
      const kv = byId('hf-kv');
      kv.replaceChildren();
      kv.appendChild(node('dt', '', 'Repo'));
      const repoCell = node('dd');
      const link = node('a', '', hf.repo);
      link.href = hf.url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      repoCell.appendChild(link);
      kv.appendChild(repoCell);
      kv.appendChild(node('dt', '', 'training/stable.json'));
      kv.appendChild(node('dd', '', hf.ok ? (hf.stable ? 'sì' : 'no') : '?'));
      kv.appendChild(node('dt', '', 'Letto'));
      kv.appendChild(node('dd', '', fmtTime(hf.checked) + (hf.ok ? '' : ' · ' + hf.error)));
      const list = byId('hf-models');
      list.replaceChildren();
      byId('hf-empty').hidden = !hf.ok || hf.models.length > 0;
      for (const name of hf.models) {
        const item = node('li', 'item');
        item.appendChild(node('span', 'what mono', name));
        if (hf.releases.indexOf(name) >= 0) item.appendChild(pill('pesi PyTorch', 'ok'));
        list.appendChild(item);
      }
    }
  }

  function updatePublishButtons() {
    const params = publishParams();
    const validName = /^[A-Za-z0-9_.-]{1,64}$/.test(params.name);
    const taken = nameOnHf(params.name);
    let nameNote = '';
    if (params.name && !validName) nameNote = 'Nome non valido: lettere, cifre, _ . - (massimo 64).';
    else if (taken) nameNote = 'Su Hugging Face c\'è già ' + slugOf(params.name) + ': la pubblicazione vera sarà rifiutata.';
    else if (params.name) nameNote = 'Su Hugging Face diventa models/' + slugOf(params.name) + '/.';
    byId('publish-name-note').textContent = nameNote;

    let dryReason = '';
    if (!params.run) dryReason = 'nessun run finito';
    else if (!validName) dryReason = 'scrivi un nome valido';
    setAction('btn-publish-dry', 'publish_dry', dryReason);

    const dry = lastDryRun();
    const dryStatus = byId('dry-status');
    dryStatus.replaceChildren();
    if (dry) {
      dryStatus.appendChild(document.createTextNode('Prova per questi parametri: '));
      dryStatus.appendChild(pill(dry.state_label, JOB_PILL[dry.state] || 'muted'));
      dryStatus.appendChild(document.createTextNode(' ' + fmtTime(dry.started) + ' '));
      const view = node('button', 'btn small', 'Log');
      view.type = 'button';
      view.addEventListener('click', () => openLog(dry.id, dry.label));
      dryStatus.appendChild(view);
    } else if (params.run && validName) {
      dryStatus.textContent = 'Nessuna prova ancora per questo run, nome e «stable».';
    }

    let reason = '';
    if (!params.run) reason = 'nessun run finito';
    else if (!validName) reason = 'scrivi un nome valido';
    else if (taken) reason = 'nome già usato su Hugging Face';
    else if (!dry || dry.state !== 'ok') reason = 'prima una prova (dry run) finita bene con gli stessi parametri';
    else if (byId('publish-confirm').value !== params.name) reason = 'scrivi il nome esatto per confermare';
    setAction('btn-publish', 'publish', reason);
  }

  byId('btn-export').addEventListener('click', () => runAction('export', {run: byId('export-run').value}));
  byId('btn-publish-dry').addEventListener('click', () => runAction('publish_dry', publishParams()));
  byId('btn-publish').addEventListener('click', () => {
    const params = publishParams();
    params.confirm = byId('publish-confirm').value;
    runAction('publish', params).then((job) => { if (job) byId('publish-confirm').value = ''; });
  });
  for (const id of ['publish-run', 'publish-name', 'publish-stable', 'publish-confirm']) {
    byId(id).addEventListener('input', () => { renderPublish(); updateButtons(); });
    byId(id).addEventListener('change', () => { renderPublish(); updateButtons(); });
  }

  // -- Comma -----------------------------------------------------------------------------------
  function renderCommaTab() {
    if (status) renderComma('comma-kv-comma');
    const finished = runsWith((run) => run.has_pt).map((run) => ({value: run.name, label: run.name}));
    fillSelect(byId('install-run'), finished);
  }
  byId('btn-comma-check').addEventListener('click', async () => {
    const button = byId('btn-comma-check');
    button.disabled = true;
    try {
      const answer = await post('/api/comma/check', {});
      status.comma = answer.comma;
      renderComma('comma-kv-comma');
      renderComma('comma-kv-stato');
      updateButtons();
    } catch (error) {
      toast(error.message, true);
    }
    button.disabled = false;
  });
  byId('btn-install').addEventListener('click', () => {
    const run = byId('install-run').value;
    runAction('install', {run: run, confirm: byId('install-confirm').value}).then((job) => { if (job) byId('install-confirm').value = ''; });
  });
  byId('btn-restore').addEventListener('click', () => {
    runAction('restore', {confirm: byId('restore-confirm').value}).then((job) => { if (job) byId('restore-confirm').value = ''; });
  });
  for (const id of ['install-confirm', 'restore-confirm', 'install-run']) byId(id).addEventListener('input', updateButtons);

  // -- Job -------------------------------------------------------------------------------------
  byId('btn-update').addEventListener('click', () => runAction('update', {}, 'Aggiorno il codice: git pull, uv sync e riavvio del sito (qualche secondo). Avvio?'));
  byId('btn-selftest').addEventListener('click', () => runAction('selftest', {}));
  byId('reload-page').addEventListener('click', () => location.reload());

  // -- log drawer ------------------------------------------------------------------------------
  function openLog(id, label) {
    clearTimeout(log.timer);
    log = {id: id, offset: 0, timer: null, follow: true};
    byId('term-title').textContent = 'Log · ' + label + ' · ' + id;
    byId('term-text').textContent = '';
    byId('term-follow').setAttribute('aria-pressed', 'true');
    byId('terminal').classList.add('open');
    document.body.classList.add('log-open');
    pollLog();
  }

  async function pollLog() {
    const id = log.id;
    if (!id) return;
    try {
      const chunk = await api('/api/jobs/' + id + '/log?offset=' + log.offset);
      if (log.id !== id) return;
      const pre = byId('term-text');
      if (chunk.text) {
        if (log.offset === 0 && chunk.offset - chunk.text.length > 0) pre.textContent = '… (inizio tagliato)\n';
        pre.textContent += chunk.text;
        if (log.follow) pre.scrollTop = pre.scrollHeight;
      }
      log.offset = chunk.offset;
    } catch (error) {
      toast(error.message, true);
      return;
    }
    const job = status && status.jobs.find((item) => item.id === id);
    const running = !job || job.state === 'running';
    byId('term-state').hidden = !running;
    log.timer = setTimeout(pollLog, running ? 1500 : 5000);
  }

  byId('term-close').addEventListener('click', () => {
    clearTimeout(log.timer);
    log.id = null;
    byId('terminal').classList.remove('open');
    document.body.classList.remove('log-open');
  });
  byId('term-follow').addEventListener('click', (event) => {
    log.follow = !log.follow;
    event.currentTarget.setAttribute('aria-pressed', String(log.follow));
    if (log.follow) byId('term-text').scrollTop = byId('term-text').scrollHeight;
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && log.id) byId('term-close').click();
  });

  // -- start -----------------------------------------------------------------------------------
  showTab(location.hash ? location.hash.slice(1) : 'stato');
  refreshStatus();
  setInterval(refreshStatus, STATUS_PERIOD);
  setInterval(() => { if (!document.hidden) loadTab(false); }, 5000);
})();

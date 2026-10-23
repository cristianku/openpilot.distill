// distill-web: the page. Vanilla JS, no libraries; the DOM is built with textContent only.
(function () {
  'use strict';

  const TABS = ['status', 'preprocessing', 'training', 'evaluation', 'compare', 'publish', 'settings'];
  // The tabs of the bar; Settings opens from the gear at the bottom of the screen.
  const BAR_TABS = TABS.filter((tab) => tab !== 'settings');
  // Where Publish puts a model (#publish/<destination>).
  const DESTINATIONS = ['comma', 'hf'];
  const JOB_PILL = {running: 'run', ok: 'ok', failed: 'bad', stopped: 'todo', unknown: 'muted'};
  const RUN_STATE = {running: ['running', 'run'], finished: ['finished', 'ok'], interrupted: ['interrupted', 'todo']};
  const SEGMENT_STATE = {localized: ['localized', 'ok'], failed: ['failed', 'bad'], todo: ['to localize', 'todo']};
  const CHART_TITLES = {'loss': 'loss', 'gt': 'gt (your driving)', 'distill': 'distill (the teacher)',
    'gt/plan': 'gt/plan (path)', 'grad_norm': 'grad_norm', 'step_time': 'time per step (s)'};
  // How often (ms) the data of each tab is read again while the tab is open.
  const TAB_PERIOD = {preprocessing: 15000, training: 10000, evaluation: 30000, compare: 30000, publish: 15000,
    status: 0, settings: 30000};
  const STATUS_PERIOD = 4000;
  const SMOOTHING = 0.85;

  let status = null;
  let firstCode = null;
  let currentTab = 'status';
  // Where the back arrow of Settings returns.
  let tabBeforeSettings = 'status';
  const data = {segments: null, runs: null, evalResults: null, videos: null, hf: null, metrics: null, detail: null, drives: null,
    settings: null};
  const loadedAt = {};
  let selectedRun = null;
  // The run opened from the list of the Training tab (#training/<run>); null = the one in progress.
  let openRun = null;
  // The destination chosen in Publish: 'comma' or 'hf'.
  let publishDest = 'comma';
  // A choice to make in a select as soon as it has that option (see goTo).
  const pendingChoice = {};
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
    return date.toLocaleString('en-GB', {day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit'});
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
    // Signed out (session ended, or another browser changed the password): the server shows the sign-in page.
    if (response.status === 401) {
      location.replace('/');
      throw new Error('Signed out.');
    }
    let body = null;
    try { body = await response.json(); } catch (error) { body = {message: 'Invalid response (HTTP ' + response.status + ').'}; }
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
    // A choice asked by goTo(): applied once, as soon as the option is there.
    let wanted = null;
    const pending = pendingChoice[select.id];
    if (pending !== undefined && options.some((option) => option.value === pending)) {
      wanted = pending;
      delete pendingChoice[select.id];
    }
    if (!same(key, options)) {
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
    if (wanted !== null && select.value !== wanted) {
      select.value = wanted;
      select.dispatchEvent(new Event('change'));
    }
  }

  // -- tabs ------------------------------------------------------------------------------------
  function showTab(name) {
    // #training/<run>: the Training tab with that run open; #publish/<destination>.
    let detail = null;
    const slash = name.indexOf('/');
    if (slash >= 0) {
      try {
        detail = decodeURIComponent(name.slice(slash + 1));
      } catch (error) {
        detail = '';
      }
      name = name.slice(0, slash);
    }
    if (name === 'training' && detail !== null) openRun = detail || null;
    // The tab was called Data before: old bookmarks still land on it.
    if (name === 'data') name = 'preprocessing';
    // The Jobs tab is gone: its job list and the code update are in Status.
    if (name === 'jobs') name = 'status';
    // The Comma tab is gone: installing and restoring are in Publish, "On your comma".
    if (name === 'comma') {
      name = 'publish';
      detail = 'comma';
    }
    // #publish/<destination>: straight to its last step (a bookmark, or "Install or restore a model").
    if (name === 'publish' && DESTINATIONS.indexOf(detail) >= 0) {
      publishDest = detail;
      publishWizard.show(3, false);
    }
    if (TABS.indexOf(name) < 0) name = 'status';
    if (name === 'settings' && currentTab !== 'settings') tabBeforeSettings = currentTab;
    currentTab = name;
    for (const tab of TABS) {
      const button = byId('tab-' + tab);
      const selected = tab === name;
      if (tab === 'settings') {
        button.setAttribute('aria-expanded', String(selected));
      } else {
        button.setAttribute('aria-selected', String(selected));
        button.tabIndex = selected ? 0 : -1;
      }
      byId('panel-' + tab).hidden = !selected;
    }
    let hash = '#' + name;
    if (name === 'training' && openRun) hash += '/' + encodeURIComponent(openRun);
    else if (name === 'publish' && publishWizard.step === 3) hash += '/' + publishDest;
    if (location.hash !== hash) history.replaceState(null, '', hash);
    loadTab(true);
  }
  for (const tab of BAR_TABS) {
    byId('tab-' + tab).addEventListener('click', () => showTab(tab));
    byId('tab-' + tab).addEventListener('keydown', (event) => {
      let index = BAR_TABS.indexOf(tab);
      if (event.key === 'ArrowRight') index = (index + 1) % BAR_TABS.length;
      else if (event.key === 'ArrowLeft') index = (index + BAR_TABS.length - 1) % BAR_TABS.length;
      else return;
      event.preventDefault();
      showTab(BAR_TABS[index]);
      byId('tab-' + BAR_TABS[index]).focus();
    });
  }
  // The gear opens Settings, and closes it again; the back arrow of Settings returns too.
  byId('tab-settings').addEventListener('click', () => {
    showTab(currentTab === 'settings' ? tabBeforeSettings : 'settings');
    window.scrollTo(0, 0);
  });
  byId('settings-back').addEventListener('click', () => {
    showTab(tabBeforeSettings);
    window.scrollTo(0, 0);
  });
  window.addEventListener('hashchange', () => showTab(location.hash.slice(1)));

  // Which data each tab needs.
  function needs(tab) {
    if (tab === 'preprocessing') return ['segments', 'drives'];
    if (tab === 'training') return ['runs', 'hf'];
    if (tab === 'evaluation') return ['runs', 'hf', 'evalResults', 'segments'];
    if (tab === 'compare') return ['segments', 'runs', 'hf', 'videos', 'drives'];
    if (tab === 'publish') return ['runs', 'hf', 'evalResults'];
    if (tab === 'settings') return ['settings', 'auth'];
    return [];
  }

  async function load(what) {
    try {
      if (what === 'segments') data.segments = await api('/api/data');
      else if (what === 'runs') data.runs = (await api('/api/runs')).runs;
      else if (what === 'evalResults') data.evalResults = (await api('/api/eval')).results;
      else if (what === 'videos') data.videos = (await api('/api/compare')).videos;
      else if (what === 'drives') data.drives = await api('/api/drives');
      else if (what === 'hf') data.hf = await api('/api/hf');
      else if (what === 'settings') data.settings = await api('/api/settings');
      else if (what === 'auth') data.auth = await api('/api/auth/state');
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
    if (tab === 'training') {
      await loadDetail(force);
      await loadMetrics(force);
    }
    renderTab();
  }

  function renderTab() {
    if (currentTab === 'preprocessing') renderData();
    else if (currentTab === 'training') renderTraining();
    else if (currentTab === 'evaluation') renderEval();
    else if (currentTab === 'compare') renderCompare();
    else if (currentTab === 'publish') renderPublish();
    else if (currentTab === 'settings') { renderSettings(); renderAdmin(); }
    updateButtons();
  }

  // -- status ----------------------------------------------------------------------------------
  async function refreshStatus() {
    try {
      status = await api('/api/status');
      byId('live-dot').className = 'dot';
      byId('live-text').textContent = 'updated ' + new Date().toLocaleTimeString('en-GB');
    } catch (error) {
      byId('live-dot').className = 'dot off';
      byId('live-text').textContent = 'unreachable: ' + error.message;
      return;
    }
    if (firstCode === null) firstCode = status.code;
    byId('code-banner').hidden = !(status.code && firstCode && status.code !== firstCode);
    byId('code-version').textContent = status.code || 'unknown';
    byId('selftest-box').hidden = !status.test_actions;
    renderMachine();
    renderRunning();
    renderComma('comma-kv-stato');
    renderTrainingNow();
    renderJobs();
    renderLiveJobs();
    // A job just ended: what it produced shows up in the open tab at once.
    const running = status.jobs.filter((job) => job.state === 'running').map((job) => job.id).join(',');
    if (previousRunning && running !== previousRunning) loadTab(true);
    else renderTab();
    previousRunning = running;
    if (log.id) renderSheetStatus();
  }

  // The last few minutes of CPU and GPU use, one point per status (every 4 s), kept in the page.
  const HISTORY_POINTS = 90;
  // Not called history: that would hide the browser's history (replaceState in showTab).
  const usageHistory = {cpu: [], gpu: []};
  function pushHistory(key, value) {
    usageHistory[key].push(value);
    if (usageHistory[key].length > HISTORY_POINTS) usageHistory[key].shift();
  }

  // An area chart 0-100% of a history, newest on the right.
  function renderSpark(id, values, color) {
    const box = byId(id);
    const SVG = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(SVG, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + HISTORY_POINTS + ' 30');
    svg.setAttribute('preserveAspectRatio', 'none');
    if (values.length > 1) {
      const start = HISTORY_POINTS - values.length;
      const points = values.map((value, index) => (start + index) + ',' + (30 - Math.max(0, Math.min(100, value)) * 0.3));
      const area = document.createElementNS(SVG, 'polygon');
      area.setAttribute('points', start + ',30 ' + points.join(' ') + ' ' + (HISTORY_POINTS - 1) + ',30');
      // A CSS colour (var(--...)) through style: it follows the theme.
      area.style.fill = color;
      area.setAttribute('fill-opacity', '0.15');
      svg.appendChild(area);
      const line = document.createElementNS(SVG, 'polyline');
      line.setAttribute('points', points.join(' '));
      line.setAttribute('fill', 'none');
      line.style.stroke = color;
      line.setAttribute('stroke-width', '1.5');
      line.setAttribute('vector-effect', 'non-scaling-stroke');
      svg.appendChild(line);
    }
    box.replaceChildren(svg);
  }

  // One small bar per core, green to orange to red with the load.
  function renderCores(cores) {
    const box = byId('cpu-cores');
    if (box.children.length !== cores.length) {
      box.replaceChildren();
      for (let index = 0; index < cores.length; index++) box.appendChild(node('span', 'core'));
    }
    cores.forEach((value, index) => {
      const bar = box.children[index];
      bar.style.setProperty('--h', Math.max(4, value) + '%');
      bar.className = 'core' + (value >= 85 ? ' hot' : value >= 50 ? ' warm' : '');
      bar.title = 'core ' + index + ': ' + Math.round(value) + '%';
    });
  }

  // Label: value lines under a machine card, in place of one dense line of numbers.
  function statRows(id, rows) {
    const box = byId(id);
    if (same('stats:' + id, rows)) return;
    box.replaceChildren();
    box.classList.add('stats');
    for (const [label, value] of rows) {
      box.appendChild(node('span', 'stat-label', label));
      box.appendChild(node('span', 'stat-value', value));
    }
  }

  function renderMachine() {
    const system = status.system;
    const gpu = system.gpu;
    if (gpu) {
      const util = gpu.utilization === null ? 0 : gpu.utilization;
      byId('gpu-value').textContent = util + '%';
      setBar('gpu-bar', util, util > 85 ? 'warn' : '');
      pushHistory('gpu', util);
      renderSpark('gpu-spark', usageHistory.gpu, 'var(--spark-gpu)');
      const rows = [['Card', gpu.name.replace(/^Tesla /, '')]];
      if (gpu.memory_total_mb) {
        rows.push(['Memory', (gpu.memory_used_mb / 1024).toFixed(1) + ' of ' + Math.round(gpu.memory_total_mb / 1024) + ' GB']);
      }
      if (gpu.temperature !== null) rows.push(['Temperature', gpu.temperature + ' °C']);
      if (gpu.power_w !== null) rows.push(['Power', Math.round(gpu.power_w) + ' W']);
      rows.push(['Doing', gpu.processes.length ? gpu.processes.length + (gpu.processes.length === 1 ? ' job' : ' jobs')
        : 'nothing (idle)']);
      statRows('gpu-sub', rows);
    } else {
      byId('gpu-value').textContent = 'n/a';
      setBar('gpu-bar', 0, '');
      statRows('gpu-sub', [['Card', 'no NVIDIA GPU found']]);
    }
    if (system.cpus) {
      const cpu = system.cpu;
      if (cpu) {
        byId('cpu-value').textContent = Math.round(cpu.total) + '%';
        pushHistory('cpu', cpu.total);
        renderCores(cpu.cores);
      } else {
        byId('cpu-value').textContent = '…';
      }
      renderSpark('cpu-spark', usageHistory.cpu, 'var(--spark-cpu)');
      const rows = [];
      if (cpu) rows.push(['Busy cores', cpu.cores.filter((value) => value >= 50).length + ' of ' + system.cpus]);
      else rows.push(['Cores', String(system.cpus)]);
      if (system.load) {
        // Load average = programs running or waiting, on average: per core it says how loaded the
        // machine was over the last 15 minutes, and comparing 1 and 15 minutes gives the trend.
        const perCore = system.load[2] / system.cpus;
        rows.push(['Last 15 minutes', perCore < 0.25 ? 'light' : (perCore < 0.7 ? 'moderate' : 'heavy')]);
        const trend = system.load[0] > system.load[2] * 1.3 + 0.5 ? 'rising'
          : (system.load[0] < system.load[2] * 0.7 - 0.5 ? 'calming down' : 'steady');
        rows.push(['Trend', trend]);
      }
      statRows('cpu-sub', rows);
    }
    const memory = system.memory;
    if (memory) {
      byId('ram-value').textContent = memory.available_gb + ' GB free';
      setBar('ram-bar', memory.used_percent, memory.used_percent > 90 ? 'bad' : (memory.used_percent > 75 ? 'warn' : ''));
      statRows('ram-sub', [['In use', (memory.total_gb - memory.available_gb).toFixed(1) + ' GB (' + memory.used_percent + '%)'],
        ['Total', memory.total_gb + ' GB']]);
    } else {
      byId('ram-value').textContent = 'n/a';
    }
    const disk = system.disk;
    if (disk) {
      const used = 100 - disk.free_gb * 100 / disk.total_gb;
      byId('disk-value').textContent = disk.free_gb + ' GB free';
      setBar('disk-bar', used, disk.free_gb < 10 ? 'bad' : (used > 80 ? 'warn' : ''));
      statRows('disk-sub', [['In use', (disk.total_gb - disk.free_gb).toFixed(1) + ' GB (' + Math.round(used) + '%)'],
        ['Total', disk.total_gb + ' GB']]);
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
      body.appendChild(row);
    }
  }

  // Segments at a time of a running localization, changed live: "- 8 +" and what it means in cores.
  function parallelControl(job) {
    const parallel = job.parallel;
    const cpus = (status && status.system.cpus) || parallel.jobs;
    const [low, high] = [1, Math.min(16, cpus)];
    const box = node('div', 'parallel');
    const stepper = node('div', 'stepper');
    stepper.setAttribute('role', 'group');
    stepper.setAttribute('aria-label', 'Segments localized at a time');
    function step(label, aria, value, disabled) {
      const button = node('button', 'stepper-button', label);
      button.type = 'button';
      button.setAttribute('aria-label', aria);
      button.disabled = disabled;
      button.addEventListener('click', async () => {
        try {
          const answer = await post('/api/jobs/' + job.id + '/parallel', {jobs: value});
          toast(answer.job.parallel ? answer.job.parallel.jobs + ' segments at a time' : 'Changed');
        } catch (error) {
          toast(error.message, true);
        }
        refreshStatus();
      });
      return button;
    }
    stepper.appendChild(step('\u2212', 'Fewer at a time', parallel.jobs - 1, parallel.jobs <= low));
    stepper.appendChild(node('span', 'stepper-value', String(parallel.jobs)));
    stepper.appendChild(step('+', 'More at a time', parallel.jobs + 1, parallel.jobs >= high));
    box.appendChild(stepper);
    let text = 'segments at a time \u00b7 about ' + parallel.jobs + ' of ' + cpus + ' cores';
    if (parallel.running !== undefined && parallel.running !== parallel.jobs) text += ' \u00b7 ' + parallel.running + ' running now';
    if (parallel.auto) text += ' \u00b7 auto was ' + parallel.auto;
    box.appendChild(node('span', 'sub', text));
    return box;
  }

  function jobItem(job, withButtons) {
    const item = node('li', 'item');
    item.appendChild(pill(job.state_label, JOB_PILL[job.state] || 'muted'));
    const text = node('div', 'grow');
    text.appendChild(node('div', 'what', job.label));
    let sub = job.summary || '';
    sub += (sub ? ' · ' : '') + fmtTime(job.started);
    if (job.state === 'running') sub += ' · for ' + fmtDuration(Date.now() / 1000 - job.started);
    else if (job.finished) sub += ' · took ' + fmtDuration(job.finished - job.started);
    if (job.exit_code !== null && job.exit_code !== undefined) sub += ' · exit code ' + job.exit_code;
    text.appendChild(node('div', 'sub', sub));
    item.appendChild(text);
    if (withButtons) {
      if (job.run) {
        const charts = node('button', 'btn small', 'Charts');
        charts.type = 'button';
        charts.addEventListener('click', () => openRunView(job.run));
        item.appendChild(charts);
      }
      if (job.video) {
        const play = node('button', 'btn small', 'Watch');
        play.type = 'button';
        play.addEventListener('click', () => { showTab('compare'); playVideo(job.video); });
        item.appendChild(play);
      }
      const view = node('button', 'btn small', 'Log');
      view.type = 'button';
      view.addEventListener('click', () => openLog(job.id, job.label));
      item.appendChild(view);
      if (job.stoppable) {
        const stop = node('button', 'btn small danger', 'Stop');
        stop.type = 'button';
        stop.addEventListener('click', () => stopJob(job));
        item.appendChild(stop);
      }
    }
    if (job.state === 'running' && job.parallel && job.parallel.jobs) item.appendChild(parallelControl(job));
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
    byId('status-sub').textContent = running.length === 0 ? 'Nothing running: the server is free.'
      : running.length === 1 ? 'Working on: ' + running[0].label + '.'
        : running.length + ' jobs running: ' + running.map((job) => job.label).join(', ') + '.';
    // Rebuilt when a job, its progress or its settings change (and once a minute for the "for N min"
    // text), not at every status: a rebuild under the finger would lose a click on its buttons.
    const list = byId('running-jobs');
    const minute = Math.floor(Date.now() / 60000);
    if (!same('running-jobs', [running.map((job) => [job.id, job.progress, job.parallel, job.stoppable]), minute])) {
      list.replaceChildren();
      for (const job of running) list.appendChild(jobItem(job, true));
    }
    byId('running-empty').hidden = running.length > 0;

    if (same('activities', status.activities)) return;
    const activities = byId('activities');
    activities.replaceChildren();
    byId('activities-empty').hidden = status.activities.length > 0;
    for (const activity of status.activities) {
      const item = node('li', 'item');
      item.appendChild(pill(activity.label, activity.group === 'other' ? 'imp' : 'run'));
      item.appendChild(node('span', 'sub', 'pid ' + activity.pid + (activity.uses_gpu ? ' · GPU' : '')));
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
    if (comma.reachable === null) row('Reachable', 'not checked yet', 'muted');
    else if (comma.reachable) row('Reachable', 'yes', 'ok');
    else row('Reachable', 'no', 'bad');
    if (comma.reachable) {
      if (comma.onroad === true) row('State', 'onroad (driving)', 'todo');
      else row('State', 'offroad (parked)', 'ok');
      if (comma.bundle_default === false) row('Model chosen in the Models menu', comma.bundle, 'imp');
      else row('Models menu', 'Default (openpilot\'s model)');
      row('Copy of the original model', comma.backup ? 'saved on the comma' : 'none yet');
    } else if (comma.error) {
      row('Error', comma.error);
    }
    row('Checked', comma.checked ? fmtTime(comma.checked) + ' (every 30 s while the page is open)' : '–');
  }

  function renderTrainingNow() {
    const card = byId('training-card');
    const box = byId('training-now');
    card.hidden = status.training.length === 0;
    box.replaceChildren();
    for (const run of status.training) {
      const item = node('div', 'item');
      item.appendChild(pill('running', 'run'));
      const text = node('div', 'grow');
      text.appendChild(node('div', 'what', run.name));
      let sub = 'step ' + run.steps_done + (run.steps_total ? ' / ' + run.steps_total : '');
      if (run.eta_seconds !== null) sub += ' · ' + fmtDuration(run.eta_seconds) + ' left';
      if (run.avg_step_time !== null) sub += ' · ' + run.avg_step_time.toFixed(2) + ' s/step';
      if (run.last && run.last.loss !== undefined) sub += ' · loss ' + fmt(run.last.loss);
      text.appendChild(node('div', 'sub', sub));
      item.appendChild(text);
      const charts = node('button', 'btn small', 'Charts');
      charts.type = 'button';
      charts.addEventListener('click', () => openRunView(run.name));
      item.appendChild(charts);
      const progress = node('div', 'progress');
      progress.appendChild(barNode(run.percent, 'run'));
      progress.appendChild(node('span', 'sub', (run.percent === null ? '?' : run.percent) + '%'));
      item.appendChild(progress);
      box.appendChild(item);
    }
  }

  // -- live progress next to the buttons --------------------------------------------------------
  // Every action card has a <div class="job-live" data-actions="a,b">: it shows the running job of
  // those actions (bar, percentage, elapsed and remaining time, stop) and, for a few minutes after
  // it ends, how it went.
  const RECENT_SECONDS = 600;

  function liveJob(actions) {
    const jobs = (status.jobs || []).filter((job) => actions.indexOf(job.action) >= 0);
    const running = jobs.find((job) => job.state === 'running');
    if (running) return running;
    const now = Date.now() / 1000;
    const recent = jobs.filter((job) => job.finished && now - job.finished < RECENT_SECONDS);
    recent.sort((a, b) => b.finished - a.finished);
    return recent[0] || null;
  }

  function liveTexts(job) {
    const now = Date.now() / 1000;
    if (job.state !== 'running') {
      const took = job.finished ? fmtDuration(job.finished - job.started) : '–';
      if (job.state === 'ok') return {head: 'Done in ' + took, sub: 'Finished at ' + fmtTime(job.finished) + '.'};
      if (job.state === 'stopped') return {head: 'Stopped after ' + took, sub: ''};
      return {head: 'Failed after ' + took, sub: 'Open the details to see what went wrong.'};
    }
    const elapsed = now - job.started;
    const progress = job.progress;
    if (!progress || !progress.total) return {head: 'Starting…', sub: fmtDuration(elapsed) + ' elapsed'};
    let sub = progress.done.toLocaleString('en-GB') + ' of ' + progress.total.toLocaleString('en-GB')
      + ' · ' + fmtDuration(elapsed) + ' elapsed';
    if (progress.rate_mb_s) sub += ' · ' + progress.rate_mb_s.toFixed(1) + ' MB/s';
    if (progress.done >= progress.total) sub += ' · finishing';
    else if (progress.done > 0 && progress.percent >= 1) {
      sub += ' · about ' + fmtDuration(Math.max(1, elapsed * (progress.total - progress.done) / progress.done)) + ' left';
    }
    return {head: Math.floor(progress.percent) + '%', sub: sub};
  }

  function renderLiveJobs() {
    for (const box of document.querySelectorAll('.job-live')) {
      const job = liveJob(box.dataset.actions.split(','));
      if (!job) {
        box.replaceChildren();
        box.dataset.job = '';
        continue;
      }
      const kind = job.state === 'running' ? 'running' : (job.state === 'ok' ? 'ok' : 'bad');
      const key = job.id + ':' + job.state + ':' + job.stoppable;
      if (box.dataset.job !== key) {
        box.dataset.job = key;
        const card = node('div', 'jobcard ' + kind);
        const head = node('div', 'jobcard-head');
        if (kind === 'running') head.appendChild(node('span', 'spinner'));
        head.appendChild(node('strong', 'jobcard-label', job.label));
        head.appendChild(node('span', 'jobcard-state'));
        card.appendChild(head);
        const bar = node('div', 'jobcard-bar');
        bar.appendChild(node('span'));
        card.appendChild(bar);
        card.appendChild(node('div', 'jobcard-sub'));
        const buttons = node('div', 'jobcard-buttons');
        const details = node('button', 'btn small', 'Details');
        details.type = 'button';
        details.addEventListener('click', () => openLog(job.id, job.label));
        buttons.appendChild(details);
        if (job.state === 'running' && job.stoppable) {
          const stop = node('button', 'btn small danger', 'Stop');
          stop.type = 'button';
          stop.addEventListener('click', () => stopJob(job));
          buttons.appendChild(stop);
        }
        card.appendChild(buttons);
        box.replaceChildren(card);
      }
      const texts = liveTexts(job);
      box.querySelector('.jobcard-state').textContent = texts.head;
      box.querySelector('.jobcard-sub').textContent = texts.sub;
      const bar = box.querySelector('.jobcard-bar');
      const known = job.progress && job.progress.total;
      bar.classList.toggle('indeterminate', job.state === 'running' && !known);
      let percent = known ? job.progress.percent : 0;
      if (job.state === 'ok') percent = 100;
      bar.firstChild.style.width = Math.max(0, Math.min(100, percent)) + '%';
    }
  }

  // Status, "Recent jobs": the finished ones (the running ones are under "Running").
  const RECENT_JOBS = 15;
  function renderJobs() {
    const list = byId('jobs');
    const finished = status.jobs.filter((job) => job.state !== 'running').slice(0, RECENT_JOBS);
    if (same('recent-jobs', finished)) return;
    list.replaceChildren();
    byId('jobs-empty').hidden = finished.length > 0;
    for (const job of finished) list.appendChild(jobItem(job, true));
  }

  // -- action buttons --------------------------------------------------------------------------
  function actionReason(name) {
    if (!status || !status.actions[name]) return 'server status not read yet';
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
      toast('Started: ' + answer.job.label);
      await refreshStatus();
      return answer.job;
    } catch (error) {
      toast(error.message, true);
      refreshStatus();
      return null;
    }
  }

  async function stopJob(job) {
    if (!window.confirm('Stop "' + job.label + '"? It stops within about twenty seconds.')) return;
    try {
      await post('/api/jobs/' + job.id + '/stop', {});
      toast('Stopping: ' + job.label);
    } catch (error) {
      toast(error.message, true);
    }
    refreshStatus();
  }

  function updateButtons() {
    if (!status) return;
    setAction('btn-fetch', 'fetch', fetchFormReason());
    setAction('btn-localize', 'localize');
    updateTrainButtons();
    setAction('btn-eval', 'eval', byId('eval-model').value ? '' : 'no model to evaluate');
    setAction('btn-compare', 'compare', compareFormReason());
    updatePublishButtons();
    setAction('btn-install', 'install', publishRun() ? '' : 'choose a finished model in step 1');
    setAction('btn-restore', 'restore');
    setAction('btn-update', 'update');
    if (status.test_actions) setAction('btn-selftest', 'selftest');
  }

  // -- Preprocessing ---------------------------------------------------------------------------
  // GPS map quality of the localized segments (web/status.py map_of).
  const GPS_ORDER = ['final', 'rapid', 'prediction', 'unknown'];
  const GPS_LABEL = {final: 'final', rapid: 'rapid', prediction: 'prediction', unknown: 'not recorded'};
  const GPS_PILL = {final: 'ok', rapid: 'run', prediction: 'todo', unknown: 'muted'};

  function gpsBar(map, className) {
    const bar = node('div', className);
    const total = GPS_ORDER.reduce((sum, key) => sum + (map[key] || 0), 0);
    bar.title = GPS_ORDER.filter((key) => map[key]).map((key) => map[key] + ' ' + GPS_LABEL[key]).join(', ');
    for (const key of GPS_ORDER) {
      if (!map[key]) continue;
      const part = node('span', 'gps-' + key);
      part.style.flexGrow = String(map[key] / total);
      bar.appendChild(part);
    }
    return bar;
  }

  function renderGps(counts) {
    const map = counts.map || {};
    const localized = GPS_ORDER.reduce((sum, key) => sum + (map[key] || 0), 0);
    byId('gps-card').hidden = localized === 0;
    byId('gps-bar').replaceWith(Object.assign(gpsBar(map, 'gps-bar'), {id: 'gps-bar'}));
    const legend = byId('gps-legend');
    legend.replaceChildren();
    for (const key of GPS_ORDER) {
      if (key === 'unknown' && !map[key]) continue;
      const item = node('li');
      item.appendChild(node('span', 'gps-dot gps-' + key));
      item.appendChild(node('strong', '', String(map[key] || 0)));
      item.appendChild(node('span', '', GPS_LABEL[key]));
      legend.appendChild(item);
    }
    // Localize does both: the new segments and the ones a better map is out for.
    const callout = byId('gps-callout');
    const work = (counts.refinable || 0) + (counts.todo || 0);
    callout.hidden = work === 0;
    if (!same('gps-callout', [counts.refinable, counts.todo])) {
      callout.replaceChildren();
      const parts = [];
      if (counts.todo) parts.push(counts.todo + (counts.todo === 1 ? ' new segment' : ' new segments') + ' to localize');
      if (counts.refinable) {
        parts.push(counts.refinable + (counts.refinable === 1 ? ' segment' : ' segments')
          + ' can probably get a more accurate map');
      }
      callout.appendChild(node('span', '', parts.join(', ') + '.'));
      const button = node('button', 'btn primary small', 'Localize now');
      button.type = 'button';
      button.id = 'btn-localize-gps';
      button.addEventListener('click', () => runAction('localize', localizeParams(),
        'Localize ' + parts.join(' and ') + '? It runs in the background; you can follow it below.'));
      callout.appendChild(button);
    }
    setAction('btn-localize-gps', 'localize');
  }

  // The drives (routes) of data/ as cards: a picture, when, how long, how far, how fast, and the state
  // of their segments. Times and pictures come from /api/drives (localized segments only).
  function renderRoutes(routes) {
    const box = byId('routes');
    const drives = data.drives ? data.drives.drives : [];
    if (same('routes', [routes, drives])) return;
    box.replaceChildren();
    byId('routes-empty').hidden = routes.length > 0;
    const byRoute = new Map();
    for (const drive of drives) {
      if (!byRoute.has(drive.route)) byRoute.set(drive.route, []);
      byRoute.get(drive.route).push(drive);
    }
    const cards = routes.map((route) => {
      const list = (byRoute.get(route.route) || []).filter((drive) => drive.start);
      list.sort((a, b) => a.start - b.start);
      // Not localized yet = just downloaded: the newest, on top.
      return {route: route, list: list, start: list.length ? list[0].start : Infinity};
    });
    cards.sort((a, b) => b.start - a.start);
    cards.forEach((item, index) => {
      const route = item.route;
      const list = item.list;
      const card = node('div', 'route-card');
      card.style.setProperty('--i', String(index));
      const picture = node('div', 'route-picture');
      if (list.length) {
        const image = document.createElement('img');
        image.loading = 'lazy';
        image.alt = '';
        image.src = '/thumb/' + list[Math.floor(list.length / 2)].segment + '.jpg';
        picture.appendChild(image);
      } else {
        picture.appendChild(node('span', 'route-placeholder', '\u{1F697}'));
      }
      card.appendChild(picture);
      const body = node('div', 'route-body');
      if (list.length) {
        const last = list[list.length - 1];
        const end = last.start + (last.duration_s || 60);
        body.appendChild(node('div', 'route-title', fmtDate(list[0].start)));
        body.appendChild(node('div', 'route-time', fmtClock(list[0].start) + ' \u2013 ' + fmtClock(end)));
        const km = list.reduce((sum, drive) => sum + (drive.distance_km || 0), 0);
        const speeds = list.map((drive) => drive.avg_kmh).filter((value) => value !== undefined);
        const facts = [list.length + ' min'];
        if (km) facts.push(km.toFixed(1) + ' km');
        if (speeds.length) facts.push(Math.round(speeds.reduce((a, b) => a + b, 0) / speeds.length) + ' km/h avg');
        body.appendChild(node('div', 'route-facts', facts.join(' \u00b7 ')));
      } else {
        body.appendChild(node('div', 'route-title', 'Not localized yet'));
        body.appendChild(node('div', 'route-time', route.segments + ' segments downloaded'));
      }
      const state = [];
      if (route.localized) state.push(route.localized + ' localized');
      if (route.todo) state.push(route.todo + ' to localize');
      if (route.failed) state.push(route.failed + ' failed');
      if (route.heldout) state.push(route.heldout + ' \u2605 hold-out');
      body.appendChild(node('div', 'route-state', state.join(' \u00b7 ')));
      if (route.localized) body.appendChild(gpsBar(route.map, 'gps-bar route'));
      card.appendChild(body);
      box.appendChild(card);
    });
    if (data.drives && data.drives.pending) {
      setTimeout(() => { if (currentTab === 'preprocessing') load('drives').then(() => renderRoutes(routes)); }, 4000);
    }
  }

  function renderData() {
    const segments = data.segments;
    if (segments) renderRoutes(segments.routes);
    if (!segments || same('data', segments)) return;
    const counts = segments.counts;
    const tiles = byId('data-tiles');
    tiles.replaceChildren();
    const items = [[counts.total, 'segments', 'total'], [counts.localized, 'localized', 'localized'],
      [counts.failed, 'failed', 'failed'], [counts.todo, 'to localize', 'todo'],
      [counts.heldout_localized, 'hold-out ready', 'heldout'], [counts.discarded, 'discarded', 'discarded']];
    for (const [value, label, key] of items) {
      const tile = node('button', 'tile');
      tile.type = 'button';
      tile.appendChild(node('div', 'big', String(value)));
      tile.appendChild(node('div', 'what', label));
      tile.appendChild(node('span', 'tile-more', '\u203a'));
      tile.addEventListener('click', () => openDataInfo(key));
      tiles.appendChild(tile);
    }
    renderGps(counts);
    renderRoutes(segments.routes);
    renderLocalizeWhat(counts);
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
      const mapCell = node('td');
      if (segment.map_quality) {
        mapCell.appendChild(pill(GPS_LABEL[segment.map_quality] || segment.map_quality, GPS_PILL[segment.map_quality] || 'muted'));
        if (segment.refinable) mapCell.appendChild(node('span', 'note', ' better one out'));
      } else {
        mapCell.textContent = '–';
      }
      row.appendChild(mapCell);
      let note = segment.reason || '';
      if (segment.broken_video) note = 'video missing from the shared folder. ' + note;
      row.appendChild(node('td', 'note', note));
      body.appendChild(row);
    }
  }
  // -- the drives: two choice cards, download or localize, and the button of the chosen one --
  // Until the user picks one, Localize is chosen by itself when new segments are waiting for it.
  let prepChosen = false;
  function prepJob() {
    const chosen = document.querySelector('input[name="prep-job"]:checked');
    return chosen ? chosen.value : 'fetch';
  }
  function choosePrepJob(job) {
    const radio = document.querySelector('input[name="prep-job"][value="' + job + '"]');
    if (radio && !radio.checked) {
      radio.checked = true;
      renderPrepJob();
    }
  }
  function renderPrepJob() {
    renderChoices(byId('prep-step'));
    byId('prep-fetch').hidden = prepJob() !== 'fetch';
    byId('prep-localize').hidden = prepJob() !== 'localize';
    byId('fetch-since').hidden = !byId('fetch-since-on').checked;
  }
  function fetchFormReason() {
    return byId('fetch-since-on').checked && !byId('fetch-since').value ? 'choose the day to start from' : '';
  }
  // What Localize would do: the new segments, and the ones a more accurate GPS map is out for.
  function renderLocalizeWhat(counts) {
    const parts = [];
    if (counts.todo) parts.push(counts.todo + (counts.todo === 1 ? ' new segment' : ' new segments') + ' to localize');
    if (counts.refinable) {
      parts.push(counts.refinable + (counts.refinable === 1 ? ' segment' : ' segments') + ' can get a more accurate map');
    }
    byId('prep-localize-what').textContent = parts.length ? capital(parts.join(' · ')) + '.' : 'Nothing new to localize.';
    let short = 'Nothing new right now';
    if (counts.todo) short = counts.todo + ' new to localize';
    else if (counts.refinable) short = counts.refinable + ' can get a better map';
    byId('prep-localize-short').textContent = short;
    if (!prepChosen && counts.todo) choosePrepJob('localize');
  }
  for (const radio of document.querySelectorAll('input[name="prep-job"]')) {
    radio.addEventListener('change', () => { prepChosen = true; renderPrepJob(); updateButtons(); });
  }
  // A tap on the card already chosen, or work in its block, is a choice too.
  for (const label of byId('prep-step').querySelectorAll('.choice')) label.addEventListener('click', () => { prepChosen = true; });
  byId('prep-fetch').addEventListener('focusin', () => { prepChosen = true; });
  byId('fetch-since-on').addEventListener('change', () => {
    renderPrepJob();
    if (byId('fetch-since-on').checked) byId('fetch-since').focus();
    updateButtons();
  });
  byId('fetch-since').addEventListener('input', updateButtons);
  renderPrepJob();
  byId('btn-fetch').addEventListener('click', () => {
    const since = byId('fetch-since-on').checked ? byId('fetch-since').value : '';
    const params = since ? {since: since} : {};
    // Localize comes next: it is ready (and waits for the download) on the other card.
    runAction('fetch', params).then((job) => { if (job) choosePrepJob('localize'); });
  });
  byId('btn-localize').addEventListener('click', () => runAction('localize', localizeParams()));

  // The localization starts with the server's choice of segments at a time (cores, free memory);
  // the Status tab changes it while it runs.
  function localizeParams() { return {}; }

  // -- Training --------------------------------------------------------------------------------
  const MODE_TEXT = {fresh: 'from scratch', continue: 'from the stable weights', resume: 'resumed'};

  function runsWith(filter) { return (data.runs || []).filter(filter); }
  function runningRun() { return (data.runs || []).find((run) => run.status === 'running') || null; }
  // The run shown in detail: the one opened from the list, otherwise the training in progress.
  function viewedRun() {
    if (openRun) return openRun;
    const running = runningRun();
    return running ? running.name : null;
  }
  function openRunView(name) {
    openRun = name;
    data.detail = null;
    same.cache['run-buttons'] = '';
    showTab('training');
    window.scrollTo(0, 0);
  }
  byId('run-back').addEventListener('click', () => openRunView(null));

  // Goes to another tab with a choice already made in one of its selects (applied by fillSelect).
  function goTo(tab, selectId, value) {
    pendingChoice[selectId] = value;
    showTab(tab);
  }

  function teacherName(path) {
    if (!path) return '–';
    const file = String(path).split('/').pop();
    if (file === 'big_driving_supercombo_1365126a_12864.onnx') return 'MD v3';
    if (file === 'big_driving_supercombo.onnx') return '084747c7 (old)';
    return file;
  }

  function trainMode() {
    const checked = document.querySelector('input[name="train-mode"]:checked');
    return checked ? checked.value : 'fresh';
  }

  function updateTrainButtons() {
    const mode = trainMode();
    byId('train-steps').disabled = mode === 'resume';
    byId('train-teacher').disabled = mode === 'resume';
    byId('train-resume-run').disabled = mode !== 'resume';
    let extra = '';
    if (mode === 'resume' && !byId('train-resume-run').value) extra = 'no run to resume';
    if (mode === 'continue' && data.hf && data.hf.ok && data.hf.stable === false) extra = 'no stable weights on Hugging Face';
    if (!byId('train-gpus').value) extra = extra || 'no GPU found';
    const wandb = accountOf('wandb');
    if (!extra && byId('train-wandb').value === 'online' && wandb && !wandb.connected) {
      extra = 'Weights & Biases is not connected: connect it in Settings, or choose offline';
    }
    setAction('btn-train', 'train', extra);
    setAction('btn-train-test', 'train_test', byId('train-gpus').value ? '' : 'no GPU found');
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
      workers: numberOrText(byId('train-workers').value), wandb: byId('train-wandb').value,
      gpus: byId('train-gpus').value};
    // Step 3 of the wizard is the confirmation: no extra question.
    if (mode === 'resume') {
      params.run = byId('train-resume-run').value;
    } else {
      params.steps = numberOrText(byId('train-steps').value);
      params.teacher = byId('train-teacher').value;
    }
    runAction('train', params).then((job) => {
      if (!job) return;
      trainWizard.show(1);
      openRunView(null);
    });
  });
  byId('btn-train-test').addEventListener('click', () => runAction('train_test', {gpus: byId('train-gpus').value}));
  for (const radio of document.querySelectorAll('input[name="train-mode"]')) {
    radio.addEventListener('change', () => { renderTrainWizard(); updateButtons(); });
  }
  byId('train-gpus').addEventListener('change', updateButtons);
  byId('train-resume-run').addEventListener('change', () => {
    const run = (data.runs || []).find((item) => item.name === byId('train-resume-run').value);
    if (run && run.config) {
      if (run.config.batch_size) byId('train-batch').value = run.config.batch_size;
      if (run.config.workers !== undefined) byId('train-workers').value = run.config.workers;
    }
    updateButtons();
  });

  function renderTrainForm() {
    const gpus = (status && status.system.gpus) || [];
    const gpuOptions = [];
    if (gpus.length > 1) gpuOptions.push({value: 'all', label: 'All (' + gpus.length + ' GPUs)'});
    for (const gpu of gpus) {
      const memory = gpu.memory_total_mb ? ' · ' + Math.round(gpu.memory_total_mb / 1024) + ' GB' : '';
      gpuOptions.push({value: String(gpu.index), label: 'GPU ' + gpu.index + ' · ' + gpu.name + memory});
    }
    fillSelect(byId('train-gpus'), gpuOptions);

    const resumable = runsWith((run) => run.resumable);
    const options = resumable.map((run) => ({value: run.name, label: run.name + ' (' + run.steps_done + (run.steps_total ? '/' + run.steps_total : '') + ' steps)'}));
    fillSelect(byId('train-resume-run'), options);
    byId('mode-resume').disabled = resumable.length === 0;
    let note = '';
    if (data.hf && data.hf.ok && data.hf.stable === false) note = 'There are no stable weights on Hugging Face yet: publish a model as “stable” first.';
    else if (data.hf && !data.hf.ok) note = 'Cannot tell whether Hugging Face has stable weights (' + data.hf.error + ').';
    byId('continue-note').textContent = note;
    // No stable weights: "Continue" cannot start, so it cannot be chosen either.
    const noStable = Boolean(data.hf && data.hf.ok && data.hf.stable === false);
    byId('mode-continue').disabled = noStable;
    for (const id of ['mode-continue', 'mode-resume']) {
      if (byId(id).disabled && byId(id).checked) document.querySelector('input[name="train-mode"][value="fresh"]').checked = true;
    }
    renderWandbChoice();
    renderTrainWizard();
  }

  // -- wizards ---------------------------------------------------------------------------------
  // One card, one question per step: <id>-step-N sections, the <id>-back arrow and the <id>-dots.
  // onShow(step) renders what the step shows; moving forward focuses the title of the new step.
  function wizard(id, count, onShow) {
    const self = {step: 1};
    self.show = (step, focus) => {
      self.step = step;
      for (let index = 1; index <= count; index++) byId(id + '-step-' + index).hidden = index !== step;
      byId(id + '-back').hidden = step === 1;
      const dots = byId(id + '-dots').children;
      for (let index = 0; index < dots.length; index++) dots[index].className = index === step - 1 ? 'on' : '';
      if (onShow) onShow(step);
      updateButtons();
      // The first title of the step not inside a hidden block (Publish has one per destination).
      const heading = Array.from(byId(id + '-step-' + step).querySelectorAll('.wizard-title'))
        .find((title) => !title.closest('[hidden]'));
      if (focus !== false && heading && step > 1) {
        heading.tabIndex = -1;
        heading.focus({preventScroll: true});
        byId(id + '-step-1').closest('.wizard').scrollIntoView({block: 'start', behavior: 'smooth'});
      }
    };
    byId(id + '-back').addEventListener('click', () => self.show(Math.max(1, self.step - 1)));
    for (let index = 1; index < count; index++) {
      const next = byId(id + '-next-' + index);
      if (next) next.addEventListener('click', () => self.show(index + 1));
    }
    return self;
  }

  // Choice cards (label.choice around a radio): the chosen one is lit, a disabled one is dimmed.
  function renderChoices(container) {
    for (const label of container.querySelectorAll('.choice')) {
      const input = label.querySelector('input');
      label.classList.toggle('on', input.checked);
      label.classList.toggle('off', input.disabled);
    }
  }
  // Preset buttons (role=radio) that hold the value of a hidden input.
  function presetGroup(groupId, inputId, onChange) {
    const buttons = Array.from(byId(groupId).querySelectorAll('.preset'));
    function render() {
      for (const button of buttons) {
        const on = button.dataset.value === byId(inputId).value;
        button.setAttribute('aria-checked', String(on));
        button.tabIndex = on ? 0 : -1;
      }
    }
    function choose(button) {
      byId(inputId).value = button.dataset.value;
      render();
      if (onChange) onChange();
    }
    for (const button of buttons) {
      button.addEventListener('click', () => choose(button));
      button.addEventListener('keydown', (event) => {
        const step = {ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1}[event.key];
        if (!step) return;
        event.preventDefault();
        const next = buttons[(buttons.indexOf(button) + step + buttons.length) % buttons.length];
        next.focus();
        choose(next);
      });
    }
    render();
  }

  // -- new training wizard ----------------------------------------------------------------------
  // Step 1: where to start; step 2: how much (dial and presets, or the run to resume); step 3: summary.
  const DIAL = {min: 10, max: 10000, cx: 120, cy: 112, r: 88, sweep: 240};
  const PRESETS = {trial: 1000, standard: 5000};
  let customOpen = false;
  const trainWizard = wizard('train', 3, () => renderTrainWizard());

  function dialPoint(fraction, radius) {
    const angle = (-DIAL.sweep / 2 + DIAL.sweep * fraction) * Math.PI / 180;
    return [DIAL.cx + radius * Math.sin(angle), DIAL.cy - radius * Math.cos(angle)];
  }
  function dialArc(fraction) {
    const start = dialPoint(0, DIAL.r);
    const end = dialPoint(fraction, DIAL.r);
    const large = DIAL.sweep * fraction > 180 ? 1 : 0;
    return 'M' + start[0].toFixed(1) + ' ' + start[1].toFixed(1) + 'A' + DIAL.r + ' ' + DIAL.r + ' 0 ' + large + ' 1 '
      + end[0].toFixed(1) + ' ' + end[1].toFixed(1);
  }
  function drawDialTicks() {
    const group = byId('dial-ticks');
    if (group.childNodes.length) return;
    for (let index = 0; index <= 10; index++) {
      const outer = dialPoint(index / 10, DIAL.r - 18);
      const inner = dialPoint(index / 10, DIAL.r - (index % 5 === 0 ? 28 : 24));
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', outer[0].toFixed(1));
      line.setAttribute('y1', outer[1].toFixed(1));
      line.setAttribute('x2', inner[0].toFixed(1));
      line.setAttribute('y2', inner[1].toFixed(1));
      group.appendChild(line);
    }
    byId('dial-track').setAttribute('d', dialArc(1));
  }

  function trainSteps() {
    const value = Number(byId('train-steps').value);
    return isFinite(value) && value > 0 ? Math.round(value) : 0;
  }
  function setTrainSteps(value) {
    byId('train-steps').value = String(value);
    renderTrainWizard();
    updateButtons();
  }
  function snapSteps(raw) {
    const value = raw < 100 ? Math.round(raw / 10) * 10 : Math.round(raw / 100) * 100;
    return Math.min(DIAL.max, Math.max(DIAL.min, value));
  }
  function presetOf(steps) {
    if (customOpen) return 'custom';
    for (const [name, value] of Object.entries(PRESETS)) if (steps === value) return name;
    return 'custom';
  }
  function presetName(preset) {
    return {trial: 'Trial', standard: 'Standard', custom: 'Custom'}[preset];
  }

  function duration(seconds) {
    if (seconds < 90) return 'about a minute';
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return 'about ' + minutes + ' min';
    const hours = Math.floor(minutes / 60);
    const rest = Math.round((minutes % 60) / 5) * 5;
    return 'about ' + hours + ' h' + (rest ? ' ' + rest + ' min' : '');
  }
  // Seconds per step on this server: the latest run with the same batch, else the latest one.
  function stepSeconds() {
    const runs = (data.runs || []).filter((run) => run.avg_step_time && run.steps_done >= 20);
    runs.sort((a, b) => (b.started || 0) - (a.started || 0));
    const batch = Number(byId('train-batch').value);
    const same = runs.find((run) => run.config && Number(run.config.batch_size) === batch);
    const run = same || runs[0];
    return run ? run.avg_step_time : null;
  }

  function gpuName() {
    const gpus = (status && status.system.gpus) || [];
    const chosen = byId('train-gpus').value;
    if (chosen === 'all') return gpus.length + ' × ' + (gpus[0] ? gpus[0].name.replace(/-PCIE.*|-SXM.*/, '') : 'GPU');
    const gpu = gpus.find((item) => String(item.index) === chosen);
    if (!gpu) return 'No GPU found';
    return gpu.name.replace(/-PCIE.*|-SXM.*/, '');
  }

  function renderTrainWizard() {
    const mode = trainMode();
    renderChoices(byId('train-step-1'));
    const resume = mode === 'resume';
    byId('train-amount').hidden = resume;
    byId('train-resume-box').hidden = !resume;
    byId('train-step-2-title').textContent = resume ? 'Which run?' : 'How much to train?';
    byId('train-step-2-sub').textContent = resume ? 'Pick the run that stopped.' : 'Choose a session.';
    byId('train-steps-field').hidden = resume;
    byId('train-teacher-field').hidden = resume;
    byId('train-custom').hidden = !customOpen;
    byId('train-hw').setAttribute('aria-expanded', customOpen ? 'true' : 'false');
    byId('train-hw-name').textContent = gpuName();
    const defaults = byId('train-batch').value === '8' && byId('train-workers').value === '6'
      && byId('train-wandb').value === '' && byId('train-teacher').value === 'md-v3';
    byId('train-hw-mode').textContent = customOpen ? 'Hide the settings' : (defaults ? 'Automatic setup' : 'Custom setup');

    drawDialTicks();
    const steps = trainSteps();
    const fraction = Math.min(1, Math.max(0, (steps - DIAL.min) / (DIAL.max - DIAL.min)));
    byId('dial-arc').setAttribute('d', fraction > 0.002 ? dialArc(fraction) : '');
    const knob = dialPoint(fraction, DIAL.r);
    byId('dial-knob').setAttribute('transform', 'translate(' + knob[0].toFixed(1) + ' ' + knob[1].toFixed(1) + ')');
    const dial = byId('train-dial');
    dial.setAttribute('aria-valuenow', String(Math.min(steps, DIAL.max)));
    dial.setAttribute('aria-valuetext', steps.toLocaleString('en-US') + ' steps');
    const preset = presetOf(steps);
    byId('dial-name').textContent = presetName(preset);
    byId('dial-value').textContent = steps ? steps.toLocaleString('en-US') + ' steps' : 'Write the steps';
    for (const button of document.querySelectorAll('#train-presets .preset')) {
      const on = button.dataset.preset === preset;
      button.setAttribute('aria-checked', on ? 'true' : 'false');
      button.tabIndex = on ? 0 : -1;
    }
    const seconds = stepSeconds();
    byId('train-eta').textContent = seconds && steps
      ? 'On this server: ' + duration(seconds * steps) + '.'
      : 'More steps take more time.';

    if (trainWizard.step === 3) renderTrainSummary();
  }

  function renderTrainSummary() {
    const mode = trainMode();
    const list = byId('train-summary');
    list.replaceChildren();
    const start = {fresh: 'New model (from scratch)', continue: 'Continue from the stable model',
      resume: 'Resume ' + (byId('train-resume-run').value || '–')}[mode];
    kvRow(list, 'Start', start);
    if (mode !== 'resume') {
      const steps = trainSteps();
      const seconds = stepSeconds();
      kvRow(list, 'Session', presetName(presetOf(steps)) + ' · ' + steps.toLocaleString('en-US') + ' steps'
        + (seconds && steps ? ' · ' + duration(seconds * steps) : ''));
      kvRow(list, 'Teacher', byId('train-teacher').selectedOptions[0].textContent);
    }
    const gpuOption = byId('train-gpus').selectedOptions[0];
    kvRow(list, 'Hardware', (gpuOption ? gpuOption.textContent : 'no GPU') + ' · batch ' + byId('train-batch').value
      + ' · ' + byId('train-workers').value + ' workers');
    kvRow(list, 'Charts', byId('train-wandb').selectedOptions[0].textContent);
  }

  byId('train-hw').addEventListener('click', () => {
    customOpen = !customOpen;
    renderTrainWizard();
  });
  for (const id of ['train-steps', 'train-batch', 'train-workers', 'train-teacher', 'train-wandb', 'train-gpus']) {
    byId(id).addEventListener('input', renderTrainWizard);
    byId(id).addEventListener('change', renderTrainWizard);
  }

  const presetButtons = Array.from(document.querySelectorAll('#train-presets .preset'));
  function choosePreset(preset) {
    customOpen = preset === 'custom';
    if (PRESETS[preset]) setTrainSteps(PRESETS[preset]);
    else renderTrainWizard();
    if (customOpen) byId('train-steps').focus();
  }
  for (const button of presetButtons) {
    button.addEventListener('click', () => choosePreset(button.dataset.preset));
    button.addEventListener('keydown', (event) => {
      const step = {ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1}[event.key];
      if (!step) return;
      event.preventDefault();
      const next = presetButtons[(presetButtons.indexOf(button) + step + presetButtons.length) % presetButtons.length];
      next.focus();
      choosePreset(next.dataset.preset);
    });
  }

  // The dial: drag the knob (or anywhere on the arc), or use the arrow keys.
  const dialElement = byId('train-dial');
  function dialFromPointer(event) {
    const svg = dialElement.querySelector('svg');
    const box = svg.getBoundingClientRect();
    const x = (event.clientX - box.left) * 240 / box.width;
    const y = (event.clientY - box.top) * 200 / box.height;
    let angle = Math.atan2(x - DIAL.cx, DIAL.cy - y) * 180 / Math.PI;
    angle = Math.max(-DIAL.sweep / 2, Math.min(DIAL.sweep / 2, angle));
    const fraction = (angle + DIAL.sweep / 2) / DIAL.sweep;
    setTrainSteps(snapSteps(DIAL.min + fraction * (DIAL.max - DIAL.min)));
  }
  dialElement.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    dialElement.setPointerCapture(event.pointerId);
    dialElement.classList.add('dragging');
    dialFromPointer(event);
  });
  dialElement.addEventListener('pointermove', (event) => {
    if (dialElement.hasPointerCapture(event.pointerId)) dialFromPointer(event);
  });
  for (const name of ['pointerup', 'pointercancel']) {
    dialElement.addEventListener(name, () => dialElement.classList.remove('dragging'));
  }
  dialElement.addEventListener('keydown', (event) => {
    const steps = Math.min(trainSteps() || DIAL.min, DIAL.max);
    const change = {ArrowRight: 100, ArrowUp: 100, ArrowLeft: -100, ArrowDown: -100, PageUp: 1000, PageDown: -1000}[event.key];
    let value = null;
    if (change) value = snapSteps(steps + change);
    else if (event.key === 'Home') value = DIAL.min;
    else if (event.key === 'End') value = DIAL.max;
    if (value === null) return;
    event.preventDefault();
    setTrainSteps(value);
  });

  // The wandb choice says what "automatic" does with the account of Settings, and points there.
  function renderWandbChoice() {
    const account = accountOf('wandb');
    // A server without accounts in its status: the choice as it always was.
    const connected = !account || account.connected;
    const select = byId('train-wandb');
    const automatic = select.querySelector('option[value=""]');
    const online = select.querySelector('option[value="online"]');
    automatic.textContent = connected ? 'automatic (online: W&B connected)' : 'automatic (offline: W&B not connected)';
    online.textContent = connected ? 'online (wandb.ai)' : 'online (connect W&B in Settings)';
    online.disabled = !connected;
    byId('wandb-callout').hidden = connected;
    if (!connected) {
      byId('wandb-callout-text').textContent = account.problem && account.problem.indexOf('no longer works') >= 0
        ? 'The saved Weights & Biases API key no longer works: runs are logged on this server only.'
        : 'Weights & Biases is not connected: runs are logged on this server only.';
    }
  }
  byId('train-wandb').addEventListener('change', updateButtons);

  function renderTraining() {
    renderTrainForm();
    const running = runningRun();
    // One training at a time: while one runs, its detail takes the place of the form.
    byId('train-new').hidden = running !== null;
    const name = viewedRun();
    byId('run-view').hidden = name === null;
    if (name) renderRunView(name);
    renderRunsTable(name);
    renderCharts();
  }

  function kvRow(list, label, value) {
    list.appendChild(node('dt', '', label));
    list.appendChild(node('dd', '', value === null || value === undefined || value === '' ? '–' : String(value)));
  }

  function renderRunView(name) {
    const run = (data.runs || []).find((item) => item.name === name);
    const detail = data.detail && data.detail.name === name ? data.detail : null;
    byId('run-back').hidden = !openRun;
    byId('run-view-name').textContent = name;
    if (!run) {
      byId('run-view-title').textContent = 'Run';
      return;
    }
    const isRunning = run.status === 'running';
    byId('run-view-title').textContent = isRunning ? 'Training in progress' : 'Run';
    const state = RUN_STATE[run.status] || [run.status, 'muted'];
    byId('run-view-state').replaceChildren(pill(state[0], state[1]));

    setBar('run-view-bar', run.percent, isRunning ? 'run' : (run.status === 'finished' ? '' : 'warn'));
    let steps = 'step ' + run.steps_done + (run.steps_total ? ' / ' + run.steps_total : '');
    if (run.percent !== null) steps += ' · ' + run.percent + '%';
    if (isRunning && run.eta_seconds !== null) steps += ' · ' + fmtDuration(run.eta_seconds) + ' left';
    if (run.avg_step_time !== null) steps += ' · ' + run.avg_step_time.toFixed(2) + ' s/step';
    byId('run-view-steps').textContent = steps;

    const job = run.job_id && status ? status.jobs.find((item) => item.id === run.job_id) : null;
    if (!same('run-buttons', [name, run.status, run.job_id, !!job, run.resumable, run.has_pt, run.has_training_pt])) {
      const box = byId('run-view-buttons');
      box.replaceChildren();
      function button(label, kind, handler) {
        const element = node('button', 'btn' + (kind ? ' ' + kind : ''), label);
        element.type = 'button';
        element.addEventListener('click', handler);
        box.appendChild(element);
      }
      if (isRunning && job) {
        button('Live log', '', () => openLog(job.id, job.label));
        button('Stop', 'danger', () => stopJob(job));
      }
      if (run.resumable) {
        button('Resume', 'primary', () => {
          const config = run.config || {};
          const params = {mode: 'resume', run: name, batch_size: config.batch_size || 8,
            workers: config.workers !== undefined ? config.workers : 6, gpus: byId('train-gpus').value || 'all'};
          runAction('train', params, 'Resume run ' + name + ' from its last checkpoint. Start?');
        });
      }
      if (run.has_pt || run.has_training_pt) {
        button('Evaluate', '', () => goTo('evaluation', 'eval-model', 'run:' + name));
        button('Comparison video', '', () => goTo('compare', 'cmp-student', 'run:' + name));
      }
      if (run.has_pt) button('Install or publish', '', () => goTo('publish', 'publish-run', 'run:' + name));
    }

    const config = run.config || {};
    const settings = byId('run-view-settings');
    settings.replaceChildren();
    kvRow(settings, 'Start', MODE_TEXT[config.mode] || config.mode);
    kvRow(settings, 'Teacher', teacherName(config.teacher));
    kvRow(settings, 'Steps', run.steps_total);
    kvRow(settings, 'Batch per GPU', config.batch_size);
    kvRow(settings, 'Workers', config.workers);
    let gpus = null;
    if (config.gpus) gpus = 'GPU ' + config.gpus;
    else if (config.world_size) gpus = config.world_size + ' GPU';
    kvRow(settings, 'GPU', gpus);
    kvRow(settings, 'Precision', config.precision);
    kvRow(settings, 'Started', fmtTime(run.started));
    if (run.finished) kvRow(settings, 'Finished', fmtTime(run.finished) + ' · took ' + fmtDuration(run.finished - run.started));
    if (detail && detail.resumed.length) kvRow(settings, 'Resumed', detail.resumed.length + (detail.resumed.length === 1 ? ' time' : ' times'));
    const files = [];
    if (run.has_pt) files.push('final model');
    if (run.has_onnx) files.push('ONNX');
    if (run.has_training_pt) files.push('checkpoint');
    kvRow(settings, 'Files', files.join(' · ') || 'none');

    const last = byId('run-view-last');
    last.replaceChildren();
    for (const key of ['loss', 'gt', 'distill', 'gt/plan', 'grad_norm']) {
      kvRow(last, (CHART_TITLES[key] || key) + ' (last step)', run.last ? fmt(run.last[key]) : null);
    }

    const list = byId('run-view-log');
    const text = detail ? detail.log_tail : '';
    if (!same('run-log', [name, text])) {
      const atEnd = list.scrollTop + list.clientHeight >= list.scrollHeight - 20;
      renderMessages(list, readLog(text).messages, 'No messages yet.');
      if (atEnd) list.scrollTop = list.scrollHeight;
    }
  }

  function renderRunsTable(viewed) {
    const runs = data.runs || [];
    if (same('runs-table', [runs, viewed])) return;
    const body = byId('runs');
    body.replaceChildren();
    byId('runs-empty').hidden = runs.length > 0;
    for (const run of runs) {
      const row = node('tr', 'clickable' + (run.name === viewed ? ' selected' : ''));
      row.tabIndex = 0;
      row.setAttribute('role', 'link');
      row.setAttribute('aria-label', 'Details of run ' + run.name);
      row.addEventListener('click', () => openRunView(run.name));
      row.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); openRunView(run.name); }
      });
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
        row.appendChild(node('td', 'num wide', run.last ? fmt(run.last[key]) : '–'));
      }
      const files = [];
      if (run.has_pt) files.push('pt');
      if (run.has_onnx) files.push('onnx');
      if (run.has_training_pt && !run.has_pt) files.push('checkpoint');
      row.appendChild(node('td', 'note wide', files.join(' · ') || '–'));
      body.appendChild(row);
    }
  }

  async function loadDetail(force) {
    const name = viewedRun();
    if (!name) {
      data.detail = null;
      return;
    }
    const stale = !data.detail || data.detail.name !== name;
    // A finished run does not change: read it once; a running one at every tab refresh.
    if (!force && !stale && data.detail.status !== 'running') return;
    try {
      data.detail = await api('/api/runs/' + encodeURIComponent(name));
    } catch (error) {
      toast(error.message, true);
      if (openRun === name) openRun = null;
    }
  }

  async function loadMetrics(force) {
    selectedRun = viewedRun();
    if (!selectedRun) return;
    const run = (data.runs || []).find((item) => item.name === selectedRun);
    const stale = !data.metrics || data.metrics.name !== selectedRun;
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
    const metrics = data.metrics;
    const smooth = byId('chart-smooth').checked;
    const logScale = byId('chart-log').checked;
    if (!metrics || metrics.name !== selectedRun) return;
    if (same('charts', [metrics, smooth, logScale])) return;
    const box = byId('charts');
    box.replaceChildren();
    byId('charts-empty').hidden = metrics.steps.length > 0;
    let note = '';
    if (metrics.bucket > 1) note = 'each point is the mean of ' + metrics.bucket + ' steps';
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
      box.appendChild(node('p', 'empty', 'no values'));
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
    lastText.textContent = 'last ' + fmt(shown[shown.length - 1][1]);

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
    svg.setAttribute('aria-label', title + ', last value ' + fmt(shown[shown.length - 1][1]));
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
        if (/^[a-z0-9_]+$/.test(name)) options.push({value: 'hf:' + name, label: 'HF ' + name, group: 'Published on Hugging Face'});
      }
    }
    return options;
  }
  function runModelOptions() {
    const options = [];
    for (const run of runsWith((item) => item.has_pt || item.has_training_pt)) {
      options.push({value: 'run:' + run.name, label: 'run ' + run.name + (run.has_pt ? '' : ' (checkpoint, not finished)'), group: 'Your models'});
    }
    return options;
  }

  // -- model picker ----------------------------------------------------------------------------
  // A tappable list in place of a <select> of model specs (the select stays, hidden, and holds the
  // value): every row says what the model is and, when it was evaluated, how it did.
  const REFERENCE_INFO = {
    stock: ['comma model (stock)', 'The model on your comma today: the one to beat.'],
    teacher: ['openpilot big model (teacher)', 'MD v3, the large model your models learn from.'],
    logged: ['What the car actually did', 'The path recorded in the drive logs; no model is run.'],
  };

  function fmtDay(unix) {
    if (!unix) return '';
    // '23\u00a0Oct, 18:10': a narrow column breaks after the comma, not inside the date.
    return new Date(unix * 1000).toLocaleString('en-GB', {day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'})
      .replace(/^(\d+) /, '$1\u00a0');
  }

  function evalResultOf(spec) {
    return (data.evalResults || []).find((result) => result.model === spec && (!result.selection
      || result.selection === 'heldout')) || null;
  }

  function modelInfo(spec) {
    if (REFERENCE_INFO[spec]) return {title: REFERENCE_INFO[spec][0], detail: REFERENCE_INFO[spec][1], badge: ''};
    if (spec.startsWith('run:')) {
      const name = spec.slice(4);
      const run = (data.runs || []).find((item) => item.name === name);
      if (!run) return {title: 'Model ' + name, detail: '', badge: ''};
      const details = [];
      if (run.config && run.config.teacher) details.push('teacher\u00a0' + teacherName(run.config.teacher));
      if (run.steps_total) details.push(run.steps_total.toLocaleString('en-GB') + '\u00a0steps');
      if (run.last && run.last.loss !== null && run.last.loss !== undefined) details.push('loss\u00a0' + fmt(run.last.loss));
      if (!run.has_pt) details.push('not finished');
      const latest = runsWith((item) => item.has_pt)[0];
      return {title: 'Your model · ' + (fmtDay(run.started) || name), short: fmtDay(run.started) || name,
        detail: details.join(' · '), badge: latest && latest.name === name ? 'Latest' : ''};
    }
    if (spec.startsWith('hf:')) {
      return {title: 'Published · ' + spec.slice(3), short: spec.slice(3), detail: 'Downloaded from Hugging Face', badge: ''};
    }
    return {title: spec, detail: '', badge: ''};
  }

  // A long group shows its first rows (and the chosen one) until "Show more"; which groups are open.
  const PICKER_ROWS = 4;
  const pickerMore = {};
  // Pickers that show only their first group(s); the others wait behind one button (Evaluation: two
  // lists side by side, or one under the other on a phone, must stay short).
  const PICKER_MAIN = {'eval-model': 1, 'eval-against': 1};
  // What an empty picker says.
  const PICKER_EMPTY = {'publish-run': 'No finished training yet: start one in Training.'};

  function pickerRowContent(row, spec, checked) {
    const info = modelInfo(spec);
    const text = node('div', 'picker-text');
    // Under its group heading a row needs no 'Your model ·' or 'Published ·' in front.
    const title = node('div', 'picker-title', info.short || info.title);
    if (info.badge) title.appendChild(node('span', 'picker-badge', info.badge));
    text.appendChild(title);
    if (info.detail) text.appendChild(node('div', 'picker-detail', info.detail));
    row.appendChild(text);
    const result = evalResultOf(spec);
    const score = node('div', 'picker-score');
    if (result && result.avg_miss_m !== null) {
      score.appendChild(node('strong', '', result.avg_miss_m.toFixed(2) + '\u00a0m'));
      score.appendChild(node('small', '', 'avg miss'));
    } else {
      score.appendChild(node('small', '', 'not evaluated'));
    }
    row.appendChild(score);
    if (checked !== undefined) row.appendChild(node('span', 'picker-check', checked ? '\u2713' : ''));
  }

  function renderPicker(selectId) {
    const select = byId(selectId);
    const box = byId(selectId + '-picker');
    const options = Array.from(select.options).map((option) => ({value: option.value,
      group: option.parentElement.tagName === 'OPTGROUP' ? option.parentElement.label : ''}));
    if (same('picker:' + selectId, [options, select.value, data.runs, data.evalResults, pickerMore])) return;
    // The rebuilt row or button under the finger keeps the keyboard focus.
    const focused = box.contains(document.activeElement) ? document.activeElement.dataset.key : null;
    box.replaceChildren();
    if (!options.length) {
      box.appendChild(node('p', 'empty', PICKER_EMPTY[selectId] || 'No models yet.'));
      return;
    }

    // Every model in sight, grouped: a tap chooses it, nothing opens or closes.
    const groups = [];
    for (const option of options) {
      if (!groups.length || groups[groups.length - 1].name !== option.group) groups.push({name: option.group, options: []});
      groups[groups.length - 1].options.push(option);
    }
    const main = PICKER_MAIN[selectId] || groups.length;
    const restKey = selectId + '|rest';
    // The groups after the main ones that do not hold the choice, and how many rows they show when open.
    const rest = groups.filter((group, groupIndex) => groupIndex >= main
      && !group.options.some((option) => option.value === select.value));
    const restRows = rest.reduce((sum, group) => sum + (group.options.length > PICKER_ROWS + 1
      && !pickerMore[selectId + '|' + group.name] ? PICKER_ROWS : group.options.length), 0);
    groups.forEach((group) => {
      if (!pickerMore[restKey] && rest.indexOf(group) >= 0) return;
      if (group.name) box.appendChild(node('div', 'picker-group', group.name));
      const list = node('div', 'picker-list');
      const key = selectId + '|' + group.name;
      const long = group.options.length > PICKER_ROWS + 1;
      let hidden = 0;
      group.options.forEach((option, position) => {
        const checked = option.value === select.value;
        if (long && !pickerMore[key] && position >= PICKER_ROWS && !checked) {
          hidden = hidden + 1;
          return;
        }
        const row = node('button', 'picker-row');
        row.type = 'button';
        row.setAttribute('role', 'radio');
        row.setAttribute('aria-checked', String(checked));
        row.dataset.key = option.value;
        pickerRowContent(row, option.value, checked);
        row.addEventListener('click', () => {
          if (select.value === option.value) return;
          select.value = option.value;
          select.dispatchEvent(new Event('change'));
        });
        list.appendChild(row);
      });
      if (long) {
        const more = node('button', 'picker-more', pickerMore[key] ? 'Show fewer'
          : 'Show ' + hidden + ' more');
        more.type = 'button';
        more.setAttribute('aria-expanded', String(!!pickerMore[key]));
        more.dataset.key = 'more|' + key;
        more.addEventListener('click', () => {
          pickerMore[key] = !pickerMore[key];
          renderPicker(selectId);
        });
        list.appendChild(more);
      }
      box.appendChild(list);
    });
    if (rest.length) {
      const other = node('button', 'picker-more picker-other', pickerMore[restKey] ? 'Hide the other models'
        : 'Show ' + restRows + (restRows === 1 ? ' other model' : ' other models'));
      other.type = 'button';
      other.setAttribute('aria-expanded', String(!!pickerMore[restKey]));
      other.dataset.key = 'more|' + restKey;
      other.addEventListener('click', () => {
        const hiding = pickerMore[restKey];
        pickerMore[restKey] = !hiding;
        renderPicker(selectId);
        // Hiding a long list can leave the button above the screen (one column under the other on a phone).
        const again = box.querySelector('.picker-other');
        const bar = document.querySelector('.topbar');
        if (hiding && again && again.getBoundingClientRect().top < (bar ? bar.getBoundingClientRect().bottom : 0)) {
          again.scrollIntoView({block: 'center'});
        }
      });
      box.appendChild(other);
    }
    const target = focused ? Array.from(box.querySelectorAll('button')).find((item) => item.dataset.key === focused) : null;
    if (target) target.focus({preventScroll: true});
  }
  for (const id of ['eval-against', 'cmp-student']) {
    byId(id).addEventListener('change', () => { renderPicker(id); updateButtons(); });
  }
  // The model on the left leaves the list on the right: renderEval fills it again.
  byId('eval-model').addEventListener('change', () => { renderEval(); updateButtons(); });
  // The model on the left at the last render: if the left takes the right's model, this one goes right.
  let evalShownModel = null;

  // -- Evaluation -------------------------------------------------------------------------------
  // Two columns: the model to test (yours first) and the one to compare it with (the references first).
  function renderEval() {
    // A status poll can land before /api/runs: no defaults until your models are known.
    if (data.runs === null) {
      renderResults(data.evalResults || []);
      return;
    }
    const runs = runModelOptions();
    const published = hfModelOptions();
    const references = ['stock', 'teacher', 'logged'].map((value) => ({value: value, label: value, group: 'Reference models'}));
    fillSelect(byId('eval-model'), runs.concat(published, references), runs.length ? runs[0].value : 'stock');
    const model = byId('eval-model').value;
    const others = references.concat(runs, published).filter((option) => option.value !== model);
    let preferred = model === 'stock' ? 'teacher' : 'stock';
    if (evalShownModel && evalShownModel !== model && others.some((option) => option.value === evalShownModel)) {
      preferred = evalShownModel;
    }
    // fillSelect takes `preferred` only when the right's model left its list: that is, the swap.
    fillSelect(byId('eval-against'), others, preferred);
    evalShownModel = model;
    renderPicker('eval-model');
    renderPicker('eval-against');
    if (evalWizard.step === 2) renderEvalSummary();
    renderResults(data.evalResults || []);
  }

  // -- evaluate a model: a wizard, the two models (step 1) then the summary (step 2) --
  const evalWizard = wizard('eval', 2, (step) => { if (step === 2) renderEvalSummary(); });
  function renderEvalSummary() {
    const spec = byId('eval-model').value;
    const against = byId('eval-against').value;
    const list = byId('eval-summary');
    list.replaceChildren();
    kvRow(list, 'Model to test', spec ? modelInfo(spec).title : '–');
    kvRow(list, 'Compared with', against ? modelInfo(against).title : '–');
    const counts = data.segments ? data.segments.counts : null;
    kvRow(list, 'Test drives', counts ? counts.heldout_localized + ' hold-out segments, never seen in training' : '–');
    const last = [spec, against].map((value) => {
      const result = value ? evalResultOf(value) : null;
      return result && result.avg_miss_m !== null ? result.avg_miss_m.toFixed(2) + '\u00a0m' : 'not evaluated';
    });
    if (last[0] !== 'not evaluated' || last[1] !== 'not evaluated') kvRow(list, 'Last time', last.join(' vs '));
    kvRow(list, 'Hardware', 'the GPU, a few minutes per model; a result already computed on these drives is reused');
  }

  // Colours of the models in the results, in ranking order (Apple system colours).
  const MODEL_COLORS = ['#34c759', '#007aff', '#ff9500', '#af52de', '#ff2d55', '#5ac8fa', '#a2845e'];

  function renderResults(all) {
    // The default evaluation only (hold-out, 5 Hz): the others are not comparable with it.
    const results = all.filter((result) => (!result.selection || result.selection === 'heldout')
      && (!result.rate_hz || result.rate_hz === 5) && result.avg_miss_m !== null);
    if (same('leaderboard', [results, results.map((result) => modelInfo(result.model).title)])) return;
    const box = byId('leaderboard');
    box.replaceChildren();
    byId('leaderboard-empty').hidden = results.length > 0;
    const best = results.length ? results[0].avg_miss_m : 0;
    const worst = results.length ? results[results.length - 1].avg_miss_m : 1;
    results.forEach((result, index) => {
      const color = MODEL_COLORS[index % MODEL_COLORS.length];
      const row = node('div', 'rank-row');
      row.style.setProperty('--i', String(index));
      const head = node('div', 'rank-head');
      head.appendChild(node('span', 'rank-pos', String(index + 1)));
      head.appendChild(node('span', 'rank-name', modelInfo(result.model).title));
      const value = node('span', 'rank-value', result.avg_miss_m.toFixed(2));
      value.appendChild(node('small', '', ' m'));
      head.appendChild(value);
      row.appendChild(head);
      const track = node('div', 'rank-track');
      const fill = node('span', 'rank-fill');
      fill.style.background = color;
      fill.style.setProperty('--w', Math.max(4, result.avg_miss_m * 100 / worst) + '%');
      track.appendChild(fill);
      row.appendChild(track);
      const notes = [];
      if (index === 0) notes.push('best');
      else notes.push((result.avg_miss_m / best).toFixed(1) + '\u00d7 the error of the best');
      if (result.lateral_miss_m !== null) notes.push('sideways ' + result.lateral_miss_m.toFixed(2) + '\u00a0m');
      if (result.segments !== null) notes.push(result.segments + ' drives');
      notes.push(fmtDay(typeof result.date === 'string' ? Date.parse(result.date) / 1000 : result.date));
      row.appendChild(node('div', 'rank-notes', notes.join(' \u00b7 ')));
      box.appendChild(row);
    });
    renderPerDrive(results);
  }

  // A dot plot: one line per hold-out segment, one dot per model, on a shared metre scale.
  function renderPerDrive(results) {
    const withSegments = results.filter((result) => result.per_segment && result.per_segment.length);
    byId('per-drive-card').hidden = withSegments.length === 0;
    if (!withSegments.length) return;
    const legend = byId('per-drive-legend');
    legend.replaceChildren();
    const segments = [];
    let maximum = 0;
    withSegments.forEach((result) => {
      const index = results.indexOf(result);
      const item = node('li');
      const dot = node('span', 'dot-swatch');
      dot.style.background = MODEL_COLORS[index % MODEL_COLORS.length];
      item.appendChild(dot);
      item.appendChild(document.createTextNode(modelInfo(result.model).title));
      legend.appendChild(item);
      for (const entry of result.per_segment) {
        if (segments.indexOf(entry.segment) < 0) segments.push(entry.segment);
        if (entry.avg_miss_m !== null) maximum = Math.max(maximum, entry.avg_miss_m);
      }
    });
    const scale = Math.ceil(maximum / 5) * 5 || 5;
    const plot = byId('per-drive');
    plot.replaceChildren();
    segments.sort();
    segments.forEach((segment, row) => {
      const line = node('div', 'dot-row');
      line.style.setProperty('--i', String(row));
      line.appendChild(node('span', 'dot-label', segment.slice(0, 8) + '\u2026' + ' #' + Number(segment.slice(18))));
      const lane = node('div', 'dot-lane');
      const values = [];
      withSegments.forEach((result) => {
        const entry = result.per_segment.find((item) => item.segment === segment);
        if (!entry || entry.avg_miss_m === null) return;
        const index = results.indexOf(result);
        const dot = node('span', 'plot-dot');
        dot.style.left = (entry.avg_miss_m * 100 / scale) + '%';
        dot.style.background = MODEL_COLORS[index % MODEL_COLORS.length];
        dot.title = modelInfo(result.model).title + ': ' + entry.avg_miss_m.toFixed(2) + ' m';
        lane.appendChild(dot);
        values.push(entry.avg_miss_m);
      });
      if (values.length > 1) {
        const span = node('span', 'dot-span');
        span.style.left = (Math.min(...values) * 100 / scale) + '%';
        span.style.width = ((Math.max(...values) - Math.min(...values)) * 100 / scale) + '%';
        lane.prepend(span);
      }
      line.appendChild(lane);
      plot.appendChild(line);
    });
    const axis = node('div', 'dot-axis');
    for (let tick = 0; tick <= scale; tick += scale / 5) {
      const label = node('span', '', tick + '\u00a0m');
      label.style.left = (tick * 100 / scale) + '%';
      axis.appendChild(label);
    }
    plot.appendChild(axis);
  }
  // Step 2 is the confirmation: no extra question.
  byId('btn-eval').addEventListener('click', () => {
    runAction('eval', {model: byId('eval-model').value, against: byId('eval-against').value});
  });

  // -- drive picker --------------------------------------------------------------------------
  // The segments as drives a person recognises: grouped by outing (route), newest first, every
  // segment a card with a picture, the time of day, speed, how curvy and the shape of the road.
  function fmtClock(unix) {
    return new Date(unix * 1000).toLocaleTimeString('en-GB', {hour: '2-digit', minute: '2-digit'});
  }
  function fmtDate(unix) {
    return new Date(unix * 1000).toLocaleDateString('en-GB', {weekday: 'short', day: 'numeric', month: 'short'});
  }
  function roadKind(drive) {
    if (drive.avg_kmh !== undefined && drive.avg_kmh < 8) return 'standing';
    if (drive.turn_deg === undefined || !drive.distance_km) return '';
    // Degrees of turning per km of road.
    const bends = drive.turn_deg / Math.max(drive.distance_km, 0.2);
    if (bends < 45) return 'straight';
    if (bends < 160) return 'a few bends';
    return 'curvy';
  }

  function traceSvg(trace) {
    const SVG = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(SVG, 'svg');
    svg.setAttribute('viewBox', '-0.08 -0.08 1.16 1.16');
    svg.setAttribute('class', 'drive-trace');
    svg.setAttribute('aria-hidden', 'true');
    const line = document.createElementNS(SVG, 'polyline');
    // North up: y grows downwards on screen.
    line.setAttribute('points', trace.map((point) => point[0] + ',' + (1 - point[1])).join(' '));
    svg.appendChild(line);
    const end = document.createElementNS(SVG, 'circle');
    const last = trace[trace.length - 1];
    end.setAttribute('cx', last[0]);
    end.setAttribute('cy', 1 - last[1]);
    end.setAttribute('r', '0.07');
    svg.appendChild(end);
    return svg;
  }

  function renderDrives() {
    const select = byId('cmp-segment');
    const box = byId('drive-picker');
    const all = byId('drive-all').checked;
    const info = data.drives ? data.drives.drives : [];
    const pending = data.drives ? data.drives.pending : 0;
    if (same('drives', [info, pending, all, select.value])) return;
    const localized = new Set(Array.from(select.options).map((option) => option.value));
    const drives = info.filter((drive) => localized.has(drive.segment) && (all || drive.heldout));
    byId('drive-note').textContent = (pending ? 'Reading ' + pending + ' more drives… ' : '')
      + (all ? 'Drives used in training are easy for the model: the hold-out ones (★) are the fair test.'
        : 'Only the drives the models never saw in training: the fair test.');
    const routes = new Map();
    for (const drive of drives) {
      if (!routes.has(drive.route)) routes.set(drive.route, []);
      routes.get(drive.route).push(drive);
    }
    const ordered = Array.from(routes.values()).map((list) => list.sort((a, b) => a.number - b.number));
    ordered.sort((a, b) => (b[0].start || 0) - (a[0].start || 0));
    box.replaceChildren();
    if (!ordered.length) box.appendChild(node('p', 'empty', pending ? 'Reading the drives…' : 'No drives to show.'));
    ordered.forEach((list, groupIndex) => {
      const group = node('section', 'drive-group');
      group.style.setProperty('--i', String(groupIndex));
      const first = list[0];
      const last = list[list.length - 1];
      const head = node('div', 'drive-group-head');
      if (first.start) {
        head.appendChild(node('strong', '', fmtDate(first.start)));
        const end = (last.start || first.start) + (last.duration_s || 60);
        head.appendChild(node('span', '', fmtClock(first.start) + ' – ' + fmtClock(end) + ' · ' + list.length
          + (list.length === 1 ? ' minute shown' : ' minutes shown')));
      } else {
        head.appendChild(node('strong', '', 'Drive ' + first.route.slice(0, 8)));
      }
      group.appendChild(head);
      const strip = node('div', 'drive-strip');
      for (const drive of list) {
        const card = node('button', 'drive-card');
        card.type = 'button';
        card.setAttribute('role', 'radio');
        card.setAttribute('aria-checked', String(drive.segment === select.value));
        const picture = node('div', 'drive-picture');
        const image = document.createElement('img');
        image.loading = 'lazy';
        image.alt = '';
        image.src = '/thumb/' + drive.segment + '.jpg';
        picture.appendChild(image);
        if (drive.start) picture.appendChild(node('span', 'drive-time', fmtClock(drive.start)));
        if (drive.heldout) picture.appendChild(node('span', 'drive-star', '\u2605 never seen'));
        card.appendChild(picture);
        const body = node('div', 'drive-body');
        const text = node('div', 'drive-text');
        const facts = [];
        if (drive.avg_kmh !== undefined) facts.push(Math.round(drive.avg_kmh) + ' km/h');
        const kind = roadKind(drive);
        if (kind) facts.push(kind);
        text.appendChild(node('div', 'drive-facts', facts.join(' \u00b7 ') || 'minute ' + drive.number));
        if (drive.distance_km !== undefined) text.appendChild(node('div', 'drive-sub', drive.distance_km.toFixed(1) + ' km'));
        body.appendChild(text);
        if (drive.trace && drive.trace.length > 1) body.appendChild(traceSvg(drive.trace));
        card.appendChild(body);
        card.addEventListener('click', () => {
          select.value = drive.segment;
          select.dispatchEvent(new Event('change'));
          renderDrives();
        });
        strip.appendChild(card);
      }
      group.appendChild(strip);
      box.appendChild(group);
    });
    // Bring the chosen card into view inside its strip.
    const chosen = box.querySelector('.drive-card[aria-checked="true"]');
    if (chosen && chosen.parentElement) chosen.parentElement.scrollLeft = chosen.offsetLeft - 16;
    // Still computing: ask again soon.
    if (pending) setTimeout(() => { if (currentTab === 'compare') load('drives').then(renderDrives); }, 4000);
  }
  byId('drive-all').addEventListener('change', renderDrives);

  // -- Comparison video ---------------------------------------------------------------------------
  function compareFormReason() {
    if (!byId('cmp-segment').value) return 'no localized segment';
    if (!byId('cmp-student').value) return 'no student model (run or HF)';
    return '';
  }

  function renderCompare() {
    const segments = (data.segments ? data.segments.segments : []).filter((segment) => segment.state === 'localized');
    const options = [];
    let firstHeldout = null;
    for (const segment of segments) {
      if (segment.heldout) {
        options.push({value: segment.name, label: '★ ' + segment.name + ' (hold-out)', group: 'Hold-out: never seen in training'});
        if (!firstHeldout) firstHeldout = segment.name;
      }
    }
    for (const segment of segments) {
      if (!segment.heldout) options.push({value: segment.name, label: segment.name, group: 'Used in training'});
    }
    fillSelect(byId('cmp-segment'), options, firstHeldout);
    renderDrives();
    const students = runModelOptions().concat(hfModelOptions());
    fillSelect(byId('cmp-student'), students);
    renderPicker('cmp-student');
    renderReferences();
    renderMoment();

    const videos = data.videos || [];
    if (same('videos', videos)) return;
    const list = byId('videos');
    list.replaceChildren();
    byId('videos-empty').hidden = videos.length > 0;
    for (const video of videos) {
      const item = node('li', 'item');
      const text = node('div', 'grow');
      const who = (video.student.label || video.student.model || '?') + ' vs '
        + (video.reference.label || video.reference.model || '?');
      text.appendChild(node('div', 'what', video.has_json ? who : video.name));
      let sub = (video.segment || video.name) + ' · ' + fmtTime(video.modified) + ' · ' + fmtBytes(video.size);
      if (video.student.avg_miss_m !== null) sub += ' · student ' + video.student.avg_miss_m.toFixed(2) + '\u00a0m';
      if (video.reference.avg_miss_m !== null) sub += ' · reference ' + video.reference.avg_miss_m.toFixed(2) + '\u00a0m';
      text.appendChild(node('div', 'sub', sub));
      item.appendChild(text);
      const play = node('button', 'btn small', 'Watch');
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
      element.appendChild(node('div', 'num', side.avg_miss_m === null ? '–' : side.avg_miss_m.toFixed(2) + '\u00a0m'));
      let note = 'avg miss on this stretch';
      if (side.lateral_miss_m !== null) note += ' · lateral ' + side.lateral_miss_m.toFixed(2) + '\u00a0m';
      element.appendChild(node('div', 'who', note));
      return element;
    }
    if (video) {
      miss.appendChild(box('Reference (top)', video.reference));
      miss.appendChild(box('Student (bottom)', video.student));
      let sub = video.segment ? 'segment ' + video.segment : '';
      if (video.start !== null) sub += ' · from ' + video.start + ' s';
      if (video.duration !== null) sub += ' · ' + video.duration + ' s';
      if (video.speed !== null) sub += ' · ' + video.speed + '×';
      miss.appendChild(node('div', 'note', sub));
    } else if (!data.videos) {
      load('videos').then(() => { renderCompare(); playVideo(name); });
    }
    card.scrollIntoView({behavior: 'smooth', block: 'start'});
  }

  // -- new comparison video: a wizard, the models (step 1), the drive (step 2), the moment (step 3) --
  const cmpWizard = wizard('cmp', 3, (step) => { if (step === 2) renderDrives(); if (step === 3) renderMoment(); });
  function compareReference() {
    const chosen = document.querySelector('input[name="cmp-reference"]:checked');
    return chosen ? chosen.value : 'stock';
  }
  // The reference cards: what each one is, and how it did when it was evaluated.
  function renderReferences() {
    renderChoices(byId('cmp-step-1'));
    for (const spec of Object.keys(REFERENCE_INFO)) {
      const result = evalResultOf(spec);
      byId('cmp-ref-' + spec + '-sub').textContent = result && result.avg_miss_m !== null
        ? result.avg_miss_m.toFixed(2) + ' m average miss' : REFERENCE_INFO[spec][1];
    }
  }
  function clock(seconds) {
    const whole = Math.floor(seconds);
    return Math.floor(whole / 60) + ':' + String(whole % 60).padStart(2, '0') + (seconds % 1 ? '.5' : '');
  }
  // The stretch fits in the minute of the segment: the start slider ends where the length allows.
  function renderMoment() {
    const start = byId('cmp-start');
    const length = Number(byId('cmp-duration').value);
    start.max = String(Math.max(Number(start.min), 60 - length));
    if (Number(start.value) > Number(start.max)) start.value = start.max;
    const from = Number(start.value);
    byId('cmp-moment').textContent = clock(from) + ' – ' + clock(Math.min(60, from + length));
    start.setAttribute('aria-valuetext', 'from ' + clock(from));
  }
  for (const radio of document.querySelectorAll('input[name="cmp-reference"]')) {
    radio.addEventListener('change', () => renderChoices(byId('cmp-step-1')));
  }
  byId('cmp-start').addEventListener('input', renderMoment);
  presetGroup('cmp-durations', 'cmp-duration', renderMoment);
  presetGroup('cmp-speeds', 'cmp-speed');
  renderChoices(byId('cmp-step-1'));

  byId('btn-compare').addEventListener('click', () => {
    const params = {segment: byId('cmp-segment').value, reference: compareReference(),
      student: byId('cmp-student').value, start: numberOrText(byId('cmp-start').value),
      duration: numberOrText(byId('cmp-duration').value), speed: Number(byId('cmp-speed').value),
      inset: byId('cmp-inset').checked, show_gt: byId('cmp-gt').checked};
    runAction('compare', params);
  });
  for (const id of ['cmp-segment', 'cmp-student']) byId(id).addEventListener('change', updateButtons);

  // -- Publish --------------------------------------------------------------------------------
  // A finished model (the picker), then where it goes: "On your comma" (install, put the original
  // back) or "On Hugging Face" (dry run, publish). The destination is in the hash: #publish/comma, #publish/hf.
  function slugOf(name) { return name.toLowerCase().replace(/[^a-z0-9]/g, '_'); }
  // The run chosen in the picker ('' when there is none).
  function publishRun() {
    const value = byId('publish-run').value;
    return value.startsWith('run:') ? value.slice(4) : '';
  }
  function publishParams() {
    return {run: publishRun(), name: byId('publish-name').value, stable: byId('publish-stable').checked};
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
  // What stops publishing to Hugging Face ('' = nothing, or a server without accounts).
  function hfProblem() {
    const account = accountOf('hf');
    return account ? account.problem || '' : '';
  }

  // -- a wizard: the model (step 1), where (step 2), install or publish there (step 3) --
  // At step 3 the hash says where: #publish/comma, #publish/hf.
  const publishWizard = wizard('publish', 3, () => {
    renderDestinations();
    const hash = '#publish' + (publishWizard.step === 3 ? '/' + publishDest : '');
    if (currentTab === 'publish' && location.hash !== hash) history.replaceState(null, '', hash);
  });
  // Putting the original model back needs no model: straight to the comma.
  byId('publish-restore-shortcut').addEventListener('click', () => {
    publishDest = 'comma';
    publishWizard.show(3);
  });

  // -- the two destinations --
  function chooseDestination(dest, focus) {
    publishDest = dest;
    renderDestinations();
    updateButtons();
    if (focus) byId('dest-' + dest).focus();
  }
  for (const dest of DESTINATIONS) byId('dest-' + dest).addEventListener('click', () => chooseDestination(dest));
  byId('dest-options').addEventListener('keydown', (event) => {
    let step = 0;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') step = 1;
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') step = -1;
    if (!step) return;
    event.preventDefault();
    const index = (DESTINATIONS.indexOf(publishDest) + step + DESTINATIONS.length) % DESTINATIONS.length;
    chooseDestination(DESTINATIONS[index], true);
  });

  // [text, kind] under each destination: what it is doing, what stops it, or that it is ready.
  function commaSummary() {
    const job = status ? liveJob(['install', 'restore']) : null;
    if (job && job.state === 'running') return [job.action === 'install' ? 'Installing…' : 'Putting the original back…', 'run'];
    const comma = status ? status.comma : null;
    if (!comma || comma.reachable === null) return ['Not checked yet', 'muted'];
    if (!comma.reachable) return ['Not reachable', 'bad'];
    if (comma.onroad) return ['Driving: park first', 'todo'];
    if (comma.bundle_default === false) return ['Models menu not on Default', 'todo'];
    return ['Ready, parked', 'ok'];
  }
  function hfSummary() {
    const job = status ? liveJob(['publish_dry', 'publish']) : null;
    if (job && job.state === 'running') return [job.action === 'publish' ? 'Publishing…' : 'Dry run…', 'run'];
    const account = accountOf('hf');
    if (!account) return [(status && status.hf_repo) || 'Your repository', 'muted'];
    if (!account.connected) {
      return [account.problem && account.problem.indexOf('no longer works') >= 0 ? 'Token no longer works' : 'Not connected', 'bad'];
    }
    if (account.problem) return ['Token cannot write', 'todo'];
    return [account.user ? 'As ' + account.user : 'Connected', 'ok'];
  }

  function renderDestinations() {
    const summaries = {comma: commaSummary(), hf: hfSummary()};
    for (const dest of DESTINATIONS) {
      const chosen = dest === publishDest;
      const option = byId('dest-' + dest);
      option.setAttribute('aria-checked', String(chosen));
      option.tabIndex = chosen ? 0 : -1;
      const [text, kind] = summaries[dest];
      option.classList.toggle('blocked', kind === 'bad');
      if (!same('dest-sub:' + dest, summaries[dest])) {
        byId('dest-' + dest + '-sub').replaceChildren(node('span', 'dest-dot ' + kind), document.createTextNode(text));
      }
      byId('dest-panel-' + dest).hidden = !chosen;
    }
  }

  // -- On your comma: a checklist of what installing and restoring need --
  const CHECK_MARK = {ok: '✓', todo: '!', bad: '✕', muted: '–'};
  function capital(text) { return text.charAt(0).toUpperCase() + text.slice(1); }

  // [kind, title, detail] rows from the comma state of the status.
  function commaChecks(comma) {
    if (comma.reachable === null) return [['muted', 'Not checked yet', 'Tap Check now to ask the comma.']];
    if (!comma.reachable) {
      return [['bad', 'Not reachable', (comma.error ? capital(comma.error) + '. ' : '') + 'Its address (' + comma.host
        + ') is set in Settings.']];
    }
    const checks = [['ok', 'Reachable', 'Address: ' + comma.host]];
    if (comma.onroad) checks.push(['todo', 'Driving', 'Installing and restoring wait until the car is off (offroad).']);
    else checks.push(['ok', 'Parked', 'The car is off (offroad).']);
    if (comma.bundle_default === false) {
      checks.push(['todo', 'Models menu: ' + comma.bundle, 'The comma drives with the model chosen there, so an '
        + 'installed model would not be used. To install, set it back to Default on the comma screen (Settings › Models).']);
    } else if (comma.bundle_default === true) {
      checks.push(['ok', 'Models menu on Default', 'The comma drives with openpilot\'s model, or the one you installed.']);
    }
    if (comma.backup === true) checks.push(['ok', 'Original model saved', 'It can be put back at any time.']);
    else if (comma.backup === false) checks.push(['muted', 'Original model not saved yet', 'Installing a model saves it first.']);
    return checks;
  }

  function renderCommaChecklist() {
    const comma = status ? status.comma : null;
    const checks = comma ? commaChecks(comma) : [['muted', 'Reading the state of the comma…', '']];
    if (!same('comma-checklist', checks)) {
      const list = byId('comma-checklist');
      list.replaceChildren();
      for (const [kind, title, detail] of checks) {
        const item = node('li');
        item.appendChild(node('span', 'check-mark ' + kind, CHECK_MARK[kind]));
        const text = node('div', 'check-text');
        text.appendChild(node('div', 'check-title', title));
        if (detail) text.appendChild(node('div', 'check-detail', detail));
        item.appendChild(text);
        list.appendChild(item);
      }
    }
    byId('comma-checked').textContent = comma && comma.checked
      ? 'Checked at ' + fmtClock(comma.checked) + ' · again every 30 s while this page is open.' : '';
  }

  byId('btn-comma-check').addEventListener('click', async () => {
    const button = byId('btn-comma-check');
    button.disabled = true;
    button.textContent = 'Checking…';
    try {
      const answer = await post('/api/comma/check', {});
      if (status) {
        status.comma = answer.comma;
        renderComma('comma-kv-stato');
      }
      renderPublish();
      updateButtons();
    } catch (error) {
      toast(error.message, true);
    }
    button.disabled = false;
    button.textContent = 'Check now';
    // The install and restore buttons follow the new state of the comma.
    refreshStatus();
  });
  byId('btn-install').addEventListener('click', () => {
    const run = publishRun();
    if (!run) return;
    runAction('install', {run: run, confirm: 'INSTALL'}, 'Install the model ' + run + ' on the comma? It takes the '
      + 'place of openpilot\'s driving model and the comma restarts.');
  });
  byId('btn-restore').addEventListener('click', () => {
    runAction('restore', {confirm: 'RESTORE'}, 'Put openpilot\'s original driving model back on the comma? The comma restarts.');
  });

  // -- On Hugging Face --
  // Publishing needs the Hugging Face account of Settings: say whose it is, or point there.
  function renderPublishAccount() {
    const account = accountOf('hf');
    const callout = byId('publish-hf-callout');
    const note = byId('publish-hf-account');
    const problem = hfProblem();
    byId('publish-name').disabled = Boolean(problem);
    byId('publish-stable').disabled = Boolean(problem);
    if (!account) {
      callout.hidden = true;
      note.hidden = true;
      return;
    }
    const repo = (status && status.hf_repo) || '';
    callout.hidden = !problem;
    if (problem) byId('publish-hf-text').textContent = capital(problem) + '.';
    note.hidden = Boolean(problem);
    note.textContent = account.user ? 'Publishes to ' + repo + ' as ' + account.user + '.'
      : 'Publishes to ' + repo + ' with the Hugging Face account of Settings.';
  }

  function renderHfCatalog() {
    const hf = data.hf;
    if (!hf || same('hf', hf)) return;
    const kv = byId('hf-kv');
    kv.replaceChildren();
    kv.appendChild(node('dt', '', 'Repository'));
    const repoCell = node('dd');
    const link = node('a', '', hf.repo);
    link.href = hf.url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    repoCell.appendChild(link);
    kv.appendChild(repoCell);
    kv.appendChild(node('dt', '', 'Stable weights'));
    kv.appendChild(node('dd', '', hf.ok ? (hf.stable ? 'yes' : 'no') : '?'));
    kv.appendChild(node('dt', '', 'Checked'));
    kv.appendChild(node('dd', '', fmtTime(hf.checked) + (hf.ok ? '' : ' · ' + hf.error)));
    const list = byId('hf-models');
    list.replaceChildren();
    byId('hf-empty').hidden = !hf.ok || hf.models.length > 0;
    for (const name of hf.models) {
      const item = node('li', 'item');
      item.appendChild(node('span', 'what mono', name));
      if (hf.releases.indexOf(name) >= 0) item.appendChild(pill('PyTorch weights', 'ok'));
      list.appendChild(item);
    }
  }

  // The name follows the chosen model until the user writes another one.
  let lastPublishRun = '';
  function followRunName() {
    const run = publishRun();
    const nameField = byId('publish-name');
    if (run !== lastPublishRun) {
      if (!nameField.value || nameField.value === lastPublishRun) nameField.value = run;
      lastPublishRun = run;
    }
  }

  function renderPublish() {
    const finished = runsWith((run) => run.has_pt).map((run) => ({value: 'run:' + run.name, label: run.name,
      group: 'Finished trainings'}));
    fillSelect(byId('publish-run'), finished);
    renderPicker('publish-run');
    followRunName();
    renderDestinations();
    renderCommaChecklist();
    renderPublishAccount();
    renderHfCatalog();
  }

  function updatePublishButtons() {
    const params = publishParams();
    const problem = hfProblem();
    const validName = /^[A-Za-z0-9_.-]{1,64}$/.test(params.name);
    const taken = nameOnHf(params.name);
    let nameNote = '';
    if (params.name && !validName) nameNote = 'Invalid name: letters, digits, _ . - (64 at most).';
    else if (taken) nameNote = 'Hugging Face already has ' + slugOf(params.name) + ': the real publish will be refused.';
    else if (params.name) nameNote = 'On Hugging Face it becomes models/' + slugOf(params.name) + '/.';
    byId('publish-name-note').textContent = problem ? '' : nameNote;

    let dryReason = '';
    if (problem) dryReason = problem;
    else if (!params.run) dryReason = 'choose a finished model in step 1';
    else if (!validName) dryReason = 'type a valid name';
    setAction('btn-publish-dry', 'publish_dry', dryReason);

    const dry = lastDryRun();
    const dryStatus = byId('dry-status');
    if (!same('dry-status', [dry ? [dry.id, dry.state, dry.state_label] : null, Boolean(params.run && validName)])) {
      dryStatus.replaceChildren();
      if (dry) {
        dryStatus.appendChild(document.createTextNode('Dry run for these settings: '));
        dryStatus.appendChild(pill(dry.state_label, JOB_PILL[dry.state] || 'muted'));
        dryStatus.appendChild(document.createTextNode(' ' + fmtTime(dry.started) + ' '));
        const view = node('button', 'btn small', 'Details');
        view.type = 'button';
        view.addEventListener('click', () => openLog(dry.id, dry.label));
        dryStatus.appendChild(view);
      } else if (params.run && validName) {
        dryStatus.textContent = 'No dry run yet for this model, name and “stable” setting.';
      }
    }

    let reason = '';
    if (!params.run) reason = 'choose a finished model in step 1';
    else if (!validName) reason = 'type a valid name';
    else if (taken) reason = 'name already used on Hugging Face';
    else if (!dry || dry.state !== 'ok') reason = 'first a successful dry run with the same settings';
    setAction('btn-publish', 'publish', reason);
  }

  byId('btn-publish-dry').addEventListener('click', () => runAction('publish_dry', publishParams()));
  byId('btn-publish').addEventListener('click', () => {
    const params = publishParams();
    // The confirmation the server asks for is the popup's OK: it sends the name back.
    params.confirm = params.name;
    runAction('publish', params, 'Publish the model ' + params.run + ' to Hugging Face as "' + params.name + '"'
      + (params.stable ? ' and make it the stable weights' : '') + '? Anyone can download it once published.');
  });
  byId('publish-run').addEventListener('change', () => {
    followRunName();
    renderPicker('publish-run');
    updateButtons();
  });
  for (const id of ['publish-name', 'publish-stable']) {
    byId(id).addEventListener('input', updateButtons);
    byId(id).addEventListener('change', updateButtons);
  }

  // -- Settings ---------------------------------------------------------------------------------
  // The account as the status reports it ({connected, user, problem}), null from an older server.
  function accountOf(kind) {
    return status && status.accounts ? status.accounts[kind] || null : null;
  }

  const ACCOUNTS = {
    hf: {name: 'Hugging Face', secret: 'token', field: 'token', path: '/api/settings/hf',
      dropped: 'Publishing to Hugging Face stops working until you connect again.'},
    wandb: {name: 'Weights & Biases', secret: 'API key', field: 'key', path: '/api/settings/wandb',
      dropped: 'New trainings log on this server only until you connect again.'},
  };
  // The form of an account that is connected, opened by "Use another token".
  const replacing = {hf: false, wandb: false};

  function hfAccess(account, repo) {
    if (account.access === 'write') return account.fine_grained ? 'can write ' + repo : 'write access';
    if (account.access === 'read') return account.fine_grained ? 'cannot write ' + repo : 'read only';
    return 'access not known';
  }

  // [line under the name, pill text, pill kind] of an account.
  function accountTexts(kind, account, repo) {
    const meta = ACCOUNTS[kind];
    let texts = null;
    if (account.state === 'none') {
      texts = ['Not connected', 'Off', 'muted'];
    } else if (account.state === 'connected') {
      let line = 'Connected as ' + (account.user || '?');
      let label = ['Connected', 'ok'];
      if (kind === 'hf') {
        line += ' \u00b7 ' + hfAccess(account, repo);
        if (account.access === 'read') label = ['Read only', 'todo'];
      } else if (account.entity && account.entity !== account.user) {
        line += ' \u00b7 team ' + account.entity;
      }
      texts = [line, label[0], label[1]];
    } else if (account.state === 'invalid') {
      texts = ['The saved ' + meta.secret + ' no longer works: connect again with a new one.', 'Not working', 'bad'];
    } else if (account.state === 'unreachable') {
      texts = ['A ' + meta.secret + ' is saved, but it cannot be checked now (' + account.error + ').', 'Not checked', 'todo'];
    } else {
      texts = ['A ' + meta.secret + ' is saved, not checked yet.', 'Saved', 'muted'];
    }
    if (account.source === 'environment') texts[0] += ' \u00b7 set by the server configuration, change it there';
    return texts;
  }

  function renderAccount(kind, account, repo) {
    const texts = accountTexts(kind, account, repo);
    byId(kind + '-account-sub').textContent = texts[0];
    if (!same('account-pill:' + kind, texts.slice(1))) byId(kind + '-account-pill').replaceChildren(pill(texts[1], texts[2]));
    const editable = account.can_change !== false;
    const saved = account.state !== 'none';
    const formOpen = editable && (!saved || account.state === 'invalid' || replacing[kind]);
    byId(kind + '-form').hidden = !formOpen;
    byId('btn-' + kind + '-cancel').hidden = !(formOpen && saved && replacing[kind]);
    byId(kind + '-manage').hidden = !(editable && saved && !formOpen);
    if (!formOpen) byId(kind + '-reason').textContent = '';
  }

  function transportNote() {
    const note = byId('transport-note');
    const host = location.hostname;
    let text = '';
    let warn = false;
    if (location.protocol === 'https:') {
      text = 'This page uses HTTPS: passwords and tokens travel encrypted from your browser to the server.';
    } else if (host === 'localhost' || host === '127.0.0.1' || host === '[::1]' || host === '::1') {
      text = 'This page talks to the server on this computer (or through an ssh tunnel): passwords and tokens do not '
        + 'cross the network in clear.';
    } else {
      text = 'Honest note: this site talks plain HTTP on the local network, so a password or token you type here travels '
        + 'unencrypted on your LAN. Fine at home; put the site behind HTTPS before exposing it anywhere else.';
      warn = true;
    }
    note.textContent = text;
    note.classList.toggle('warn', warn);
  }

  // Site settings: the field shows the saved value (or the one in use), until the user changes it.
  const SITE_FIELDS = {
    hf_repo: {input: 'site-hf-repo', pattern: /^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/,
      label: 'Hugging Face repository', about: 'Where Publish uploads your models: owner/name on huggingface.co.',
      invalid: 'The Hugging Face repository is owner/name: letters, digits, "-", "_" and ".".'},
    comma_host: {input: 'site-comma-host', pattern: /^[A-Za-z0-9][A-Za-z0-9_.@-]*$/, label: 'Comma address',
      about: 'How this server reaches your comma over ssh: a host name, an IP address or user@host.',
      invalid: 'The comma address is a host name, an IP address or user@host: letters, digits, ".", "-", "_" and "@".'},
  };

  function renderSite(site) {
    for (const [name, field] of Object.entries(SITE_FIELDS)) {
      const info = site[name];
      if (!info) continue;
      const input = byId(field.input);
      const shown = info.from_environment ? (info.saved || info.default) : info.value;
      // Not while the user is changing it: only a field still showing the last server value follows it.
      if (input.dataset.server === undefined || input.value === input.dataset.server) input.value = shown;
      input.dataset.server = shown;
      input.placeholder = info.default;
      const note = byId(field.input + '-note');
      let text = field.about;
      if (info.from_environment) {
        text = 'In use: ' + info.value + '. The server configuration sets it, and it wins over the value saved here.';
      } else if (info.value === info.default) {
        text += ' Empty = the default.';
      }
      note.textContent = text;
      note.classList.toggle('warn', Boolean(info.from_environment));
    }
    updateSiteButton();
  }

  // {name: value} of the fields the user changed, or a reason why they cannot be saved.
  function siteChanges() {
    const changes = {};
    for (const [name, field] of Object.entries(SITE_FIELDS)) {
      const input = byId(field.input);
      if (input.dataset.server === undefined) continue;
      const value = input.value.trim();
      if (value === input.dataset.server) continue;
      if (value && !field.pattern.test(value)) return {reason: field.invalid};
      changes[name] = value;
    }
    if (!Object.keys(changes).length) return {reason: 'nothing changed'};
    return {changes: changes};
  }

  // The server's answer to the last save that failed, shown until the fields change.
  let siteError = '';
  function updateSiteButton() {
    const result = siteChanges();
    byId('btn-site-save').disabled = Boolean(result.reason);
    byId('site-reason').textContent = result.reason && result.reason !== 'nothing changed' ? result.reason : siteError;
  }

  function renderSettings() {
    transportNote();
    const settings = data.settings;
    if (!settings) return;
    const repo = settings.site && settings.site.hf_repo ? settings.site.hf_repo.value : '';
    renderAccount('hf', settings.accounts.hf, repo);
    renderAccount('wandb', settings.accounts.wandb, repo);
    if (settings.site) renderSite(settings.site);
    renderAndroid(settings.android);
  }

  // The Android app built on this server (helpers/build_android.sh), for download from the phone.
  function renderAndroid(build) {
    const link = byId('android-download');
    link.hidden = !build;
    // The .aab of helpers/build_android.sh, for an upload by hand in Play Console.
    byId('android-bundle').hidden = !(build && build.bundle);
    byId('android-note').hidden = !build;
    if (!build) {
      byId('android-sub').textContent = 'No app built on this server yet.';
      return;
    }
    const parts = ['Version ' + (build.version || '?')];
    if (build.built) parts.push('built ' + fmtDay(build.built));
    parts.push(fmtBytes(build.size));
    if (build.signed_with === 'debug') parts.push('debug key: uninstall the previous version first');
    byId('android-sub').textContent = parts.join(' \u00b7 ');
    link.textContent = 'Download the app';
  }

  async function connectAccount(kind) {
    const meta = ACCOUNTS[kind];
    const input = byId(kind + '-secret');
    const reason = byId(kind + '-reason');
    const button = byId('btn-' + kind + '-connect');
    const value = input.value.trim();
    if (!value) {
      reason.textContent = 'Paste the ' + meta.secret + ' first.';
      input.focus();
      return;
    }
    const payload = {};
    payload[meta.field] = value;
    button.disabled = true;
    button.textContent = 'Checking\u2026';
    reason.textContent = '';
    try {
      const answer = await post(meta.path, payload);
      input.value = '';
      replacing[kind] = false;
      if (data.settings) data.settings.accounts[kind] = answer.account;
      toast(meta.name + ': connected as ' + (answer.account.user || '?'));
      renderSettings();
      refreshStatus();
    } catch (error) {
      reason.textContent = error.message;
      toast(error.message, true);
    }
    button.disabled = false;
    button.textContent = 'Connect';
  }

  async function disconnectAccount(kind) {
    const meta = ACCOUNTS[kind];
    if (!window.confirm('Disconnect ' + meta.name + '? The ' + meta.secret + ' is deleted from this server. '
      + meta.dropped)) return;
    try {
      const answer = await post(meta.path + '/logout', {});
      if (data.settings) data.settings.accounts[kind] = answer.account;
      toast(meta.name + ' disconnected');
      renderSettings();
      refreshStatus();
    } catch (error) {
      toast(error.message, true);
    }
  }

  for (const kind of Object.keys(ACCOUNTS)) {
    byId('btn-' + kind + '-connect').addEventListener('click', () => connectAccount(kind));
    byId(kind + '-secret').addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        connectAccount(kind);
      }
    });
    byId('btn-' + kind + '-replace').addEventListener('click', () => {
      replacing[kind] = true;
      renderSettings();
      byId(kind + '-secret').focus();
    });
    byId('btn-' + kind + '-cancel').addEventListener('click', () => {
      replacing[kind] = false;
      byId(kind + '-secret').value = '';
      renderSettings();
    });
    byId('btn-' + kind + '-disconnect').addEventListener('click', () => disconnectAccount(kind));
  }

  for (const field of Object.values(SITE_FIELDS)) {
    byId(field.input).addEventListener('input', () => {
      siteError = '';
      updateSiteButton();
    });
  }
  byId('btn-site-save').addEventListener('click', async () => {
    const result = siteChanges();
    if (result.reason) return;
    const button = byId('btn-site-save');
    button.disabled = true;
    try {
      const answer = await post('/api/settings/site', result.changes);
      siteError = '';
      for (const field of Object.values(SITE_FIELDS)) delete byId(field.input).dataset.server;
      if (data.settings) data.settings.site = answer.site;
      const overridden = Object.keys(result.changes).filter((name) => answer.site[name] && answer.site[name].from_environment);
      if (overridden.length) {
        toast('Saved, but the server configuration sets ' + overridden.map((name) => SITE_FIELDS[name].label).join(' and ')
          + ', and it wins.', true);
      } else {
        toast(answer.applied ? 'Saved: in use now.' : 'Saved: restart the site to use it.');
      }
      renderSettings();
      refreshStatus();
    } catch (error) {
      siteError = error.message;
      updateSiteButton();
      toast(error.message, true);
    }
  });

  // -- administrator -----------------------------------------------------------------------------
  function renderAdmin() {
    const state = data.auth;
    if (!state) return;
    let text = 'Signed in as ' + state.user + '.';
    if (state.sessions > 1) text += ' Signed in on ' + state.sessions + ' browsers.';
    byId('admin-sub').textContent = text;
  }

  async function signOut(everywhere) {
    if (everywhere && !confirm('Sign out on every browser, this one included?')) return;
    try {
      await post(everywhere ? '/api/auth/logout-all' : '/api/auth/logout', {});
    } catch (error) {
      // Already signed out: the sign-in page follows anyway.
    }
    location.replace('/');
  }
  byId('btn-signout').addEventListener('click', () => signOut(false));
  byId('btn-topbar-signout').addEventListener('click', () => signOut(false));
  byId('btn-signout-all').addEventListener('click', () => signOut(true));

  byId('admin-password-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const reason = byId('admin-reason');
    const current = byId('admin-current').value;
    const fresh = byId('admin-new').value;
    reason.textContent = '';
    if (!current) { reason.textContent = 'Write the current password.'; return; }
    if (fresh.length < 12) { reason.textContent = 'The new password needs at least 12 characters.'; return; }
    if (fresh !== byId('admin-repeat').value) { reason.textContent = 'The two new passwords are different.'; return; }
    const button = byId('btn-admin-password');
    button.disabled = true;
    try {
      const answer = await post('/api/auth/password', {current: current, password: fresh});
      for (const id of ['admin-current', 'admin-new', 'admin-repeat']) byId(id).value = '';
      data.auth = answer.state;
      renderAdmin();
      toast('Password changed: every other browser is signed out.');
    } catch (error) {
      reason.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });

  // -- appearance --------------------------------------------------------------------------------
  // Dark (the default) removes <html data-theme>; 'light' sets it, 'auto' sets it and the CSS follows the system.
  // Kept in this browser only; the <head> of the page applies it before the first paint.
  const THEME_KEY = 'distill-theme';
  function storedTheme() {
    try {
      const value = localStorage.getItem(THEME_KEY);
      return value === 'light' || value === 'auto' ? value : 'dark';
    } catch (error) {
      return 'dark';
    }
  }
  function applyTheme(choice) {
    if (choice === 'light' || choice === 'auto') document.documentElement.dataset.theme = choice;
    else delete document.documentElement.dataset.theme;
    try {
      if (choice === 'light' || choice === 'auto') localStorage.setItem(THEME_KEY, choice);
      else localStorage.removeItem(THEME_KEY);
    } catch (error) {
      // Storage blocked (private window): the choice lasts until the page is closed.
    }
    renderThemeChoice(choice);
  }
  function renderThemeChoice(choice) {
    for (const button of document.querySelectorAll('[data-theme-choice]')) {
      const on = button.dataset.themeChoice === choice;
      button.setAttribute('aria-checked', on ? 'true' : 'false');
      button.tabIndex = on ? 0 : -1;
    }
  }
  const themeButtons = Array.from(document.querySelectorAll('[data-theme-choice]'));
  for (const button of themeButtons) {
    button.addEventListener('click', () => applyTheme(button.dataset.themeChoice));
    button.addEventListener('keydown', (event) => {
      const step = {ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1}[event.key];
      if (!step) return;
      event.preventDefault();
      const next = themeButtons[(themeButtons.indexOf(button) + step + themeButtons.length) % themeButtons.length];
      applyTheme(next.dataset.themeChoice);
      next.focus();
    });
  }
  renderThemeChoice(storedTheme());

  // "Connect in Settings" buttons of the other tabs.
  for (const button of document.querySelectorAll('[data-goto]')) {
    button.addEventListener('click', () => {
      showTab(button.dataset.goto);
      const target = button.dataset.account ? byId('account-' + button.dataset.account) : null;
      if (target && target.scrollIntoView) target.scrollIntoView({block: 'start'});
      else window.scrollTo(0, 0);
    });
  }

  // -- Jobs -------------------------------------------------------------------------------------
  byId('btn-update').addEventListener('click', () => runAction('update', {}, 'Update the code: pull the latest version, update the libraries and restart the site (a few seconds). Start?'));
  byId('btn-selftest').addEventListener('click', () => runAction('selftest', {}));
  byId('reload-page').addEventListener('click', () => location.reload());

  // -- job panel -------------------------------------------------------------------------------
  // A job's output, made readable: progress lines become the progress bar, machine lines
  // (RUN_DIR=, OUTPUT_*=) and blank lines go away, paths are shown relative to the repo, a line
  // rewritten with \r keeps its last state and consecutive training steps collapse into the last one.
  const MACHINE_LINE = /^(RUN_DIR|OUTPUT_MP4|OUTPUT_JSON)=/;
  const PROGRESS_LINE = /^PROGRESS (\d+)\/(\d+)$/;
  const STEP_LINE = /^step (\d+)\/(\d+):/;
  // rsync's progress lines: the speed is shown in the progress, not as messages.
  const TRANSFER_LINE = /^[\d,]+\s+\d+%\s+[\d.]+[kMG]B\/s/;
  const MAX_MESSAGES = 400;

  function readLog(text) {
    const messages = [];
    let progress = null;
    for (const raw of text.split('\n')) {
      const parts = raw.split('\r');
      let line = parts[parts.length - 1].trim() || parts.map((part) => part.trim()).filter(Boolean).pop() || '';
      if (!line || MACHINE_LINE.test(line) || TRANSFER_LINE.test(line)) continue;
      const match = PROGRESS_LINE.exec(line) || STEP_LINE.exec(line);
      if (match) progress = {done: Number(match[1]), total: Number(match[2])};
      if (PROGRESS_LINE.test(line)) continue;
      line = line.replace(/\/[^\s]*\/openpilot\.distill\//g, '');
      // "== ..." lines are the site's own header and footer: the sheet shows that as its status.
      if (line.startsWith('== ')) continue;
      const kind = /^(ERROR|Traceback)/.test(line) ? 'error' : '';
      const last = messages[messages.length - 1];
      if (STEP_LINE.test(line) && last && STEP_LINE.test(last.text)) messages.pop();
      messages.push({text: line, kind: kind});
    }
    return {messages: messages.slice(-MAX_MESSAGES), progress: progress};
  }

  function renderMessages(list, messages, empty) {
    list.replaceChildren();
    if (!messages.length) list.appendChild(node('li', 'muted', empty));
    for (const message of messages) list.appendChild(node('li', message.kind, message.text));
  }

  function openLog(id, label) {
    clearTimeout(log.timer);
    log = {id: id, offset: 0, timer: null, text: '', view: null};
    byId('term-title').textContent = label;
    byId('term-sub').replaceChildren();
    byId('term-text').replaceChildren(node('li', 'muted', 'Loading…'));
    byId('term-progress').hidden = true;
    byId('term-actions').replaceChildren();
    byId('term-actions').dataset.job = '';
    byId('terminal').classList.add('open');
    byId('sheet-backdrop').classList.add('open');
    document.body.classList.add('sheet-open');
    pollLog();
  }

  function sheetJob() {
    return status && log.id ? status.jobs.find((item) => item.id === log.id) || null : null;
  }

  // Status line, progress and buttons of the open sheet, from the job as the status reports it.
  function renderSheetStatus() {
    const job = sheetJob();
    const sub = byId('term-sub');
    sub.replaceChildren();
    if (!job) return;
    const now = Date.now() / 1000;
    if (job.state === 'running') {
      sub.appendChild(node('span', 'run', 'Running'));
      sub.appendChild(document.createTextNode(' · started ' + fmtTime(job.started) + ' · ' + fmtDuration(now - job.started)));
    } else {
      const took = job.finished ? fmtDuration(job.finished - job.started) : '–';
      const text = job.state === 'ok' ? 'Done' : (job.state === 'stopped' ? 'Stopped' : 'Failed');
      sub.appendChild(node('span', job.state === 'ok' ? 'ok' : 'bad', text));
      sub.appendChild(document.createTextNode(' · ' + fmtTime(job.started) + ' · took ' + took));
    }

    const progress = job.progress || (log.view && log.view.progress);
    byId('term-progress').hidden = !progress || !progress.total;
    if (progress && progress.total) {
      let percent = progress.done * 100 / progress.total;
      if (job.state === 'ok') percent = 100;
      byId('term-percent').textContent = Math.floor(percent) + '%';
      byId('term-bar').style.width = Math.max(0, Math.min(100, percent)) + '%';
      let eta = progress.done.toLocaleString('en-GB') + ' of ' + progress.total.toLocaleString('en-GB');
      if (job.state === 'running' && progress.rate_mb_s) eta += ' · ' + progress.rate_mb_s.toFixed(1) + ' MB/s';
      if (job.state === 'running' && progress.done >= progress.total) eta += ' · finishing';
      else if (job.state === 'running' && progress.done > 0) {
        eta += ' · about ' + fmtDuration(Math.max(1, (now - job.started) * (progress.total - progress.done) / progress.done))
          + ' left';
      }
      byId('term-progress-text').textContent = eta;
    }

    const actions = byId('term-actions');
    const key = job.id + ':' + job.state + ':' + job.stoppable;
    if (actions.dataset.job !== key) {
      actions.dataset.job = key;
      actions.replaceChildren();
      if (job.state === 'running' && job.stoppable) {
        const stop = node('button', 'btn danger', 'Stop');
        stop.type = 'button';
        stop.addEventListener('click', () => stopJob(job));
        actions.appendChild(stop);
      }
    }

    // The newest line is the step in progress while the job runs; the end gets its own line.
    const items = byId('term-text').children;
    if (items.length) {
      const last = items[items.length - 1];
      last.classList.toggle('current', job.state === 'running' && !last.classList.contains('muted'));
    }
  }

  function renderTimeline() {
    const list = byId('term-text');
    const body = byId('term-body');
    const atEnd = body.scrollTop + body.clientHeight >= body.scrollHeight - 30;
    const job = sheetJob();
    const messages = log.view.messages.slice();
    if (job && job.state === 'ok') messages.push({text: 'Finished', kind: 'done'});
    else if (job && job.state === 'failed') messages.push({text: 'Failed (exit code ' + job.exit_code + ')', kind: 'error'});
    else if (job && job.state === 'stopped') messages.push({text: 'Stopped', kind: 'error'});
    renderMessages(list, messages, 'Starting…');
    if (atEnd) body.scrollTop = body.scrollHeight;
  }

  async function pollLog() {
    const id = log.id;
    if (!id) return;
    try {
      const chunk = await api('/api/jobs/' + id + '/log?offset=' + log.offset);
      if (log.id !== id) return;
      log.offset = chunk.offset;
      log.text += chunk.text;
      const previous = log.view ? JSON.stringify(log.view) : '';
      log.view = readLog(log.text);
      const job = sheetJob();
      const signature = JSON.stringify(log.view) + (job ? job.state : '');
      if (signature !== previous + (log.state || '')) renderTimeline();
      log.state = job ? job.state : '';
      renderSheetStatus();
    } catch (error) {
      toast(error.message, true);
      return;
    }
    const job = sheetJob();
    const running = !job || job.state === 'running';
    log.timer = setTimeout(pollLog, running ? 1500 : 5000);
  }

  function closeSheet() {
    clearTimeout(log.timer);
    log.id = null;
    byId('terminal').classList.remove('open');
    byId('info-sheet').classList.remove('open');
    byId('sheet-backdrop').classList.remove('open');
    document.body.classList.remove('sheet-open');
  }
  byId('term-close').addEventListener('click', closeSheet);
  byId('info-close').addEventListener('click', closeSheet);
  byId('sheet-backdrop').addEventListener('click', closeSheet);

  // -- explanations of the Preprocessing numbers, in a sheet ---------------------------------------------
  function openInfo(title, sub, build) {
    byId('info-title').textContent = title;
    byId('info-sub').textContent = sub;
    const body = byId('info-body');
    body.replaceChildren();
    build(body);
    body.scrollTop = 0;
    byId('info-sheet').classList.add('open');
    byId('sheet-backdrop').classList.add('open');
    document.body.classList.add('sheet-open');
  }

  function infoText(body, text) { body.appendChild(node('p', 'info-text', text)); }

  // The drives (outings) of a list of segment names, newest first: "Wed 21 Oct · 12:00 – 12:33".
  function infoDrives(body, names, title) {
    const drives = (data.drives ? data.drives.drives : []).filter((drive) => names.has(drive.segment));
    const routes = new Map();
    for (const drive of drives) {
      if (!routes.has(drive.route)) routes.set(drive.route, []);
      routes.get(drive.route).push(drive);
    }
    const rows = Array.from(routes.values()).map((list) => list.sort((a, b) => (a.start || 0) - (b.start || 0)));
    rows.sort((a, b) => (b[0].start || 0) - (a[0].start || 0));
    if (!rows.length) return;
    body.appendChild(node('h3', 'sheet-section', title));
    const list = node('ul', 'info-list');
    for (const route of rows) {
      const item = node('li');
      const first = route[0];
      item.appendChild(node('strong', '', first.start ? fmtDate(first.start) : 'Drive ' + first.route.slice(0, 8)));
      const times = route.filter((drive) => drive.start).map((drive) => fmtClock(drive.start));
      item.appendChild(node('span', '', (times.length ? times.join(', ') : route.length + ' segments')));
      list.appendChild(item);
    }
    body.appendChild(list);
  }

  function infoLocalizeButton(body) {
    const button = node('button', 'btn primary', 'Localize now');
    button.type = 'button';
    button.addEventListener('click', () => { closeSheet(); runAction('localize', localizeParams(), 'Start the localization now?'); });
    const reason = actionReason('localize');
    button.disabled = reason !== '';
    const box = node('div', 'info-actions');
    box.appendChild(button);
    if (reason) box.appendChild(node('div', 'reason', reason));
    body.appendChild(box);
  }

  function openDataInfo(key) {
    const segments = data.segments;
    if (!segments) return;
    const counts = segments.counts;
    const rows = segments.segments;
    const names = (filter) => new Set(rows.filter(filter).map((row) => row.name));
    if (key === 'total') {
      openInfo('Segments', counts.total + ' in your data', (body) => {
        infoText(body, 'A segment is one minute of a drive: the road videos and the car\'s logs, downloaded from '
          + 'the comma. Only dashcam drives of at least 5 minutes are taken, without the first and last two minutes.');
        infoDrives(body, names(() => true), 'Your drives');
      });
    } else if (key === 'localized') {
      openInfo('Localized', counts.localized + ' of ' + counts.total + ' segments', (body) => {
        infoText(body, 'The localizer works out where the car really was, frame by frame: position, speed and '
          + 'heading, from the GPS and the motion sensors. That path is what the model learns to predict, so only '
          + 'localized segments are used for training and evaluation.');
        body.appendChild(node('h3', 'sheet-section', 'GPS accuracy'));
        body.appendChild(gpsBar(counts.map || {}, 'gps-bar'));
        const legend = node('ul', 'gps-legend');
        for (const quality of GPS_ORDER) {
          if (!(counts.map || {})[quality]) continue;
          const item = node('li');
          item.appendChild(node('span', 'gps-dot gps-' + quality));
          item.appendChild(node('strong', '', String(counts.map[quality])));
          item.appendChild(node('span', '', GPS_LABEL[quality]));
          legend.appendChild(item);
        }
        body.appendChild(legend);
        if (counts.refinable) {
          infoText(body, counts.refinable + ' segments can probably get a more accurate map now.');
          infoLocalizeButton(body);
        }
      });
    } else if (key === 'failed') {
      openInfo('Failed', counts.failed ? counts.failed + ' segments could not be localized' : 'Nothing failed', (body) => {
        infoText(body, 'A segment fails when the localizer cannot work out the path, for example without a GPS '
          + 'fix. It is left out of training; the localization tries it again next time.');
        const failed = rows.filter((row) => row.state === 'failed');
        if (failed.length) {
          const list = node('ul', 'info-list');
          for (const row of failed) {
            const item = node('li');
            item.appendChild(node('strong', '', 'Minute ' + (row.number !== null ? row.number : '?') + ' of drive '
              + row.route.slice(0, 8)));
            item.appendChild(node('span', '', row.reason || 'reason not recorded'));
            list.appendChild(item);
          }
          body.appendChild(list);
        }
      });
    } else if (key === 'todo') {
      openInfo('To localize', counts.todo + ' segments waiting', (body) => {
        infoText(body, 'These segments are downloaded but not localized yet, so training cannot use them. The '
          + 'localization takes a few seconds per segment.');
        if (counts.todo || counts.refinable) infoLocalizeButton(body);
      });
    } else if (key === 'heldout') {
      openInfo('Hold-out', counts.heldout_localized + ' segments ready', (body) => {
        infoText(body, 'About one segment in ten is never shown to the model during training. Those drives are '
          + 'the fair test: Evaluation and the comparison videos use them to show how the model does on roads it '
          + 'has not learned by heart.');
        infoDrives(body, names((row) => row.heldout && row.state === 'localized'), 'Hold-out drives');
      });
    } else if (key === 'discarded') {
      openInfo('Discarded', counts.discarded + ' segments set aside', (body) => {
        infoText(body, 'Segments the localizer can never use are moved aside, so they do not stop the training. '
          + 'Why:');
        const list = node('ul', 'info-list');
        for (const reason of segments.discarded_reasons) {
          const item = node('li');
          item.appendChild(node('strong', '', String(reason.count)));
          item.appendChild(node('span', '', reason.reason));
          list.appendChild(item);
        }
        body.appendChild(list);
      });
    }
  }
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && log.id) closeSheet();
  });

  // -- start -----------------------------------------------------------------------------------
  showTab(location.hash ? location.hash.slice(1) : 'status');
  refreshStatus();
  setInterval(refreshStatus, STATUS_PERIOD);
  setInterval(() => { if (!document.hidden) loadTab(false); }, 5000);
})();

import { escapeHtml, formatTime } from './ui.js';

const HISTORY_KEY = 'echocount.history.v1';
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

export function initHistory(ui) {
  const { history, toast } = ui;
  const stats = $('#historyStats');
  const search = $('#historySearch');
  const list = $('#historyList');
  const plot = $('#historyPlot');
  const yAxis = $('#chartYAxis');
  const xAxis = $('#chartXAxis');
  const countLabel = $('#historyCount');
  const importInput = $('#historyImportFile');
  let currentRows = [];

  function rows() {
    return history.all().filter(row => row && typeof row.filename === 'string'
      && Number.isFinite(Number(row.ts)) && Number.isSafeInteger(Number(row.count))
      && Number.isFinite(Number(row.conf)) && Number.isFinite(Number(row.duration)));
  }

  function renderStats(items) {
    const total = items.length;
    const mean = total ? items.reduce((sum, row) => sum + Number(row.count), 0) / total : 0;
    const latest = total ? new Date(Math.max(...items.map(row => Number(row.ts)))).toLocaleDateString() : '—';
    stats.innerHTML = [
      ['Saved analyses', String(total)],
      ['Mean estimated count', total ? mean.toFixed(1) : '—'],
      ['Most recent', latest]
    ].map(([label, value]) => `<div class="history-stat"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`).join('');
  }

  function drawPlot(items) {
    plot.replaceChildren();
    if (!items.length) {
      yAxis?.replaceChildren(...['—', 'estimated sources', '—'].map(value => {
        const label = document.createElement('span'); label.textContent = value; return label;
      }));
      xAxis?.replaceChildren(...['—', 'analysis date', '—'].map(value => {
        const label = document.createElement('span'); label.textContent = value; return label;
      }));
    }
    if (!items.length) {
      const empty = document.createElement('p');
      empty.className = 'plot-empty';
      empty.textContent = currentRows.length ? 'No saved results match this search.' : 'No saved analyses yet.';
      plot.append(empty);
      return;
    }
    const times = items.map(row => Number(row.ts));
    const counts = items.map(row => Number(row.count));
    const minTime = Math.min(...times), maxTime = Math.max(...times);
    const minCount = Math.min(...counts), maxCount = Math.max(...counts);
    const dateLabel = time => new Date(time).toLocaleDateString([], { month: 'short', day: 'numeric' });
    yAxis?.replaceChildren(...[
      String(maxCount), 'estimated sources', String(minCount)
    ].map(value => { const label = document.createElement('span'); label.textContent = value; return label; }));
    xAxis?.replaceChildren(...[
      dateLabel(minTime), 'analysis date', dateLabel(maxTime)
    ].map(value => { const label = document.createElement('span'); label.textContent = value; return label; }));
    const points = [...items].reverse();
    points.forEach((row, index) => {
      const x = maxTime === minTime ? 50 : 6 + ((Number(row.ts) - minTime) / (maxTime - minTime)) * 88;
      const y = maxCount === minCount ? 50 : 83 - ((Number(row.count) - minCount) / (maxCount - minCount)) * 66;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'plot-point';
      button.dataset.historyId = row.id;
      button.style.setProperty('--x', `${x}%`);
      button.style.setProperty('--y', `${y}%`);
      button.setAttribute('aria-label', `${row.filename}: estimated ${row.count} ${row.count === 1 ? 'source' : 'sources'}, ${new Date(Number(row.ts)).toLocaleString()}`);
      const label = document.createElement('span');
      label.textContent = `${row.filename} · ${row.count} ${row.count === 1 ? 'source' : 'sources'}`;
      button.append(label);
      plot.append(button);
    });
  }

  function makeRow(row) {
    const article = document.createElement('article');
    article.className = 'history-row';
    article.dataset.historyId = row.id;
    const count = document.createElement('span');
    count.className = 'history-count-mark';
    count.textContent = String(row.count);
    count.setAttribute('aria-label', `${row.count} ${row.count === 1 ? 'source' : 'sources'}`);
    const details = document.createElement('div');
    details.className = 'history-file';
    const name = document.createElement('strong');
    name.textContent = row.filename;
    const meta = document.createElement('small');
    const date = new Date(Number(row.ts)).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
    meta.textContent = `${date} · ${formatTime(Number(row.duration))} · ${Math.round(Number(row.conf) * 100)}% model confidence`;
    details.append(name, meta);
    const bars = document.createElement('span');
    bars.className = 'history-mini-track';
    bars.setAttribute('aria-hidden', 'true');
    const values = Array.isArray(row.counts) ? row.counts.slice(0, 32) : [];
    values.forEach(value => {
      const bar = document.createElement('i');
      bar.style.setProperty('--bar-h', `${Math.max(8, Math.min(100, Number(value) / Math.max(1, row.count) * 100))}%`);
      bars.append(bar);
    });
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'history-delete';
    remove.dataset.deleteHistory = row.id;
    remove.setAttribute('aria-label', `Delete ${row.filename} from history`);
    remove.innerHTML = '<svg><use href="#icon-trash"/></svg>';
    article.append(count, details, bars, remove);
    return article;
  }

  function render() {
    currentRows = rows();
    const query = search.value.trim().toLocaleLowerCase();
    const visible = currentRows.filter(row => row.filename.toLocaleLowerCase().includes(query));
    renderStats(currentRows);
    countLabel.textContent = `${visible.length} ${visible.length === 1 ? 'record' : 'records'}`;
    drawPlot(visible);
    list.replaceChildren();
    if (!visible.length) {
      const empty = document.createElement('p');
      empty.className = 'history-empty';
      const message = document.createElement('span');
      message.textContent = currentRows.length ? 'No recordings match that search.' : 'No analyses saved yet. A result appears after the first completed analysis.';
      empty.append(message);
      if (!currentRows.length) {
        const link = document.createElement('a');
        link.href = '/analyze';
        link.dataset.nav = '';
        link.textContent = 'Analyze a recording';
        empty.append(link);
      }
      list.append(empty);
      return;
    }
    visible.forEach(row => list.append(makeRow(row)));
  }

  function parseCsv(text) {
    const output = [];
    let row = [], cell = '', quoted = false;
    for (let index = 0; index < text.length; index++) {
      const char = text[index];
      if (quoted && char === '"') {
        if (text[index + 1] === '"') { cell += '"'; index++; }
        else quoted = false;
      } else if (char === '"' && cell === '') quoted = true;
      else if (!quoted && char === ',') { row.push(cell); cell = ''; }
      else if (!quoted && (char === '\n' || char === '\r')) {
        if (char === '\r' && text[index + 1] === '\n') index++;
        row.push(cell); cell = '';
        if (row.some(value => value.trim())) output.push(row);
        row = [];
      } else cell += char;
    }
    row.push(cell);
    if (row.some(value => value.trim())) output.push(row);
    if (output.length < 2) return [];
    const headers = output.shift().map(value => value.trim().toLowerCase());
    return output.map(values => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ''])));
  }

  function parseDate(value) {
    const numeric = Number(value);
    if (Number.isFinite(numeric) && numeric > 0) return numeric;
    const parsed = Date.parse(String(value || ''));
    return Number.isFinite(parsed) ? parsed : NaN;
  }

  async function importCsv(file) {
    try {
      const text = await file.text();
      const parsed = parseCsv(text);
      const existing = history.all();
      const seen = new Set(existing.map(row => `${row.filename}| ${row.ts}| ${row.count}`));
      const added = [];
      for (const record of parsed) {
        const filename = String(record.file || record.filename || '').trim();
        const ts = parseDate(record.date || record.ts);
        const count = Number(record.estimated_sources ?? record.count);
        const conf = Number(record.model_confidence ?? record.avg_confidence ?? record.conf);
        const duration = Number(record.duration_s ?? record.duration);
        const key = `${filename}| ${ts}| ${count}`;
        if (!filename || !Number.isFinite(ts) || !Number.isSafeInteger(count) || count < 1
          || !Number.isFinite(conf) || conf < 0 || conf > 1 || !Number.isFinite(duration) || duration < 0
          || seen.has(key)) continue;
        const rawCounts = String(record.window_counts || record.counts || '').trim();
        const counts = rawCounts ? rawCounts.split(/[;\s|]+/).map(Number).filter(Number.isSafeInteger) : [];
        const row = {
          id: `import-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
          filename, ts, count, conf, duration, counts
        };
        added.push(row);
        seen.add(key);
      }
      if (!added.length) {
        toast(parsed.length ? 'No valid new rows were found in that CSV.' : 'The CSV does not contain history rows.', 'error');
        return;
      }
      history.save([...added, ...existing].filter(row => row && typeof row.id === 'string').slice(0, 200));
      render();
      toast(`Imported ${added.length} ${added.length === 1 ? 'analysis' : 'analyses'}.`);
    } catch (error) {
      toast(`Could not import that CSV: ${error.message}`, 'error');
    } finally {
      importInput.value = '';
    }
  }

  function csvCell(value) {
    let text = String(value ?? '');
    if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`;
    return `"${text.replace(/"/g, '""')}"`;
  }

  function exportCsv() {
    const records = currentRows;
    if (!records.length) return toast('There is no saved history to export.');
    const header = ['file', 'date', 'estimated_sources', 'model_confidence', 'duration_s', 'window_counts'];
    const lines = [header, ...records.map(row => [
      row.filename, new Date(Number(row.ts)).toISOString(), row.count, row.conf, row.duration,
      Array.isArray(row.counts) ? row.counts.join(' ') : ''
    ])].map(cells => cells.map(csvCell).join(','));
    const blob = new Blob([lines.join('\r\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'sanada-history.csv';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast('History exported as CSV.');
  }

  $('#importHistory').addEventListener('click', () => importInput.click());
  importInput.addEventListener('change', () => {
    if (importInput.files?.[0]) importCsv(importInput.files[0]);
  });
  $('#exportHistory').addEventListener('click', exportCsv);
  $('#clearHistory').addEventListener('click', () => {
    if (!history.all().length) return toast('History is already clear.');
    if (!confirm('Clear all saved analyses from this browser?')) return;
    history.clear();
    render();
    toast('Saved history cleared.');
  });
  search.addEventListener('input', render);
  list.addEventListener('click', event => {
    const button = event.target.closest('[data-delete-history]');
    if (!button) return;
    history.remove(button.dataset.deleteHistory);
    render();
    toast('Analysis removed from history.');
  });
  plot.addEventListener('click', event => {
    const point = event.target.closest('[data-history-id]');
    if (!point) return;
    const target = list.querySelector(`[data-history-id="${CSS.escape(point.dataset.historyId)}"]`);
    target?.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center' });
    target?.classList.add('is-linked');
    setTimeout(() => target?.classList.remove('is-linked'), 1400);
  });
  plot.addEventListener('pointerover', event => linkHover(event, true));
  plot.addEventListener('pointerout', event => linkHover(event, false));
  list.addEventListener('pointerover', event => linkHover(event, true));
  list.addEventListener('pointerout', event => linkHover(event, false));
  addEventListener('sanada:history-change', render);
  addEventListener('storage', event => { if (event.key === HISTORY_KEY) render(); });
  render();

  function linkHover(event, on) {
    const target = event.target.closest('[data-history-id]');
    if (!target) return;
    const id = target.dataset.historyId;
    const row = list.querySelector(`[data-history-id="${CSS.escape(id)}"]`);
    const point = plot.querySelector(`[data-history-id="${CSS.escape(id)}"]`);
    row?.classList.toggle('is-linked', on);
    point?.classList.toggle('is-linked', on);
  }
}

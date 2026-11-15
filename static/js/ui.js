const HISTORY_KEY = 'echocount.history.v1';
const THEME_KEY = 'echocount.theme';
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
export const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
export const formatTime = seconds => {
  const value = Math.max(0, Number(seconds) || 0);
  return `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
};

export function initUI({ scene } = {}) {
  const root = document.documentElement;
  const toastRoot = $('#toasts');
  const history = {
    all() {
      try {
        const value = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
        return Array.isArray(value) ? value : [];
      } catch { return []; }
    },
    save(rows) {
      try { localStorage.setItem(HISTORY_KEY, JSON.stringify(rows)); return true; }
      catch { toast('History could not be saved in this browser.', 'error'); return false; }
    },
    add(row) {
      const entries = history.all();
      entries.unshift({
        id: String(row.id),
        filename: String(row.filename),
        ts: Number(row.ts),
        count: Number(row.count),
        conf: Number(row.conf),
        duration: Number(row.duration),
        counts: Array.isArray(row.counts) ? row.counts.map(Number) : []
      });
      history.save(entries.filter(isHistoryRow).slice(0, 200));
    },
    remove(id) { history.save(history.all().filter(row => row.id !== id)); },
    clear() { history.save([]); }
  };
  function isHistoryRow(row) {
    return row && typeof row.id === 'string' && typeof row.filename === 'string'
      && Number.isFinite(Number(row.ts)) && Number.isSafeInteger(Number(row.count))
      && Number.isFinite(Number(row.conf)) && Number.isFinite(Number(row.duration));
  }
  function toast(message, kind = 'info') {
    if (!toastRoot) return;
    const node = document.createElement('div');
    node.className = 'toast';
    node.dataset.kind = kind;
    node.setAttribute('role', kind === 'error' ? 'alert' : 'status');
    node.textContent = String(message);
    toastRoot.append(node);
    setTimeout(() => {
      node.style.opacity = '0';
      node.style.transform = 'translateY(8px)';
      setTimeout(() => node.remove(), 220);
    }, 3600);
  }
  const ui = { $, $$, escapeHtml, formatTime, toast, history };
  window.SANADA = { ...(window.SANADA || {}), ui, scene };

  /* Theme state is saved with the existing preference key. */
  const themeButton = $('#themeToggle');
  const themeMeta = $('#themeColor');
  const commandShortcut = $('#commandShortcut');
  const applyTheme = theme => {
    root.dataset.theme = theme;
    if (themeMeta) themeMeta.content = theme === 'dark' ? '#07080B' : '#F1F0EB';
    themeButton?.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    dispatchEvent(new CustomEvent('sanada:themechange', { detail: { theme } }));
  };
  const storedTheme = () => {
    try {
      const value = localStorage.getItem(THEME_KEY);
      return value === 'light' || value === 'dark' ? value : null;
    } catch { return null; }
  };
  themeButton?.addEventListener('click', () => {
    const theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    root.classList.add('theme-changing');
    applyTheme(theme);
    try { localStorage.setItem(THEME_KEY, theme); } catch {}
    setTimeout(() => root.classList.remove('theme-changing'), 400);
  });
  const colorScheme = matchMedia('(prefers-color-scheme: dark)');
  applyTheme(storedTheme() || (colorScheme.matches ? 'dark' : 'light'));
  if (commandShortcut && /mac|iphone|ipad/i.test(navigator.platform || navigator.userAgent)) commandShortcut.textContent = '⌘ K';
  colorScheme.addEventListener?.('change', event => {
    if (!storedTheme()) applyTheme(event.matches ? 'dark' : 'light');
  });
  addEventListener('storage', event => {
    if (event.key === THEME_KEY && (event.newValue === 'light' || event.newValue === 'dark')) applyTheme(event.newValue);
  });

  /* Header and mobile navigation. */
  const header = $('#siteHeader');
  let headerFrame = 0;
  const updateHeader = () => {
    headerFrame = 0;
    header?.classList.toggle('is-scrolled', scrollY > 12);
  };
  const scheduleHeader = () => { if (!headerFrame) headerFrame = requestAnimationFrame(updateHeader); };
  addEventListener('scroll', scheduleHeader, { passive: true });
  updateHeader();
  const menuButton = $('#menuButton');
  const mobileNav = $('#mobileNav');
  function setMobileNav(open) {
    if (!menuButton || !mobileNav) return;
    const wasOpen = !mobileNav.hidden;
    mobileNav.hidden = !open;
    mobileNav.inert = !open;
    menuButton.setAttribute('aria-expanded', String(open));
    menuButton.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    menuButton.innerHTML = `<svg><use href="#icon-${open ? 'close' : 'menu'}"/></svg>`;
    if (open) mobileNav.querySelector('a')?.focus();
    else if (wasOpen) menuButton.focus({ preventScroll: true });
  }
  menuButton?.addEventListener('click', () => setMobileNav(mobileNav.hidden));
  $$('[data-nav]').forEach(link => link.addEventListener('click', () => {
    if (mobileNav && !mobileNav.hidden) setMobileNav(false);
  }));

  /* Command menu: the /how route is the Flask endpoint. */
  const menu = $('#commandMenu'), input = $('#commandInput'), list = $('#commandList');
  const commands = [
    { title: 'Open overview', kind: 'Page', run: () => { location.href = '/'; } },
    { title: 'Analyze audio', kind: 'Page', run: () => { location.href = '/analyze'; } },
    { title: 'Open analysis history', kind: 'Page', run: () => { location.href = '/history'; } },
    { title: 'Explore the method', kind: 'Page', run: () => { location.href = '/how'; } },
    { title: 'Read documentation', kind: 'Page', run: () => { location.href = '/docs'; } },
    { title: 'Toggle light / dark theme', kind: 'Action', run: () => themeButton?.click() },
    { title: 'Check server status', kind: 'Action', run: () => updateHealth(true) },
    { title: 'Copy training command', kind: 'Action', run: () => copyText('python train.py --metadata data/metadata.csv --epochs 25 --max-sources 31') },
    { title: 'Clear saved history', kind: 'Action', run: () => {
      if (!history.all().length || !confirm('Clear all saved analyses from this browser?')) return;
      history.clear();
      toast('Saved history cleared.');
      if (document.body.dataset.page === 'history') dispatchEvent(new CustomEvent('sanada:history-change'));
    } }
  ];
  let filtered = commands, selected = 0, returnFocus = null;
  function paintCommands() {
    if (!input || !list) return;
    const query = input.value.trim().toLowerCase();
    filtered = commands.filter(command => command.title.toLowerCase().includes(query));
    selected = Math.min(selected, Math.max(0, filtered.length - 1));
    list.innerHTML = filtered.length
      ? filtered.map((command, index) => `<li role="option" aria-selected="${index === selected}"><button type="button" data-command-index="${index}"><span class="command-mark" aria-hidden="true">↗</span><span>${escapeHtml(command.title)}</span><small>${escapeHtml(command.kind)}</small></button></li>`).join('')
      : '<li class="command-empty">No matching command.</li>';
  }
  function openCommand(open) {
    if (!menu || !input) return;
    if (open) {
      returnFocus = document.activeElement;
      menu.hidden = false;
      menu.inert = false;
      input.value = '';
      selected = 0;
      paintCommands();
      requestAnimationFrame(() => input.focus());
    } else {
      menu.hidden = true;
      menu.inert = true;
      returnFocus?.focus?.({ preventScroll: true });
    }
  }
  function runCommand(index) {
    const command = filtered[index];
    if (!command) return;
    openCommand(false);
    command.run();
  }
  $('#commandButton')?.addEventListener('click', () => openCommand(true));
  menu?.addEventListener('pointerdown', event => { if (event.target === menu) openCommand(false); });
  input?.addEventListener('input', () => { selected = 0; paintCommands(); });
  list?.addEventListener('click', event => {
    const button = event.target.closest('[data-command-index]');
    if (button) runCommand(Number(button.dataset.commandIndex));
  });
  list?.addEventListener('pointermove', event => {
    const option = event.target.closest('li[role="option"]');
    if (!option) return;
    const index = [...list.children].indexOf(option);
    if (index >= 0 && index !== selected) { selected = index; paintCommands(); }
  });
  addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      openCommand(menu?.hidden ?? true);
      return;
    }
    if (event.key === 'Escape') {
      if (menu && !menu.hidden) openCommand(false);
      if (mobileNav && !mobileNav.hidden) setMobileNav(false);
    }
    if (!menu || menu.hidden) return;
    if (event.key === 'ArrowDown' && filtered.length) { event.preventDefault(); selected = (selected + 1) % filtered.length; paintCommands(); }
    if (event.key === 'ArrowUp' && filtered.length) { event.preventDefault(); selected = (selected - 1 + filtered.length) % filtered.length; paintCommands(); }
    if (event.key === 'Enter' && filtered.length) { event.preventDefault(); runCommand(selected); }
    if (event.key === 'Tab') {
      event.preventDefault();
      input.focus();
    }
  });

  /* One actual scene is prepared once per session; progress represents readiness milestones. */
  initPreloader(scene);

  /* Motion helpers: fine-pointer only, with no transforms for reduced motion. */
  const cursor = $('#customCursor');
  const finePointer = matchMedia('(hover: hover) and (pointer: fine)').matches;
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (finePointer && !reduceMotion && cursor && ['home', 'how', 'analyze'].includes(document.body.dataset.page)) {
    document.body.classList.add('has-custom-cursor');
    document.addEventListener('pointermove', event => {
      cursor.style.left = `${event.clientX}px`;
      cursor.style.top = `${event.clientY}px`;
      cursor.classList.add('is-visible');
      cursor.classList.toggle('is-active', Boolean(event.target.closest('a,button')));
      cursor.classList.toggle('is-wave', Boolean(event.target.closest('.drop-zone')));
    }, { passive: true });
    document.addEventListener('pointerleave', () => cursor.classList.remove('is-visible'));
    document.addEventListener('pointerover', event => {
      if (event.target.closest('input,textarea,select')) cursor.classList.remove('is-visible');
    });
    $$('.magnetic').forEach(button => {
      button.addEventListener('pointermove', event => {
        const rect = button.getBoundingClientRect();
        const dx = (event.clientX - rect.left - rect.width / 2) * .08;
        const dy = (event.clientY - rect.top - rect.height / 2) * .1;
        button.style.setProperty('--magnetic-x', `${dx}px`);
        button.style.setProperty('--magnetic-y', `${dy}px`);
      });
      button.addEventListener('pointerleave', () => {
        button.style.removeProperty('--magnetic-x');
        button.style.removeProperty('--magnetic-y');
      });
    });
  }

  /* Scroll reveals are progressive enhancement: content remains visible without IO. */
  const revealNodes = $$('.reveal');
  if ('IntersectionObserver' in window && !reduceMotion) {
    const revealObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          revealObserver.unobserve(entry.target);
        }
      });
    }, { threshold: .14 });
    revealNodes.forEach(node => revealObserver.observe(node));
  } else revealNodes.forEach(node => node.classList.add('is-visible'));

  /* Docs tabs support click and arrow-key operation. */
  $$('[data-tabs]').forEach(tabRoot => {
    const tabs = $$('[role="tab"]', tabRoot);
    const select = tab => {
      tabs.forEach(item => {
        const active = item === tab;
        item.setAttribute('aria-selected', String(active));
        item.tabIndex = active ? 0 : -1;
        const panel = document.getElementById(item.getAttribute('aria-controls'));
        if (panel) panel.hidden = !active;
      });
    };
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => select(tab));
      tab.addEventListener('keydown', event => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
          : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
        select(tabs[next]);
        tabs[next].focus();
      });
    });
  });

  document.addEventListener('click', async event => {
    const button = event.target.closest('[data-copy]');
    if (!button) return;
    const source = $(button.dataset.copy);
    if (!source) return;
    const success = await copyText(source.textContent.trim(), false);
    if (success) {
      button.dataset.copied = 'true';
      const oldText = button.textContent;
      button.textContent = 'Copied';
      setTimeout(() => { button.textContent = oldText; delete button.dataset.copied; }, 1300);
    } else toast('Clipboard access was unavailable. Select and copy the text manually.', 'error');
  });

  $('#healthCheck')?.addEventListener('click', () => updateHealth(true));
  const tocLinks = $$('.docs-toc a');
  const docsSections = $$('.docs-section');
  if (tocLinks.length && 'IntersectionObserver' in window) {
    const activeObserver = new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) tocLinks.forEach(link => link.classList.toggle('is-active', link.hash === `#${entry.target.id}`));
    }), { rootMargin: '-25% 0px -65% 0px' });
    docsSections.forEach(section => activeObserver.observe(section));
  }

  updateHealth(false);
  return ui;

  async function copyText(text, notify = true) {
    try {
      await navigator.clipboard.writeText(text);
      if (notify) toast('Copied to clipboard.');
      return true;
    } catch {
      const temporary = document.createElement('textarea');
      temporary.value = text;
      temporary.style.position = 'fixed';
      temporary.style.opacity = '0';
      document.body.append(temporary);
      temporary.select();
      const copied = document.execCommand('copy');
      temporary.remove();
      if (copied && notify) toast('Copied to clipboard.');
      return copied;
    }
  }

  async function updateHealth(showMessage) {
    const pill = $('#statusPill');
    const output = $('#healthOutput');
    try {
      const response = await fetch('/api/health', { headers: { Accept: 'application/json' } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const trained = Boolean(data.model?.trained);
      if (pill) {
        pill.dataset.state = trained ? 'ready' : 'untrained';
        pill.querySelector('span').textContent = trained ? 'Model ready' : 'Model not trained';
      }
      if (output) output.textContent = JSON.stringify(data, null, 2);
      if (showMessage) toast(trained ? 'Server is ready and the model checkpoint was found.' : 'Server is running; no trained checkpoint was found.');
    } catch (error) {
      if (pill) {
        pill.dataset.state = 'offline';
        pill.querySelector('span').textContent = 'Server offline';
      }
      if (output) output.textContent = `The server did not respond: ${error.message}`;
      if (showMessage) toast('The server did not respond.', 'error');
    }
  }
}

function initPreloader(scene) {
  const loader = document.getElementById('preloader');
  if (!loader) return;
  let alreadyShown = false;
  try { alreadyShown = sessionStorage.getItem('sanada.preloader.seen') === '1'; } catch {}
  if (alreadyShown || matchMedia('(prefers-reduced-motion: reduce)').matches) {
    loader.hidden = true;
    return;
  }
  const bar = document.getElementById('preloaderBar');
  const meter = loader.querySelector('[role="progressbar"]');
  const percent = document.getElementById('preloaderPercent');
  const status = document.getElementById('preloaderStatus');
  const skip = document.getElementById('preloaderSkip');
  const milestones = new Map([
    ['Interface scripts ready', Promise.resolve()],
    ['Typography ready', document.fonts?.ready || Promise.resolve()],
    ['Scene ready', scene?.ready || Promise.resolve()]
  ]);
  const order = [...milestones.entries()];
  let complete = 0, dismissed = false;
  const update = label => {
    if (dismissed) return;
    complete += 1;
    const value = Math.round((complete / order.length) * 100);
    status.textContent = label;
    percent.textContent = String(value);
    meter?.setAttribute('aria-valuenow', String(value));
    bar?.style.setProperty('width', `${value}%`);
    if (complete >= order.length) finish();
  };
  const finish = (ready = true) => {
    if (dismissed) return;
    dismissed = true;
    if (ready) status.textContent = 'Interface ready';
    try { sessionStorage.setItem('sanada.preloader.seen', '1'); } catch {}
    setTimeout(() => { loader.classList.add('is-done'); loader.inert = true; }, 180);
    setTimeout(() => { loader.hidden = true; }, 850);
  };
  skip?.addEventListener('click', () => finish(false));
  order.forEach(([label, promise]) => Promise.resolve(promise).then(() => update(label), () => update(label)));
}

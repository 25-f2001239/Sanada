import { createScene } from './scene.js';
import { initScroll } from './scroll.js';
import { initUI } from './ui.js';
import { initAnalyzer } from './analyze.js';
import { initHistory } from './history.js';

const scene = createScene();
const ui = initUI({ scene });
const scroll = initScroll(scene);
window.SANADA = { ...(window.SANADA || {}), scene, ui, scroll };

if (document.body.dataset.page === 'analyze') initAnalyzer({ scene, ui });
if (document.body.dataset.page === 'history') initHistory(ui);

addEventListener('pagehide', () => {
  scroll?.destroy?.();
  scene?.destroy?.();
}, { once: true });

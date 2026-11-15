const clamp01 = value => Math.max(0, Math.min(1, value));

function loadScript(src, available, timeout = 2600) {
  if (available()) return Promise.resolve(true);
  return new Promise(resolve => {
    let settled = false;
    const finish = value => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(value);
    };
    const timer = setTimeout(() => finish(available()), timeout);
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.crossOrigin = 'anonymous';
    script.onload = () => finish(available());
    script.onerror = () => finish(false);
    document.head.append(script);
  });
}

export function initScroll(scene) {
  const story = document.querySelector('.world-story');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const mobile = matchMedia('(max-width: 760px), (pointer: coarse)').matches;
  let gsapReady = false, lenis = null, lenisTick = null, timeline = null, nativeFrame = 0;
  const storyTriggers = [];

  function applyProgress(value) {
    const normalized = clamp01(value);
    scene.setProgress(normalized);
    const counter = document.getElementById('heroSourceCount');
    if (counter && document.body.dataset.page === 'home') {
      const count = Math.min(4, 1 + Math.floor(Math.min(.999, normalized / .22) * 4));
      counter.textContent = String(count).padStart(2, '0');
      counter.setAttribute('aria-label', `${count} illustrative source ${count === 1 ? 'lane' : 'lanes'}`);
    }
    const stageCounter = document.getElementById('methodStageCount');
    if (stageCounter && document.body.dataset.page === 'how') {
      stageCounter.textContent = String(Math.min(4, 1 + Math.floor(normalized * 4))).padStart(2, '0');
    }
  }

  function readScroll() {
    nativeFrame = 0;
    if (!story || gsapReady) return;
    const rect = story.getBoundingClientRect();
    const range = Math.max(1, rect.height - innerHeight);
    applyProgress(-rect.top / range);
  }

  function queueRead() {
    if (!nativeFrame) nativeFrame = requestAnimationFrame(readScroll);
  }

  function revealBeats() {
    const beats = [...document.querySelectorAll('.story-beat')];
    if (!beats.length) return;
    if (reduced || !('IntersectionObserver' in window)) {
      beats.forEach(beat => beat.classList.add('is-active'));
      return;
    }
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => entry.target.classList.toggle('is-active', entry.isIntersecting));
    }, { threshold: .28 });
    beats.forEach(beat => observer.observe(beat));
  }

  revealBeats();
  addEventListener('scroll', queueRead, { passive: true });
  addEventListener('resize', queueRead, { passive: true });
  queueRead();

  if (reduced || mobile || !story) {
    return { destroy() { cancelAnimationFrame(nativeFrame); removeEventListener('scroll', queueRead); removeEventListener('resize', queueRead); } };
  }

  const gsapSrc = 'https://cdn.jsdelivr.net/npm/gsap@3.15.1/dist/gsap.min.js';
  const triggerSrc = 'https://cdn.jsdelivr.net/npm/gsap@3.15.1/dist/ScrollTrigger.min.js';
  const lenisSrc = 'https://unpkg.com/lenis@1.3.27/dist/lenis.min.js';
  const motionReady = loadScript(gsapSrc, () => Boolean(window.gsap)).then(async hasGsap => {
    if (!hasGsap || !window.gsap) return false;
    const hasTrigger = await loadScript(triggerSrc, () => Boolean(window.ScrollTrigger || window.ScrollTriggerPlugin));
    if (!hasTrigger) return false;
    const ScrollTrigger = window.ScrollTrigger || window.ScrollTriggerPlugin;
    if (!ScrollTrigger) return false;
    window.gsap.registerPlugin(ScrollTrigger);
    const hasLenis = await loadScript(lenisSrc, () => Boolean(window.Lenis));
    if (hasLenis && window.Lenis) {
      try {
        lenis = new window.Lenis({ smoothWheel: true, syncTouch: false, autoRaf: false, anchors: true, stopInertiaOnNavigate: true });
        lenis.on('scroll', ScrollTrigger.update);
        lenisTick = time => lenis?.raf(time * 1000);
        window.gsap.ticker.add(lenisTick);
        window.gsap.ticker.lagSmoothing(0);
        window.SANADA && (window.SANADA.lenis = lenis);
      } catch {
        lenis = null;
      }
    }

    const state = { value: 0 };
    timeline = window.gsap.timeline({
      scrollTrigger: {
        trigger: story,
        start: 'top top',
        end: 'bottom bottom',
        scrub: 1,
        invalidateOnRefresh: true,
        onUpdate: self => applyProgress(self.progress)
      }
    }).to(state, {
      value: 1,
      duration: 1,
      ease: 'none',
      onUpdate: () => applyProgress(state.value)
    });
    document.querySelectorAll('.story-beat').forEach(beat => {
      storyTriggers.push(ScrollTrigger.create({
        trigger: beat,
        start: 'top 68%',
        end: 'bottom 32%',
        onToggle: self => beat.classList.toggle('is-active', self.isActive)
      }));
    });
    document.fonts?.ready.then(() => ScrollTrigger.refresh());
    ScrollTrigger.refresh();
    gsapReady = true;
    return true;
  }).catch(() => false);

  motionReady.then(queueRead);

  return {
    ready: motionReady,
    destroy() {
      cancelAnimationFrame(nativeFrame);
      removeEventListener('scroll', queueRead);
      removeEventListener('resize', queueRead);
      timeline?.scrollTrigger?.kill();
      timeline?.kill();
      storyTriggers.forEach(trigger => trigger.kill());
      if (lenisTick && window.gsap) window.gsap.ticker.remove(lenisTick);
      lenis?.destroy();
    }
  };
}

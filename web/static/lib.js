// Shared bits for the dashboard modules: rendering, API calls, toasts and
// the small controls every page uses.
import { h } from './vendor/preact.js';
import { useEffect, useRef, useState } from './vendor/hooks.js';
import htm from './vendor/htm.js';
import { t } from './strings.js';

export const html = htm.bind(h);
export { t };

export const S = (v) => (v === null || v === undefined ? '' : String(v));
export const num = (n) => (n ?? 0).toLocaleString();
export const when = (iso) => (iso ? String(iso).slice(0, 16).replace('T', ' ') : '');
export const secs = (s) => (s === null || s === undefined ? '—' : `${Number(s).toFixed(1)} s`);
export const mb = (bytes) => `${((bytes || 0) / 1048576).toFixed((bytes || 0) < 10485760 ? 1 : 0)} MB`;

export function load(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v === null ? fallback : JSON.parse(v);
  } catch { return fallback; }
}

export function save(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ }
}

// ---------------------------------------------------------------- toasts

const toastListeners = new Set();
export function toast(text, error = false) {
  for (const fn of toastListeners) fn(text, error);
}
export function onToast(fn) {
  toastListeners.add(fn);
  return () => toastListeners.delete(fn);
}

// ---------------------------------------------------------------- API

// JSON API call. Throws an Error with .field (which part of a form is wrong) and .status.
export async function api(method, url, body) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(url, opts);
  if (r.status === 401) { location.href = '/login'; throw new Error(t('common.signedOut')); }
  let data = {};
  try { data = await r.json(); } catch { /* not JSON */ }
  if (!r.ok) {
    const err = new Error(data.error || `HTTP ${r.status}`);
    err.field = data.field;
    err.status = r.status;
    throw err;
  }
  return data;
}

// run(fn, okText): busy flag, a toast on success, the error as a toast on failure.
export function useRunner() {
  const [busy, setBusy] = useState(false);
  const run = async (fn, okText) => {
    setBusy(true);
    try {
      const out = await fn();
      if (okText) toast(okText);
      return out ?? true;
    } catch (err) {
      toast(err.message || String(err), true);
      return false;
    } finally {
      setBusy(false);
    }
  };
  return [run, busy];
}

// ---------------------------------------------------------------- icons

// 20px line icons drawn on a 24 grid (stroke = currentColor).
const ICONS = {
  overview: 'M3 12l9-8 9 8M5 10v10h5v-6h4v6h5V10',
  live: 'M3 12h3l3-7 4 14 3-9 2 2h3',
  gags: 'M4 5h16a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1h-9l-5 4v-4H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1zM8 10h8M8 13h5',
  voices: 'M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3zM5 11a7 7 0 0 0 14 0M12 18v3M9 21h6',
  sounds: 'M4 9h4l5-4v14l-5-4H4zM16.5 8.5a5 5 0 0 1 0 7M19 6a8.5 8.5 0 0 1 0 12',
  people: 'M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM2 21a7 7 0 0 1 14 0M17 3.5a4 4 0 0 1 0 7.5M22 21a7 7 0 0 0-4.5-6.5',
  stats: 'M4 20V10M10 20V4M16 20v-7M22 20H2',
  settings: 'M4 7h10M18 7h2M4 17h4M12 17h8M14 4v6M8 14v6',
  packs: 'M3 7l9-4 9 4-9 4-9-4zM3 7v10l9 4 9-4V7M12 11v10',
  check: 'M5 12l5 5 9-10',
  dot: 'M12 12h.01',
  chevron: 'M9 6l6 6-6 6',
  menu: 'M4 6h16M4 12h16M4 18h16',
  close: 'M6 6l12 12M18 6L6 18',
  sun: 'M12 4V2M12 22v-2M4 12H2M22 12h-2M5 5l1.5 1.5M17.5 17.5L19 19M5 19l1.5-1.5M17.5 6.5L19 5M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z',
  moon: 'M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z',
  auto: 'M12 3a9 9 0 1 0 0 18zM12 3a9 9 0 0 1 0 18',
  out: 'M15 4h4a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1h-4M10 8l-4 4 4 4M6 12h11',
  play: 'M7 5v14l12-7z',
  stop: 'M6 6h12v12H6z',
  plus: 'M12 5v14M5 12h14',
  up: 'M12 19V5M6 11l6-6 6 6',
  down: 'M12 5v14M6 13l6 6 6-6',
  trash: 'M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13',
  edit: 'M4 20h4L19 9l-4-4L4 16zM13 7l4 4',
  info: 'M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 11v6M12 7.5h.01',
  warn: 'M12 3l10 18H2zM12 10v5M12 18h.01',
  search: 'M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14zM21 21l-5-5',
  refresh: 'M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7',
  server: 'M4 4h16v6H4zM4 14h16v6H4zM8 7h.01M8 17h.01',
  external: 'M14 4h6v6M20 4l-9 9M18 14v6H4V6h6',
};

export function Icon({ name, size = 20, label }) {
  return html`<svg class="icon" width=${size} height=${size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden=${label ? undefined : 'true'}
    role=${label ? 'img' : undefined} aria-label=${label}><path d=${ICONS[name] || ICONS.dot} /></svg>`;
}

// The Heckler mark: a speech bubble with an exclamation, on the accent.
export function Mark({ size = 28 }) {
  return html`<svg class="mark" width=${size} height=${size} viewBox="0 0 32 32" aria-hidden="true">
    <rect width="32" height="32" rx="9" class="mark-bg" />
    <path class="mark-fg" d="M9 9.5h14a2.5 2.5 0 0 1 2.5 2.5v7a2.5 2.5 0 0 1-2.5 2.5h-7.2L11 25v-3.5H9A2.5 2.5 0 0 1 6.5 19v-7A2.5 2.5 0 0 1 9 9.5z" />
    <path class="mark-bang" d="M16 12.6v4" stroke-width="2.4" stroke-linecap="round" />
    <circle class="mark-dot" cx="16" cy="19.2" r="1.35" />
  </svg>`;
}

// ---------------------------------------------------------------- controls

let uid = 0;
export function useId(prefix = 'f') {
  const ref = useRef(null);
  if (ref.current === null) ref.current = `${prefix}-${++uid}`;
  return ref.current;
}

// A labelled on/off switch (a real checkbox underneath, so keyboard and screen readers work).
export function Toggle({ checked, onChange, label, hint, disabled, id }) {
  const own = useId('tg');
  const fid = id || own;
  return html`<div class="toggle-row">
    <input type="checkbox" role="switch" class="switch" id=${fid} checked=${!!checked} disabled=${disabled}
      aria-describedby=${hint ? `${fid}-hint` : undefined} onChange=${(e) => onChange(e.target.checked)} />
    ${label && html`<label for=${fid} class="toggle-label">${label}
      ${hint && html`<span class="hint" id=${`${fid}-hint`}>${hint}</span>`}</label>`}
  </div>`;
}

// Label + control + help text, stacked.
export function Field({ label, hint, error, children, id, wide }) {
  return html`<div class="field ${wide ? 'wide' : ''} ${error ? 'has-error' : ''}">
    <label class="field-label" for=${id}>${label}</label>
    ${children}
    ${error ? html`<span class="field-error" role="alert">${error}</span>` : hint && html`<span class="hint">${hint}</span>`}
  </div>`;
}

// A range slider with its value beside it.
export function Slider({ id, value, min = 0, max = 100, step = 1, onInput, onChange, format = (v) => v, disabled }) {
  return html`<div class="slider">
    <input type="range" id=${id} min=${min} max=${max} step=${step} value=${value} disabled=${disabled}
      onInput=${(e) => onInput && onInput(Number(e.target.value))}
      onChange=${(e) => onChange && onChange(Number(e.target.value))} />
    <output for=${id} class="slider-value">${format(value)}</output>
  </div>`;
}

// A button that asks once more before doing something destructive.
export function ConfirmButton({ label, confirmLabel, onConfirm, disabled, cls = '', icon }) {
  const [armed, setArmed] = useState(false);
  const yes = useRef(null);
  useEffect(() => {
    if (!armed) return undefined;
    if (yes.current) yes.current.focus();
    const timer = setTimeout(() => setArmed(false), 6000);
    return () => clearTimeout(timer);
  }, [armed]);
  if (!armed) {
    return html`<button type="button" class="btn ${cls}" disabled=${disabled} onClick=${() => setArmed(true)}>
      ${icon && html`<${Icon} name=${icon} size=${16} />`}${label}</button>`;
  }
  return html`<span class="confirm" role="group">
    <button type="button" ref=${yes} class="btn danger ${cls.includes('small') ? 'small' : ''}"
      onClick=${() => { setArmed(false); onConfirm(); }}>${confirmLabel}</button>
    <button type="button" class="btn ghost ${cls.includes('small') ? 'small' : ''}" onClick=${() => setArmed(false)}>${t('common.cancel')}</button>
  </span>`;
}

// A list of short strings edited as chips: Enter or comma adds, × removes.
export function Chips({ values, onChange, placeholder, invalid = false, id, label }) {
  const [text, setText] = useState('');
  const list = values || [];
  const commit = () => {
    const parts = text.split(',').map((s) => s.trim()).filter(Boolean);
    if (parts.length) onChange([...list, ...parts.filter((p) => !list.includes(p))]);
    setText('');
  };
  return html`<div class="chips-input ${invalid ? 'invalid' : ''}">
    ${list.map((v, i) => html`<span class="chip" key=${v}>${v}
      <button type="button" class="x" aria-label=${t('common.removeItem', { item: v })}
        onClick=${() => onChange(list.filter((_, j) => j !== i))}>×</button></span>`)}
    <input type="text" id=${id} value=${text} placeholder=${list.length ? '' : placeholder} aria-label=${label}
      onInput=${(e) => setText(e.target.value)}
      onKeyDown=${(e) => {
        if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); commit(); }
        else if (e.key === 'Backspace' && !text && list.length) onChange(list.slice(0, -1));
      }}
      onBlur=${commit} />
  </div>`;
}

// An input that saves on Enter or when it loses focus.
export function Commit({ value, onCommit, type = 'text', placeholder, id, label, min, max, step, cls = '' }) {
  const [text, setText] = useState(S(value));
  useEffect(() => setText(S(value)), [value]);
  const commit = () => {
    if (text === S(value)) return;
    if (type === 'number') onCommit(text === '' ? null : Number(text));
    else onCommit(text.trim());
  };
  return html`<input type=${type} id=${id} class=${cls} value=${text} placeholder=${placeholder} aria-label=${label}
    min=${min} max=${max} step=${step}
    onInput=${(e) => setText(e.target.value)} onBlur=${commit}
    onKeyDown=${(e) => { if (e.key === 'Enter') { e.preventDefault(); commit(); } else if (e.key === 'Escape') setText(S(value)); }} />`;
}

// A friendly empty state: what goes here, and the first thing to do.
export function Empty({ title, body, action }) {
  return html`<div class="empty-state">
    <p class="empty-title">${title}</p>
    ${body && html`<p class="empty-body">${body}</p>`}
    ${action}
  </div>`;
}

// Tabs inside a page. items: [[key, label, badge?]]
export function Tabs({ items, value, onChange, label }) {
  return html`<div class="tabs" role="tablist" aria-label=${label}>
    ${items.map(([key, text, badge]) => html`<button key=${key} type="button" role="tab" aria-selected=${value === key}
      class="tab ${value === key ? 'active' : ''}" onClick=${() => onChange(key)}>
      ${text}${badge ? html`<span class="badge">${badge}</span>` : ''}</button>`)}
  </div>`;
}

// A status pill: tone = ok | warn | bad | info | muted.
export function Pill({ tone = 'muted', children, title }) {
  return html`<span class="pill ${tone}" title=${title}><span class="dot"></span>${children}</span>`;
}

// ---------------------------------------------------------------- page frame

// The title and one-line description of a section, with its actions on the right.
export function PageHead({ section, title, desc, children }) {
  return html`<header class="page-head">
    <div class="page-title">
      <h1>${title || t(`page.${section}.title`)}</h1>
      <p class="page-desc">${desc || t(`page.${section}.desc`)}</p>
    </div>
    ${children && html`<div class="page-actions">${children}</div>`}
  </header>`;
}

// What a content page shows while its data loads, or when the bot runs without a store.
export function Waiting({ section, missing, guild }) {
  return html`<div class="page">
    <${PageHead} section=${section} />
    ${!guild ? html`<${Empty} title=${t('common.noServerTitle')} body=${t('common.noServerBody')} />`
      : missing ? html`<${Empty} title=${t('common.noStoreTitle')} body=${t('common.noStoreBody')} />`
        : html`<p class="muted loading">${t('common.loading')}</p>`}
  </div>`;
}

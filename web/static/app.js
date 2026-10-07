// Heckler admin dashboard: the shell (sidebar, server switcher, routing, live
// connection, toasts). Each section lives in its own module. No build step.
import { render } from './vendor/preact.js';
import { useEffect, useReducer, useRef, useState } from './vendor/hooks.js';
import { api, html, Icon, load, Mark, onToast, S, save, t, toast } from './lib.js';
import { Overview } from './overview.js';
import { Live, feedReducer } from './live.js';
import { Gags } from './gags.js';
import { Sounds, Voices } from './media.js';
import { People } from './people.js';
import { Stats } from './stats.js';
import { Settings } from './settings.js';
import { Packs } from './packs.js';

const SECTIONS = [
  ['overview', 'overview', Overview],
  ['live', 'live', Live],
  ['gags', 'gags', Gags],
  ['voices', 'voices', Voices],
  ['sounds', 'sounds', Sounds],
  ['people', 'people', People],
  ['stats', 'stats', Stats],
  ['settings', 'settings', Settings],
  ['packs', 'packs', Packs],
];
const KEYS = SECTIONS.map(([k]) => k);

// ---------------------------------------------------------------- theme

const THEMES = ['system', 'light', 'dark'];
function applyTheme(theme) {
  if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}
applyTheme(load('heckler.theme', 'system'));

// ---------------------------------------------------------------- live connection

function useSocket(onMessage) {
  const [state, setState] = useState('connecting');
  const handler = useRef(onMessage);
  handler.current = onMessage;
  useEffect(() => {
    let ws = null, timer = null, delay = 1000, stopped = false;
    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      ws = new WebSocket(`${proto}//${location.host}/ws`);
      ws.onopen = () => { delay = 1000; setState('live'); };
      ws.onmessage = (m) => {
        let msg;
        try { msg = JSON.parse(m.data); } catch { return; }
        handler.current(msg);
      };
      ws.onclose = async () => {
        if (stopped) return;
        setState('down');
        try {
          const r = await fetch('/api/status', { cache: 'no-store' });
          if (r.status === 401) { location.href = '/login'; return; }
        } catch { /* server down: keep retrying */ }
        timer = setTimeout(connect, delay);
        delay = Math.min(delay * 2, 15000);
      };
    };
    connect();
    return () => { stopped = true; clearTimeout(timer); if (ws) ws.close(); };
  }, []);
  return state;
}

// The selected server's editable content (GET /api/g/{gid}/content), refreshed
// when the store changes (content_changed over the socket).
function useContent(gid, changed) {
  const [data, setData] = useState(null);
  const [missing, setMissing] = useState(false);
  const seq = useRef(0);
  const reload = async () => {
    if (!gid) return;
    const mine = ++seq.current;
    try {
      const d = await api('GET', `/api/g/${gid}/content`);
      if (mine === seq.current) { setData(d); setMissing(false); }
    } catch (err) {
      if (mine !== seq.current) return;
      if (err.status === 404) setMissing(true);
      else toast(err.message, true);
    }
  };
  useEffect(() => { setData(null); reload(); }, [gid]);
  useEffect(() => {
    if (changed && (changed.guild_id === null || changed.guild_id === undefined || S(changed.guild_id) === gid)) reload();
  }, [changed]);
  return { data, missing, reload };
}

// ---------------------------------------------------------------- shell parts

function Toasts() {
  const [items, setItems] = useState([]);
  useEffect(() => onToast((text, error) => {
    const id = Math.random();
    setItems((xs) => [...xs.slice(-3), { id, text, error }]);
    setTimeout(() => setItems((xs) => xs.filter((x) => x.id !== id)), error ? 7000 : 3500);
  }), []);
  return html`<div class="toasts" role="status" aria-live="polite">
    ${items.map((x) => html`<div key=${x.id} class="toast ${x.error ? 'error' : ''}">
      <${Icon} name=${x.error ? 'warn' : 'check'} size=${16} /><span>${x.text}</span></div>`)}
  </div>`;
}

function ServerSwitcher({ guilds, gid, setGid }) {
  if (!guilds.length) return html`<p class="switcher-empty">${t('shell.noServers')}</p>`;
  return html`<div class="switcher">
    <label for="server-switcher" class="eyebrow">${t('shell.server')}</label>
    <div class="select-wrap">
      <${Icon} name="server" size=${16} />
      <select id="server-switcher" value=${gid} onChange=${(e) => setGid(e.target.value)}>
        ${guilds.map((g) => html`<option key=${g.id} value=${S(g.id)}>${g.name}</option>`)}
      </select>
    </div>
  </div>`;
}

function Sidebar({ section, go, guilds, gid, setGid, status, socket, theme, cycleTheme, badges, open, close }) {
  const bot = status?.bot;
  const themeIcon = { system: 'auto', light: 'sun', dark: 'moon' }[theme];
  return html`<aside class="sidebar ${open ? 'open' : ''}" aria-label=${t('shell.navigation')}>
    <div class="brand">
      <${Mark} size=${30} />
      <div class="brand-text">
        <span class="wordmark">Heckler</span>
        <span class="instance" title=${t('shell.instanceTitle')}>
          <span class="dot ${bot?.connected ? 'ok' : 'bad'}"></span>${bot?.name || t('shell.connecting')}
        </span>
      </div>
      <button class="btn icon ghost only-mobile" aria-label=${t('shell.closeMenu')} onClick=${close}><${Icon} name="close" /></button>
    </div>
    <${ServerSwitcher} guilds=${guilds} gid=${gid} setGid=${setGid} />
    <nav class="nav">
      ${SECTIONS.map(([key, icon]) => html`<a key=${key} href=${`#${key}`} class="nav-item ${section === key ? 'active' : ''}"
        aria-current=${section === key ? 'page' : undefined} onClick=${(e) => { e.preventDefault(); go(key); }}>
        <${Icon} name=${icon} /><span>${t(`nav.${key}`)}</span>
        ${badges[key] ? html`<span class="badge" title=${t(`nav.badge.${key}`, { n: badges[key] })}>${badges[key]}</span>` : ''}
      </a>`)}
    </nav>
    <div class="sidebar-foot">
      <span class="conn ${socket === 'live' ? 'ok' : 'warn'}"><span class="dot"></span>${socket === 'live' ? t('shell.live') : t('shell.reconnecting')}</span>
      <div class="foot-actions">
        <button class="btn icon ghost" onClick=${cycleTheme} title=${t(`shell.theme.${theme}`)} aria-label=${t(`shell.theme.${theme}`)}>
          <${Icon} name=${themeIcon} /></button>
        <form method="post" action="/logout"><button class="btn icon ghost" type="submit" title=${t('shell.signOut')} aria-label=${t('shell.signOut')}>
          <${Icon} name="out" /></button></form>
      </div>
    </div>
  </aside>`;
}

// ---------------------------------------------------------------- app

function App() {
  const [status, setStatus] = useState(null);
  const [feed, dispatch] = useReducer(feedReducer, []);
  const [changed, setChanged] = useState(null);
  const [busy, setBusy] = useState({});
  const [theme, setTheme] = useState(() => load('heckler.theme', 'system'));
  const [section, setSection] = useState(() => (KEYS.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'overview'));
  const [menu, setMenu] = useState(false);
  const [picked, setPicked] = useState(() => load('heckler.server', ''));

  const socket = useSocket((msg) => {
    if (msg.type === 'status') { const { type, ...rest } = msg; setStatus(rest); }
    else if (msg.type === 'content_changed') setChanged({ ...msg, at: Date.now() });
    else dispatch(msg);
  });

  const guilds = status?.guilds || [];
  const ids = guilds.map((g) => S(g.id));
  const gid = ids.includes(picked) ? picked : ids[0] || '';
  const guild = guilds.find((g) => S(g.id) === gid) || null;
  const setGid = (v) => { setPicked(v); save('heckler.server', v); };
  const content = useContent(gid, changed);

  useEffect(() => {
    const onHash = () => { const k = location.hash.slice(1); if (KEYS.includes(k)) setSection(k); };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  useEffect(() => {
    const name = status?.bot?.name;
    document.title = `${t(`nav.${section}`)} · ${name ? `${name} · ` : ''}Heckler`;
  }, [section, status?.bot?.name]);
  useEffect(() => {
    const main = document.getElementById('main');
    if (main) main.scrollTop = 0;
  }, [section]);

  const go = (key) => {
    setSection(key);
    setMenu(false);
    if (location.hash !== `#${key}`) history.replaceState(null, '', `#${key}`);
  };
  const cycleTheme = () => {
    const next = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
    setTheme(next); save('heckler.theme', next); applyTheme(next);
    toast(t(`shell.theme.${next}`));
  };

  // A bot control (join, say, toggle...). Returns true on success.
  const control = async (action, params = {}) => {
    const key = params.guild_id ? `${action}:${params.guild_id}` : action;
    setBusy((b) => ({ ...b, [key]: true }));
    try {
      const data = await api('POST', '/api/control', { action, ...params });
      if (data.status) setStatus(data.status);
      toast(data.message || t('common.done'));
      return true;
    } catch (err) {
      if (action === 'restart' && err instanceof TypeError) { toast(t('overview.restarting')); return true; }
      toast(err.message || String(err), true);
      return false;
    } finally {
      setBusy((b) => { const n = { ...b }; delete n[key]; return n; });
    }
  };

  const d = content.data;
  const badges = {
    gags: d ? d.reactions.filter((r) => r.status === 'pending').length : 0,
    people: d ? d.requests.filter((q) => q.status === 'pending').length : 0,
  };
  const Page = SECTIONS.find(([k]) => k === section)[2];
  const props = {
    status, socket, gid, guild, guilds, setGid, feed, dispatch, changed, control, busy, go,
    data: d, missing: content.missing, reload: content.reload,
  };

  return html`<div class="shell">
    <header class="topbar only-mobile">
      <button class="btn icon ghost" aria-label=${t('shell.openMenu')} aria-expanded=${menu} onClick=${() => setMenu(true)}><${Icon} name="menu" /></button>
      <${Mark} size=${24} /><span class="wordmark">Heckler</span>
      <span class="spacer"></span>
      ${guild && html`<span class="topbar-server">${guild.name}</span>`}
    </header>
    ${menu && html`<div class="backdrop only-mobile" onClick=${() => setMenu(false)}></div>`}
    <${Sidebar} section=${section} go=${go} guilds=${guilds} gid=${gid} setGid=${setGid} status=${status} socket=${socket}
      theme=${theme} cycleTheme=${cycleTheme} badges=${badges} open=${menu} close=${() => setMenu(false)} />
    <main class="main" id="main">
      <${Page} ...${props} />
    </main>
    <${Toasts} />
  </div>`;
}

const root = document.getElementById('app');
root.textContent = ''; // drop the static "Loading…"
render(html`<${App} />`, root);

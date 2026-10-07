// Live: the feed of what the bot hears (as it happens), and the stored history.
import { useEffect, useLayoutEffect, useRef, useState } from './vendor/hooks.js';
import { api, ConfirmButton, Empty, html, Icon, load, num, PageHead, S, save, t, Tabs, toast, when } from './lib.js';

const MAX_FEED = 500;
let logKey = 0;

export function feedReducer(items, msg) {
  switch (msg.type) {
    case 'hello': {
      let next = [];
      for (const e of msg.recent || []) next = feedReducer(next, e);
      return next;
    }
    case 'feed':
      return [...items, { ...msg, _key: `f${msg.id}` }].slice(-MAX_FEED);
    case 'log':
      return [...items, { ...msg, _key: `l${++logKey}` }].slice(-MAX_FEED);
    case 'feed_update': {
      for (let i = items.length - 1; i >= 0; i--) {
        if (items[i].type === 'feed' && items[i].id === msg.id) {
          const next = items.slice();
          next[i] = { ...items[i], outcome: msg.outcome ?? items[i].outcome };
          return next;
        }
      }
      return items;
    }
    case 'clear':
      return [];
    default:
      return items;
  }
}

function clock(iso) {
  if (!iso) return '';
  const i = String(iso).indexOf('T');
  return i >= 0 ? String(iso).slice(i + 1, i + 9) : String(iso);
}

export function outcomeTone(outcome) {
  const o = String(outcome || '').toLowerCase();
  if (o.startsWith('played') || o === 'done' || o === 'matched') return 'ok';
  if (o.startsWith('queued') || o.startsWith('playing')) return 'info';
  if (o.startsWith('skip') || o.includes('stale') || o.includes('busy')) return 'warn';
  if (o.includes('error') || o.includes('fail')) return 'bad';
  return 'muted';
}

// ---------------------------------------------------------------- live feed

function FeedRow({ e, showGuild }) {
  if (e.type === 'log') {
    const tone = e.level === 'WARNING' ? 'warn' : e.level === 'INFO' ? 'muted' : 'bad';
    return html`<li class="ev log">
      <time class="tabular">${clock(e.time)}</time>
      <span class="who"><span class="tag ${tone}">${e.level.toLowerCase()}</span></span>
      <span class="text mono">${e.logger ? `${e.logger}: ` : ''}${e.message}</span>
      <span class="meta"></span>
    </li>`;
  }
  const m = e.matched;
  return html`<li class="ev ${m ? 'matched' : ''}">
    <time class="tabular" title=${e.time}>${clock(e.time)}</time>
    <span class="who" title=${e.guild || ''}>${e.user || '?'}${showGuild && e.guild && html`<span class="where">${e.guild}</span>`}</span>
    <span class="text">${e.text ?? html`<span class="muted not-saved">${t('live.notSaved')}</span>`}</span>
    <span class="meta">
      ${m && html`<span class="tag accent" title=${t('live.matched')}>${m.name}</span>`}
      ${e.voice && html`<span class="tag" title=${t('live.voice')}>${e.voice}</span>`}
      ${e.outcome && html`<span class="tag ${outcomeTone(e.outcome)}">${e.outcome}</span>`}
    </span>
  </li>`;
}

function Feed({ feed, dispatch, gid, guilds }) {
  const [scope, setScope] = useState(() => load('heckler.feed.scope', 'here'));
  const [matchesOnly, setMatchesOnly] = useState(() => load('heckler.feed.matches', false));
  const [showLogs, setShowLogs] = useState(() => load('heckler.feed.logs', false));
  const [hover, setHover] = useState(false);
  const [away, setAway] = useState(false);
  const [unseen, setUnseen] = useState(0);
  const list = useRef(null);
  const lastCount = useRef(0);
  useEffect(() => save('heckler.feed.scope', scope), [scope]);
  useEffect(() => save('heckler.feed.matches', matchesOnly), [matchesOnly]);
  useEffect(() => save('heckler.feed.logs', showLogs), [showLogs]);

  const visible = feed.filter((e) => {
    if (e.type === 'log') return showLogs;
    if (e.type !== 'feed') return false;
    if (scope === 'here' && S(e.guild_id) !== gid) return false;
    if (matchesOnly && !e.matched) return false;
    return true;
  });
  const paused = hover || away;
  const toBottom = () => {
    const el = list.current;
    if (el) el.scrollTop = el.scrollHeight;
    setUnseen(0);
  };
  useLayoutEffect(() => {
    const added = visible.length - lastCount.current;
    lastCount.current = visible.length;
    if (!paused) toBottom();
    else if (added > 0) setUnseen((n) => n + added);
  }, [visible.length, feed]);
  const onScroll = () => {
    const el = list.current;
    const isAway = el.scrollHeight - el.scrollTop - el.clientHeight > 40;
    setAway(isAway);
    if (!isAway) setUnseen(0);
  };

  return html`<section class="card feed" aria-label=${t('live.feed')}>
    <div class="toolbar">
      <div class="seg" role="group" aria-label=${t('live.scope')}>
        <button class=${scope === 'here' ? 'on' : ''} aria-pressed=${scope === 'here'} onClick=${() => setScope('here')}>${t('live.thisServer')}</button>
        <button class=${scope === 'all' ? 'on' : ''} aria-pressed=${scope === 'all'} onClick=${() => setScope('all')}>${t('live.allServers')}</button>
      </div>
      <label class="check"><input type="checkbox" checked=${matchesOnly} onChange=${(e) => setMatchesOnly(e.target.checked)} />${t('live.matchesOnly')}</label>
      <label class="check"><input type="checkbox" checked=${showLogs} onChange=${(e) => setShowLogs(e.target.checked)} />${t('live.logLines')}</label>
      <span class="spacer"></span>
      <span class="muted small">${paused ? t('live.paused') : t('live.shown', { n: visible.length })}</span>
      <button class="btn ghost small" onClick=${() => dispatch({ type: 'clear' })}>${t('live.clear')}</button>
    </div>
    <div class="feed-wrap">
      <ol class="feed-list" ref=${list} onScroll=${onScroll} tabindex="0" aria-live="off"
        onMouseEnter=${() => setHover(true)} onMouseLeave=${() => { setHover(false); if (!away) toBottom(); }}>
        ${visible.length === 0 && html`<li class="feed-empty"><${Empty} title=${t('live.emptyTitle')} body=${t('live.emptyBody')} /></li>`}
        ${visible.map((e) => html`<${FeedRow} key=${e._key} e=${e} showGuild=${scope === 'all' && guilds.length > 1} />`)}
      </ol>
      ${paused && unseen > 0 && html`<button class="jump" onClick=${() => { setAway(false); toBottom(); }}>
        ${t('live.newItems', { n: unseen })}</button>`}
    </div>
  </section>`;
}

// ---------------------------------------------------------------- history

function History({ gid }) {
  const [filters, setFilters] = useState({ user_id: '', outcome: '', q: '' });
  const [search, setSearch] = useState('');
  const [events, setEvents] = useState([]);
  const [next, setNext] = useState(null);
  const [meta, setMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const seq = useRef(0);

  const url = (before) => {
    const p = new URLSearchParams({ limit: '50' });
    for (const [k, v] of Object.entries(filters)) if (v) p.set(k, v);
    if (before) p.set('before_id', before);
    return `/api/g/${gid}/history?${p}`;
  };
  const first = async () => {
    const mine = ++seq.current;
    setLoading(true);
    try {
      const out = await api('GET', url());
      if (mine !== seq.current) return;
      setEvents(out.events);
      setNext(out.next_before_id);
      setMeta({ people: out.people, outcomes: out.outcomes });
    } catch (err) {
      toast(err.message, true);
    } finally {
      if (mine === seq.current) setLoading(false);
    }
  };
  const older = async () => {
    setLoading(true);
    try {
      const out = await api('GET', url(next));
      setEvents((e) => [...e, ...out.events]);
      setNext(out.next_before_id);
    } catch (err) { toast(err.message, true); }
    setLoading(false);
  };
  useEffect(() => { first(); }, [gid, filters]);
  useEffect(() => {
    const timer = setTimeout(() => setFilters((f) => (f.q === search.trim() ? f : { ...f, q: search.trim() })), 350);
    return () => clearTimeout(timer);
  }, [search]);

  const person = meta?.people.find((p) => S(p.user_id) === filters.user_id);
  const forget = async () => {
    try {
      const out = await api('DELETE', `/api/g/${gid}/history/${filters.user_id}`);
      toast(t('live.historyDeleted', { n: num(out.deleted), name: person ? person.name : '' }));
      setFilters({ ...filters, user_id: '' });
    } catch (err) { toast(err.message, true); }
  };

  return html`<section class="card" aria-label=${t('live.history')}>
    <div class="toolbar">
      <label class="visually-hidden" for="hist-person">${t('live.person')}</label>
      <select id="hist-person" value=${filters.user_id} onChange=${(e) => setFilters({ ...filters, user_id: e.target.value })}>
        <option value="">${t('live.everyone')}</option>
        ${(meta?.people || []).map((p) => html`<option key=${S(p.user_id)} value=${S(p.user_id)}>${p.name} (${num(p.count)})</option>`)}
      </select>
      <label class="visually-hidden" for="hist-outcome">${t('live.outcome')}</label>
      <select id="hist-outcome" value=${filters.outcome} onChange=${(e) => setFilters({ ...filters, outcome: e.target.value })}>
        <option value="">${t('live.anyOutcome')}</option>
        ${(meta?.outcomes || []).map((o) => html`<option key=${o} value=${o}>${o}</option>`)}
      </select>
      <div class="search grow">
        <${Icon} name="search" size=${16} />
        <input type="search" id="hist-search" placeholder=${t('live.search')} value=${search}
          onInput=${(e) => setSearch(e.target.value)} aria-label=${t('live.search')} />
      </div>
      <button class="btn ghost small" onClick=${first} aria-label=${t('common.refresh')}><${Icon} name="refresh" size=${16} /></button>
    </div>
    ${person && html`<div class="notice">
      <span class="grow">${t('live.personEntries', { n: num(person.count), name: person.name })}</span>
      <${ConfirmButton} label=${t('live.deleteHistory', { name: person.name })} confirmLabel=${t('live.deleteHistoryConfirm')}
        cls="small" icon="trash" onConfirm=${forget} />
    </div>`}
    ${events.length === 0 && !loading && html`<${Empty} title=${t('live.historyEmptyTitle')} body=${t('live.historyEmptyBody')} />`}
    <ol class="history">${events.map((e) => html`<li key=${e.id} class="hrow">
      <time class="muted small tabular" title=${e.time}>${when(e.time)}</time>
      <span class="who">${e.user}</span>
      <span class="text">${e.text ?? html`<span class="muted not-saved">${t('live.notSaved')}</span>`}</span>
      <span class="meta">
        ${e.reaction && html`<span class="tag accent">${e.reaction}</span>`}
        <span class="tag ${outcomeTone(e.outcome)}">${e.outcome}</span>
      </span>
    </li>`)}</ol>
    ${next && html`<div class="center"><button class="btn" disabled=${loading} onClick=${older}>
      ${loading ? t('common.loading') : t('live.older')}</button></div>`}
  </section>`;
}

export function Live({ feed, dispatch, gid, guilds, missing }) {
  const [tab, setTab] = useState(() => load('heckler.live.tab', 'feed'));
  useEffect(() => save('heckler.live.tab', tab), [tab]);
  return html`<div class="page page-live">
    <${PageHead} section="live" />
    <${Tabs} label=${t('page.live.title')} value=${tab} onChange=${setTab}
      items=${[['feed', t('live.feed')], ['history', t('live.history')]]} />
    ${tab === 'feed' ? html`<${Feed} feed=${feed} dispatch=${dispatch} gid=${gid} guilds=${guilds} />`
      : !gid || missing ? html`<${Empty} title=${t('common.noStoreTitle')} body=${t('common.noStoreBody')} />`
        : html`<${History} gid=${gid} />`}
  </div>`;
}

// Stats: what the bot heard and did over a time range. Backend: web/routes_stats.py.
import { useEffect, useRef, useState } from './vendor/hooks.js';
import { api, ConfirmButton, Empty, html, Icon, load, num, PageHead, S, save, t, useRunner, Waiting, when } from './lib.js';

const RANGES = ['24h', '7d', '30d', '90d'];
const pct = (part, whole) => (whole ? `${Math.round((100 * part) / whole)}%` : '—');

function niceMax(v) {
  if (v <= 4) return 4;
  const step = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * step >= v) return m * step;
  return 10 * step;
}

// One series: sentences heard per hour of the day. Inline SVG drawn at its real
// width so the text keeps its CSS size; a tooltip per hour and a table view.
function HourChart({ hours }) {
  const [hover, setHover] = useState(null);
  const [W, setW] = useState(720);
  const box = useRef(null);
  useEffect(() => {
    const ro = new ResizeObserver(() => box.current && setW(Math.max(280, box.current.clientWidth)));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);
  const H = 200, left = 36, right = 8, top = 14, bottom = 26;
  const max = niceMax(Math.max(1, ...hours.map((h) => h.heard)));
  const band = (W - left - right) / 24;
  const barW = Math.max(2, Math.min(24, band - 3));
  const y = (v) => top + (H - top - bottom) * (1 - v / max);
  const ticks = [0, max / 2, max];
  const bar = (h) => {
    const x = left + h.hour * band + (band - barW) / 2;
    const y0 = y(h.heard), base = y(0);
    if (base - y0 <= 0) return '';
    const r = Math.min(4, base - y0, barW / 2);
    return `M${x},${base} V${y0 + r} Q${x},${y0} ${x + r},${y0} H${x + barW - r} Q${x + barW},${y0} ${x + barW},${y0 + r} V${base} Z`;
  };
  const hh = (n) => String(n).padStart(2, '0');
  const cur = hover !== null ? hours[hover] : null;
  const busiest = hours.reduce((a, b) => (b.heard > a.heard ? b : a), hours[0]);
  return html`<div class="chart" ref=${box}>
    <svg viewBox=${`0 0 ${W} ${H}`} width=${W} height=${H} role="img"
      aria-label=${t('stats.chartLabel', { hour: hh(busiest.hour), n: num(busiest.heard) })}>
      ${ticks.map((v) => html`<g key=${v}>
        <line class="grid" x1=${left} x2=${W - right} y1=${y(v)} y2=${y(v)} />
        <text class="axis" x=${left - 6} y=${y(v) + 4} text-anchor="end">${num(v)}</text></g>`)}
      ${hours.map((h) => html`<path key=${h.hour} class="bar ${hover === h.hour ? 'on' : ''}" d=${bar(h)} />`)}
      ${hours.filter((h) => h.hour % 3 === 0).map((h) => html`<text key=${h.hour} class="axis"
        x=${left + h.hour * band + band / 2} y=${H - 8} text-anchor="middle">${hh(h.hour)}</text>`)}
      ${hours.map((h) => html`<rect key=${`hit${h.hour}`} class="hit" x=${left + h.hour * band} y=${top} width=${band}
        height=${H - top - bottom} onMouseEnter=${() => setHover(h.hour)} onMouseLeave=${() => setHover(null)} />`)}
    </svg>
    <div class="chart-tip ${cur ? 'show' : ''}" style=${{ left: cur ? `${((left + (cur.hour + 0.5) * band) / W) * 100}%` : '0' }}>
      ${cur && html`<b>${hh(cur.hour)}:00–${hh((cur.hour + 1) % 24)}:00</b>
        <div>${t('stats.tip', { heard: num(cur.heard), matched: num(cur.matched) })}</div>`}
    </div>
    <details class="table-view"><summary>${t('stats.asTable')}</summary>
      <div class="table-wrap"><table class="table compact"><thead><tr><th>${t('stats.hour')}</th><th class="num-col">${t('stats.heard')}</th>
        <th class="num-col">${t('stats.matched')}</th></tr></thead>
      <tbody>${hours.map((h) => html`<tr key=${h.hour}><td>${hh(h.hour)}:00</td><td class="num-col">${num(h.heard)}</td>
        <td class="num-col">${num(h.matched)}</td></tr>`)}</tbody></table></div>
    </details>
  </div>`;
}

function Tile({ label, value, sub, tone }) {
  return html`<div class="tile ${tone || ''}"><span class="tile-label">${label}</span><span class="tile-value tabular">${value}</span>
    ${sub && html`<span class="tile-sub">${sub}</span>`}</div>`;
}

export function Stats({ gid, guild, missing, reload }) {
  const [range, setRange] = useState(() => load('heckler.stats.range', '7d'));
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [run, busy] = useRunner();
  useEffect(() => save('heckler.stats.range', range), [range]);
  const fetchStats = async () => {
    if (!gid) return;
    try { setData(await api('GET', `/api/g/${gid}/stats?range=${range}`)); setError(null); } catch (err) { setError(err); }
  };
  useEffect(() => { setData(null); fetchStats(); }, [gid, range]);
  if (!gid || missing || error?.status === 404) return html`<${Waiting} section="stats" missing=${true} guild=${guild} />`;

  const act = (fn, okText) => run(async () => { await fn(); await fetchStats(); await reload(); }, okText);
  const head = html`<${PageHead} section="stats">
    <div class="seg" role="group" aria-label=${t('stats.range')}>
      ${RANGES.map((k) => html`<button key=${k} class=${range === k ? 'on' : ''} aria-pressed=${range === k} onClick=${() => setRange(k)}>${t(`stats.ranges.${k}`)}</button>`)}
    </div>
    <button class="btn icon ghost" onClick=${fetchStats} aria-label=${t('common.refresh')} title=${t('common.refresh')}><${Icon} name="refresh" size=${18} /></button>
  <//>`;
  if (error) return html`<div class="page">${head}<p class="notice bad">${error.message}</p></div>`;
  if (!data) return html`<div class="page">${head}<p class="muted loading">${t('common.loading')}</p></div>`;
  const tt = data.totals;
  const skipped = Object.values(tt.skipped).reduce((a, b) => a + b, 0);
  const skippedSub = Object.entries(tt.skipped).map(([k, v]) => `${t(`stats.skip.${k.replace(/^skipped \(|\)$/g, '')}`)} ${num(v)}`).join(' · ');

  return html`<div class="page">
    ${head}
    <div class="tiles">
      <${Tile} label=${t('stats.heard')} value=${num(tt.heard)} />
      <${Tile} label=${t('stats.matched')} value=${num(tt.matched)} sub=${t('stats.ofHeard', { pct: pct(tt.matched, tt.heard) })} />
      <${Tile} label=${t('stats.skipped')} value=${num(skipped)} sub=${skippedSub || t('stats.none')} />
      <${Tile} label=${t('stats.noMatch')} value=${num(tt.no_match)} />
      <${Tile} label=${t('stats.echo')} value=${num(tt.echo)} sub=${t('stats.echoSub')} />
    </div>
    <section class="card">
      <div class="card-head"><div><h2>${t('stats.byHour')}</h2><p class="sub">${t('stats.byHourDesc')}</p></div></div>
      ${tt.heard ? html`<${HourChart} hours=${data.hours} />` : html`<${Empty} title=${t('stats.nothingHeard')} />`}
    </section>
    <div class="two-col">
      <section class="card">
        <h2>${t('stats.top')}</h2>
        ${data.top_reactions.length === 0 ? html`<${Empty} title=${t('stats.nothingFired')} />` : html`<div class="table-wrap"><table class="table">
          <thead><tr><th>${t('stats.reaction')}</th><th class="num-col">${t('stats.fired')}</th><th class="num-col">${t('stats.allTime')}</th></tr></thead>
          <tbody>${data.top_reactions.map((r) => html`<tr key=${r.reaction_id}>
            <td>${r.name}${r.kind && html` <span class="muted small">${t(`kinds.${r.kind}.one`)}</span>`}</td>
            <td class="num-col"><b>${num(r.fires)}</b></td><td class="num-col">${r.deleted ? '—' : num(r.uses)}</td></tr>`)}</tbody></table></div>`}
      </section>
      <section class="card">
        <h2>${t('stats.people')}</h2>
        ${data.people.length === 0 ? html`<${Empty} title=${t('stats.nobody')} />` : html`<div class="table-wrap"><table class="table">
          <thead><tr><th>${t('stats.person')}</th><th class="num-col">${t('stats.sentences')}</th><th class="num-col">${t('stats.gags')}</th><th>${t('stats.favorite')}</th></tr></thead>
          <tbody>${data.people.map((p) => html`<tr key=${S(p.user_id)}>
            <td>${p.name}</td><td class="num-col">${num(p.heard)}</td><td class="num-col">${num(p.gags)}</td>
            <td>${p.top_gag ? html`${p.top_gag.name} <span class="muted small">×${p.top_gag.count}</span>` : html`<span class="muted">—</span>`}</td>
          </tr>`)}</tbody></table></div>`}
      </section>
    </div>
    ${data.skipped_reactions.length > 0 && html`<section class="card">
      <div class="card-head"><div><h2>${t('stats.often')}</h2><p class="sub">${t('stats.oftenDesc')}</p></div></div>
      <div class="table-wrap"><table class="table"><thead><tr><th>${t('stats.reaction')}</th><th class="num-col">${t('stats.fired')}</th>
        <th class="num-col">${t('stats.skip.cooldown')}</th><th class="num-col">${t('stats.skip.chance')}</th><th class="num-col">${t('stats.skippedShare')}</th></tr></thead>
      <tbody>${data.skipped_reactions.map((r) => html`<tr key=${r.reaction_id}>
        <td>${r.name}</td><td class="num-col">${num(r.fires)}</td>
        <td class="num-col">${num(r.skips['skipped (cooldown)'])}</td><td class="num-col">${num(r.skips['skipped (chance)'])}</td>
        <td class="num-col ${r.skip_share >= 0.5 ? 'warn-text' : ''}">${Math.round(r.skip_share * 100)}%</td></tr>`)}</tbody></table></div>
    </section>`}
    <section class="card">
      <div class="card-head"><div><h2>${t('stats.never')}</h2><p class="sub">${t('stats.neverDesc')}</p></div></div>
      ${data.never_fired.length === 0 ? html`<${Empty} title=${t('stats.allFired')} />` : html`<div class="table-wrap"><table class="table">
        <thead><tr><th>${t('stats.reaction')}</th><th class="num-col">${t('stats.allTime')}</th><th>${t('stats.lastUsed')}</th><th>${t('stats.created')}</th>
          <th><span class="visually-hidden">${t('common.actions')}</span></th></tr></thead>
        <tbody>${data.never_fired.map((r) => html`<tr key=${r.reaction_id}>
          <td>${r.name} <span class="muted small">${t(`kinds.${r.kind}.one`)}</span></td><td class="num-col">${num(r.uses)}</td>
          <td class="muted small">${when(r.last_used_at) || t('stats.neverUsed')}</td><td class="muted small">${when(r.created_at)}</td>
          <td><div class="row end">
            <button class="btn small" disabled=${busy} onClick=${() => act(() => api('PATCH', `/api/g/${gid}/reactions/${r.reaction_id}`, { enabled: false }), t('gags.disabled', { name: r.name }))}>${t('stats.disable')}</button>
            <${ConfirmButton} label=${t('common.delete')} confirmLabel=${t('gags.deleteConfirm')} cls="small" disabled=${busy}
              onConfirm=${() => act(() => api('DELETE', `/api/g/${gid}/reactions/${r.reaction_id}`), t('gags.deleted', { name: r.name }))} />
          </div></td></tr>`)}</tbody></table></div>`}
    </section>
  </div>`;
}

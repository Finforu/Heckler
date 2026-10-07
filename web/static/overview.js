// Overview: first-run checklist, the bot's health, and the selected server's
// call controls and soundboard.
import { useState } from './vendor/hooks.js';
import { ConfirmButton, html, Icon, load, PageHead, Pill, S, save, t, Toggle } from './lib.js';
import { Soundboard } from './media.js';

function duration(s) {
  s = Math.max(0, Math.floor(s || 0));
  const d = Math.floor(s / 86400), hh = Math.floor(s / 3600) % 24, mm = Math.floor(s / 60) % 60;
  if (d) return `${d}d ${hh}h`;
  if (hh) return `${hh}h ${mm}m`;
  if (mm) return `${mm}m ${s % 60}s`;
  return `${s}s`;
}

export const privacyKey = (gid) => `heckler.privacyReviewed.${gid}`;

// ---------------------------------------------------------------- setup checklist

function checklist({ status, guilds, gid, data }) {
  const botVoiceId = data?.settings?.global?.['voice.bot'];
  const botVoice = data?.voices?.find((v) => S(v.id) === S(botVoiceId));
  const own = data?.settings?.own || {};
  const global = data?.settings?.global || {};
  return [
    { key: 'token', done: true },
    { key: 'connected', done: !!status?.bot?.connected },
    { key: 'server', done: guilds.length > 0 },
    { key: 'voice', done: botVoice?.status === 'ready', go: 'voices', wait: !data },
    { key: 'language', done: !!(own.language || global.language), go: 'settings', wait: !data },
    { key: 'privacy', done: !!load(privacyKey(gid), false), go: 'settings', wait: !gid },
  ];
}

function Setup({ items, go }) {
  const todo = items.filter((i) => !i.done && !i.wait).length;
  const [open, setOpen] = useState(todo > 0);
  if (!todo && !open) {
    return html`<div class="setup-done">
      <${Icon} name="check" size=${18} /><span>${t('setup.allDone')}</span>
      <button class="btn link" onClick=${() => setOpen(true)}>${t('setup.show')}</button>
    </div>`;
  }
  const done = items.filter((i) => i.done).length;
  return html`<section class="card setup" aria-labelledby="setup-title">
    <div class="card-head">
      <div>
        <h2 id="setup-title">${t('setup.title')}</h2>
        <p class="sub">${t('setup.progress', { done, total: items.length })}</p>
      </div>
      <div class="progress" aria-hidden="true"><span style=${{ width: `${(100 * done) / items.length}%` }}></span></div>
      ${!todo && html`<button class="btn ghost small" onClick=${() => setOpen(false)}>${t('setup.hide')}</button>`}
    </div>
    <ol class="checklist">
      ${items.map((i) => html`<li key=${i.key} class=${i.done ? 'done' : ''}>
        <span class="check-mark" aria-label=${i.done ? t('setup.doneLabel') : t('setup.todoLabel')}>${i.done && html`<${Icon} name="check" size=${14} />`}</span>
        <div class="check-text">
          <span class="check-title">${t(`setup.${i.key}.title`)}</span>
          <span class="hint">${t(`setup.${i.key}.${i.done ? 'done' : 'todo'}`)}</span>
        </div>
        ${!i.done && i.go && html`<button class="btn small" onClick=${() => go(i.go)}>${t(`setup.${i.key}.action`)}
          <${Icon} name="chevron" size=${14} /></button>`}
      </li>`)}
    </ol>
  </section>`;
}

// ---------------------------------------------------------------- bot health

function Health({ status, control, busy }) {
  if (!status) return html`<section class="card"><h2>${t('overview.bot')}</h2><p class="muted">${t('common.loading')}</p></section>`;
  const { bot, models = [], gpu, tts_queue: q } = status;
  const pct = gpu && gpu.total_gb ? Math.min(100, (100 * gpu.used_gb) / gpu.total_gb) : 0;
  const level = pct > 92 ? 'bad' : pct > 80 ? 'warn' : '';
  const items = q?.items || [];
  return html`<section class="card health" aria-labelledby="health-title">
    <div class="card-head">
      <div>
        <h2 id="health-title">${bot.name}</h2>
        <p class="sub">${bot.user || t('overview.notLoggedIn')}</p>
      </div>
      <${Pill} tone=${bot.connected ? 'ok' : 'bad'}>${bot.connected ? t('overview.online') : t('overview.offline')}<//>
    </div>
    <dl class="facts">
      <div><dt>${t('overview.uptime')}</dt><dd>${duration(bot.uptime_s)}</dd></div>
      <div><dt>${t('overview.speechToText')}</dt><dd>${bot.stt_engine}</dd></div>
    </dl>
    <h3>${t('overview.gpu')}</h3>
    ${gpu ? html`<div class="meter-label"><span>${gpu.name}</span>
        <span class="tabular">${Number(gpu.used_gb).toFixed(1)} / ${Number(gpu.total_gb).toFixed(1)} GB</span></div>
      <div class="meter" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${Math.round(pct)} aria-label=${t('overview.gpu')}>
        <span class=${level} style=${{ width: `${pct}%` }}></span></div>` : html`<p class="muted small">${t('overview.noGpu')}</p>`}
    <h3>${t('overview.models')}</h3>
    <ul class="models">
      ${models.map((m) => html`<li key=${m.role + m.name}>
        <span class="dot ${m.loaded ? 'ok' : ''}" title=${m.loaded ? t('overview.loaded') : t('overview.notLoaded')}></span>
        <span class="model-name" title=${m.name}>${m.name}</span>
        <span class="tag">${m.role}</span><span class="tag muted">${m.device}</span>
      </li>`)}
    </ul>
    <h3>${t('overview.queue', { n: q?.pending ?? 0 })}</h3>
    ${!q?.running && !items.length ? html`<p class="muted small">${t('overview.queueIdle')}</p>` : html`<ul class="queue">
      ${q?.running && html`<li><span class="tag accent">${q.running.kind}</span><span class="queue-text">${q.running.text}</span>
        <span class="muted small">${t('overview.running')}</span></li>`}
      ${items.slice(0, 5).map((it, i) => html`<li key=${i}><span class="tag">${it.kind}</span>
        <span class="queue-text" title=${it.text}>${it.text}</span><span class="muted small">${it.voice}</span></li>`)}
      ${items.length > 5 && html`<li class="muted small">${t('overview.queueMore', { n: items.length - 5 })}</li>`}
    </ul>`}
    <div class="card-actions">
      <button class="btn" disabled=${busy.reload} onClick=${() => control('reload')}>${t('overview.reload')}</button>
      <${ConfirmButton} label=${t('overview.restart')} confirmLabel=${t('overview.restartConfirm')} disabled=${busy.restart}
        onConfirm=${() => control('restart')} />
    </div>
  </section>`;
}

// ---------------------------------------------------------------- the selected server

function bestChannel(g) {
  if (g.voice) return g.voice.channel_id;
  const chans = g.voice_channels || [];
  if (!chans.length) return '';
  return chans.reduce((a, b) => (b.people > a.people ? b : a)).id;
}

function ServerPanel({ g, control, busy, changed, go }) {
  const [channel, setChannel] = useState(() => bestChannel(g));
  const chans = g.voice_channels || [];
  const selected = chans.some((c) => c.id === channel) ? channel : bestChannel(g);
  const b = (action) => busy[`${action}:${g.id}`];
  const inCall = !!g.voice;
  const toggles = g.toggles || {};
  const set = (name, value) => control('toggle', { guild_id: g.id, name, value });

  return html`<section class="card server" aria-labelledby="server-title">
    <div class="card-head">
      <div>
        <h2 id="server-title">${g.name}</h2>
        <p class="sub">${inCall ? t('overview.inCall', { channel: g.voice.channel, n: g.voice.people })
          : t('overview.notInCall')}</p>
      </div>
      <${Pill} tone=${inCall ? 'ok' : 'muted'}>${inCall ? t('overview.connected') : t('overview.idle')}<//>
    </div>

    <div class="call-row">
      <label for="channel-pick" class="visually-hidden">${t('overview.voiceChannel')}</label>
      <select id="channel-pick" value=${selected} onChange=${(e) => setChannel(e.target.value)}>
        ${chans.length === 0 && html`<option value="">${t('overview.noChannels')}</option>`}
        ${chans.map((c) => html`<option key=${c.id} value=${c.id}>${t('overview.channelOption', { name: c.name, n: c.people })}</option>`)}
      </select>
      <button class="btn primary" disabled=${!selected || b('join') || (inCall && g.voice.channel_id === selected)}
        onClick=${() => control('join', { guild_id: g.id, channel_id: selected })}>${inCall ? t('overview.move') : t('overview.join')}</button>
      <button class="btn" disabled=${!inCall || b('leave')} onClick=${() => control('leave', { guild_id: g.id })}>${t('overview.leave')}</button>
    </div>
    <div class="call-row">
      <button class="btn" disabled=${b('stop')} onClick=${() => control('stop', { guild_id: g.id })}>
        <${Icon} name="stop" size=${14} />${t('overview.stop')}</button>
      <button class="btn" disabled=${b('clear_queue') || !g.reply_queue} onClick=${() => control('clear_queue', { guild_id: g.id })}>
        ${t('overview.clearQueue')}</button>
      <span class="muted small">${t('overview.queueCounts', { replies: g.reply_queue, timers: g.timers })}</span>
    </div>

    <div class="toggles">
      <${Toggle} label=${t('overview.autojoin')} hint=${t('overview.autojoinHint')} checked=${toggles.autojoin}
        disabled=${b('toggle')} onChange=${(v) => set('autojoin', v)} />
      <${Toggle} label=${t('overview.userclone')} hint=${t('overview.userclonehint')} checked=${toggles.userclone}
        disabled=${b('toggle')} onChange=${(v) => set('userclone', v)} />
      <${Toggle} label=${t('overview.record')} hint=${toggles.record ? t('overview.recordOn') : t('overview.recordHint')}
        checked=${toggles.record} disabled=${b('toggle') || !toggles.record} onChange=${(v) => set('record', v)} />
      ${'transcripts' in toggles && html`<div class="toggle-row readonly">
        <${Pill} tone=${toggles.transcripts ? 'info' : 'muted'}>${toggles.transcripts ? t('overview.transcriptsOn') : t('overview.transcriptsOff')}<//>
        <button class="btn link" onClick=${() => go('settings')}>${t('overview.transcriptsChange')}</button>
      </div>`}
    </div>

    <h3>${t('overview.soundboard')}</h3>
    <${Soundboard} gid=${g.id} changed=${changed} control=${control} busy=${busy} inCall=${inCall} go=${go} />
  </section>`;
}

function OtherServers({ guilds, gid, setGid }) {
  const others = guilds.filter((g) => S(g.id) !== gid);
  if (!others.length) return null;
  return html`<section class="card" aria-labelledby="others-title">
    <h2 id="others-title">${t('overview.otherServers')}</h2>
    <ul class="server-list">
      ${others.map((g) => html`<li key=${g.id}>
        <button class="server-row" onClick=${() => setGid(S(g.id))}>
          <span class="dot ${g.voice ? 'ok' : ''}"></span>
          <span class="server-name">${g.name}</span>
          <span class="muted small">${g.voice ? t('overview.inCall', { channel: g.voice.channel, n: g.voice.people }) : t('overview.notInCall')}</span>
          <${Icon} name="chevron" size=${16} />
        </button>
      </li>`)}
    </ul>
  </section>`;
}

export function Overview({ status, guilds, gid, guild, setGid, data, control, busy, changed, go }) {
  const items = checklist({ status, guilds, gid, data });
  return html`<div class="page">
    <${PageHead} section="overview" />
    <${Setup} items=${items} go=${go} />
    <div class="overview-grid">
      <div class="col-main">
        ${guild ? html`<${ServerPanel} key=${guild.id} g=${guild} control=${control} busy=${busy} changed=${changed} go=${go} />`
          : html`<section class="card"><h2>${t('overview.noServerTitle')}</h2><p class="muted">${t('overview.noServerBody')}</p></section>`}
        <${OtherServers} guilds=${guilds} gid=${gid} setGid=${setGid} />
      </div>
      <div class="col-side">
        <${Health} status=${status} control=${control} busy=${busy} />
      </div>
    </div>
  </div>`;
}

// Voices and Sounds pages, the waveform editor and the soundboard on the
// Overview page. Backend: web/routes_media.py.
import { useEffect, useRef, useState } from './vendor/hooks.js';
import {
  api, ConfirmButton, Empty, Field, html, Icon, mb, PageHead, S, secs, t, Tabs, toast, useId, useRunner, Waiting,
} from './lib.js';

function statusTone(v) {
  if (v.status === 'ready') return 'ok';
  if (v.status === 'failed') return 'bad';
  if (v.status === 'queued' || v.status === 'building') return 'info';
  return 'warn';
}

export function statusLabel(v) {
  if (v.status === 'ready') return t('voices.status.ready');
  if (v.status === 'queued') return t('voices.status.queued');
  if (v.status === 'building') return t('voices.status.building');
  if (v.status === 'failed') return t('voices.status.failed');
  return v.has_prompt ? t('voices.status.rebuild') : t('voices.status.notBuilt');
}

// ---------------------------------------------------------------- waveform

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}

// Redraw when the theme changes (explicit choice or the system's).
function useThemeTick() {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const bump = () => setTick((n) => n + 1);
    const mo = new MutationObserver(bump);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener('change', bump);
    return () => { mo.disconnect(); mq.removeEventListener('change', bump); };
  }, []);
  return tick;
}

// peaks: 0..1 per bucket. selection: [start_s, end_s] or null, dragged with onSelect.
// segments: [[s, e], ...] shaded (speech found automatically). audioRef: <audio> for the playhead.
export function Waveform({ peaks, duration, segments = [], selection = null, onSelect, audioRef, height = 88, label }) {
  const canvas = useRef(null);
  const drag = useRef(null);
  const [playhead, setPlayhead] = useState(null);
  const tick = useThemeTick();

  useEffect(() => {
    const el = audioRef?.current;
    if (!el) return undefined;
    const update = () => setPlayhead(el.paused && el.currentTime === 0 ? null : el.currentTime);
    el.addEventListener('timeupdate', update);
    el.addEventListener('seeked', update);
    return () => { el.removeEventListener('timeupdate', update); el.removeEventListener('seeked', update); };
  }, [audioRef?.current]);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return undefined;
    const draw = () => {
      const width = c.clientWidth || 600;
      const dpr = window.devicePixelRatio || 1;
      c.width = Math.round(width * dpr);
      c.height = Math.round(height * dpr);
      const g = c.getContext('2d');
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      const x = (s) => (duration ? (s / duration) * width : 0);
      g.fillStyle = cssVar('--sunken');
      g.fillRect(0, 0, width, height);
      g.globalAlpha = 0.22;
      g.fillStyle = cssVar('--ok');
      for (const [s, e] of segments) g.fillRect(x(s), 0, Math.max(1, x(e) - x(s)), height);
      g.globalAlpha = 1;
      if (selection) {
        g.fillStyle = cssVar('--accent');
        g.globalAlpha = 0.16;
        g.fillRect(x(selection[0]), 0, x(selection[1]) - x(selection[0]), height);
        g.globalAlpha = 1;
      }
      const n = peaks.length || 1;
      const bar = width / n;
      const mid = height / 2;
      const off = cssVar('--wave');
      const on = cssVar('--chart');
      for (let i = 0; i < peaks.length; i++) {
        const at = ((i + 0.5) / n) * duration;
        g.fillStyle = selection && at >= selection[0] && at <= selection[1] ? on : off;
        const hh = Math.max(1, peaks[i] * (height - 10));
        g.fillRect(i * bar, mid - hh / 2, Math.max(1, bar - (bar > 3 ? 1 : 0)), hh);
      }
      if (selection) {
        g.fillStyle = cssVar('--accent');
        for (const s of selection) g.fillRect(x(s) - 1, 0, 2, height);
      }
      if (playhead !== null) {
        g.fillStyle = cssVar('--text');
        g.fillRect(x(playhead), 0, 1.5, height);
      }
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(c);
    return () => ro.disconnect();
  }, [peaks, duration, segments, selection, playhead, height, tick]);

  const timeAt = (e) => {
    const r = canvas.current.getBoundingClientRect();
    return Math.min(duration, Math.max(0, ((e.clientX - r.left) / r.width) * duration));
  };
  const down = (e) => {
    if (!onSelect || !duration) return;
    const at = timeAt(e);
    const r = canvas.current.getBoundingClientRect();
    const near = (edge) => selection && Math.abs(((selection[edge] - at) / duration) * r.width) < 10;
    if (near(0)) drag.current = { edge: 0 };
    else if (near(1)) drag.current = { edge: 1 };
    else { drag.current = { anchor: at }; onSelect(at, at); }
    canvas.current.setPointerCapture(e.pointerId);
  };
  const move = (e) => {
    if (!drag.current) return;
    const at = timeAt(e);
    if (drag.current.anchor !== undefined) onSelect(Math.min(drag.current.anchor, at), Math.max(drag.current.anchor, at));
    else {
      const sel = [...selection];
      sel[drag.current.edge] = at;
      onSelect(Math.min(...sel), Math.max(...sel));
    }
  };
  const up = () => { drag.current = null; };
  const seek = (e) => {
    if (onSelect || !audioRef?.current || !duration) return;
    audioRef.current.currentTime = timeAt(e);
  };
  return html`<canvas class="wave ${onSelect ? 'selectable' : ''}" ref=${canvas} style=${{ height: `${height}px` }}
    onPointerDown=${down} onPointerMove=${move} onPointerUp=${up} onPointerCancel=${up} onClick=${seek}
    role="img" aria-label=${label || t('voices.waveform')}></canvas>`;
}

// ---------------------------------------------------------------- voices

export function Voices({ data, gid, guild, missing, reload }) {
  const [info, setInfo] = useState(null);
  const [noLibrary, setNoLibrary] = useState(false);
  const [open, setOpen] = useState(null);
  const [creating, setCreating] = useState(null);
  const [tab, setTab] = useState('voices');
  const [run, busy] = useRunner();

  const fetchInfo = async () => {
    if (!gid) return;
    try {
      setInfo(await api('GET', `/api/g/${gid}/voices`));
      setNoLibrary(false);
    } catch (err) {
      if (err.status === 404) setNoLibrary(true);
      else toast(err.message, true);
    }
  };
  useEffect(() => { fetchInfo(); }, [gid, data]);
  useEffect(() => { setOpen(null); setCreating(null); }, [gid]);

  if (!data) return html`<${Waiting} section="voices" missing=${missing} guild=${guild} />`;
  if (noLibrary) {
    return html`<div class="page"><${PageHead} section="voices" />
      <${Empty} title=${t('voices.noLibraryTitle')} body=${t('voices.noLibraryBody')} /></div>`;
  }
  if (!info) return html`<${Waiting} section="voices" guild=${guild} />`;
  const refresh = async () => { await fetchInfo(); await reload(); };
  const voice = info.voices.find((v) => S(v.id) === S(open)) || info.speakers.find((v) => S(v.id) === S(open));

  if (voice) {
    return html`<div class="page"><${VoiceDetail} key=${voice.id} v=${voice} info=${info} gid=${gid}
      refresh=${refresh} close=${() => setOpen(null)} /></div>`;
  }

  return html`<div class="page">
    <${PageHead} section="voices">
      <button class="btn" onClick=${() => setCreating('designed')}><${Icon} name="plus" size=${16} />${t('voices.newDesigned')}</button>
      <button class="btn primary" onClick=${() => setCreating('clone')}><${Icon} name="plus" size=${16} />${t('voices.newClone')}</button>
    <//>
    ${creating && html`<${NewVoice} kind=${creating} info=${info} gid=${gid}
      done=${async (id) => { setCreating(null); await refresh(); if (id) setOpen(id); }} cancel=${() => setCreating(null)} />`}
    <${Tabs} label=${t('page.voices.title')} value=${tab} onChange=${setTab} items=${[
      ['voices', t('voices.tabVoices')], ['people', t('voices.tabPeople'), null], ['cache', t('voices.tabCache')]]} />
    ${tab === 'voices' && html`<section class="card">
      ${info.voices.length === 0 ? html`<${Empty} title=${t('voices.emptyTitle')} body=${t('voices.emptyBody')}
          action=${html`<button class="btn primary" onClick=${() => setCreating('clone')}>${t('voices.newClone')}</button>`} />`
        : html`<div class="table-wrap"><table class="table voices">
          <thead><tr><th>${t('voices.colName')}</th><th>${t('voices.colKind')}</th><th>${t('voices.colStatus')}</th>
            <th>${t('voices.colReference')}</th><th>${t('voices.colTranscript')}</th><th><span class="visually-hidden">${t('common.actions')}</span></th></tr></thead>
          <tbody>${info.voices.map((v) => html`<tr key=${v.id}>
            <td><button class="link-title" onClick=${() => setOpen(v.id)}>${v.name}</button>
              ${v.is_bot && html` <span class="tag accent">${t('voices.botVoice')}</span>`}
              ${v.scope === 'global' && !v.is_bot && html` <span class="tag">${t('voices.global')}</span>`}</td>
            <td>${t(`voices.kind.${v.kind}`)}</td>
            <td><span class="tag ${statusTone(v)}" title=${v.error || ''}>${statusLabel(v)}</span></td>
            <td class="tabular">${v.kind === 'designed' ? html`<span class="muted small">${v.instruct}</span>` : secs(v.ref_seconds)}</td>
            <td class="ellipsis" title=${v.ref_text || ''}>${v.ref_text || html`<span class="muted">—</span>`}</td>
            <td><button class="btn small" onClick=${() => setOpen(v.id)}>${t('common.open')}</button></td>
          </tr>`)}</tbody>
        </table></div>`}
    </section>`}
    ${tab === 'people' && html`<${Speakers} info=${info} gid=${gid} refresh=${refresh} open=${setOpen} />`}
    ${tab === 'cache' && html`<section class="card">
      <div class="card-head"><div><h2>${t('voices.cacheTitle')}</h2><p class="sub">${t('voices.cacheDesc')}</p></div></div>
      <div class="meter-label"><span>${t('voices.cacheUsed', { used: mb(info.cache.bytes) })}</span><span>${t('voices.cacheLimit', { limit: mb(info.cache.limit) })}</span></div>
      <div class="meter"><span style=${{ width: `${Math.min(100, (100 * info.cache.bytes) / (info.cache.limit || 1))}%` }}></span></div>
      <div class="card-actions">
        <${ConfirmButton} label=${t('voices.cacheClear')} confirmLabel=${t('voices.cacheClearConfirm')} disabled=${busy}
          onConfirm=${() => run(async () => { await api('POST', `/api/g/${gid}/voices/cache/clear`, {}); await fetchInfo(); }, t('voices.cacheCleared'))} />
      </div>
    </section>`}
  </div>`;
}

function InstructComposer({ design, value, onChange, id }) {
  const items = value.split(/\s*[,，]\s*/).map((x) => x.trim().toLowerCase()).filter(Boolean);
  const toggle = (cat, item) => {
    const others = items.filter((x) => !cat.values.includes(x));
    onChange((items.includes(item) ? others : [...others, item]).join(', '));
  };
  return html`<div class="composer">
    ${design.map((cat) => html`<div key=${cat.category} class="composer-row" role="group" aria-label=${t(`design.${cat.category}`)}>
      <span class="eyebrow">${t(`design.${cat.category}`)}</span>
      <div class="composer-chips">${cat.values.map((item) => html`<button type="button" key=${item} aria-pressed=${items.includes(item)}
        class="choice ${items.includes(item) ? 'on' : ''}" onClick=${() => toggle(cat, item)}>${item}</button>`)}</div>
    </div>`)}
    <input type="text" id=${id} value=${value} placeholder=${t('voices.instructPlaceholder')}
      onInput=${(e) => onChange(e.target.value)} aria-label=${t('voices.instruct')} />
    <span class="hint">${t('voices.instructHint')}</span>
  </div>`;
}

function NewVoice({ kind, info, gid, done, cancel }) {
  const [name, setName] = useState('');
  const [file, setFile] = useState(null);
  const [instruct, setInstruct] = useState('');
  const [speed, setSpeed] = useState('');
  const [err, setErr] = useState(null);
  const [saving, setSaving] = useState(false);
  const ids = { name: useId('vn'), file: useId('vf'), speed: useId('vs'), instruct: useId('vi') };
  const submit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setErr(null);
    try {
      let out;
      if (kind === 'clone') {
        const form = new FormData();
        form.append('name', name);
        if (file) form.append('file', file);
        out = await api('POST', `/api/g/${gid}/voices`, form);
      } else {
        out = await api('POST', `/api/g/${gid}/voices`, { kind: 'designed', name, instruct, speed: speed || null });
      }
      toast(t('voices.added', { name: out.voice.name }));
      await done(out.voice.id);
    } catch (error) {
      setErr(error);
    } finally {
      setSaving(false);
    }
  };
  const fieldErr = (f) => err?.field === f && err.message;
  return html`<form class="card editor" onSubmit=${submit} onKeyDown=${(e) => { if (e.key === 'Escape') cancel(); }}>
    <div class="card-head"><div>
      <h2>${kind === 'clone' ? t('voices.cloneTitle') : t('voices.designTitle')}</h2>
      <p class="sub">${kind === 'clone' ? t('voices.cloneDesc') : t('voices.designDesc')}</p>
    </div></div>
    ${err && !err.field && html`<p class="error-line" role="alert"><${Icon} name="warn" size=${16} />${err.message}</p>`}
    <div class="form-grid">
      <${Field} id=${ids.name} label=${t('voices.name')} error=${fieldErr('name')}>
        <input type="text" id=${ids.name} value=${name} required onInput=${(e) => setName(e.target.value)} /><//>
      ${kind === 'clone' && html`<${Field} id=${ids.file} label=${t('voices.audioFile', { n: info.upload_mb })} hint=${t('voices.audioHint')} error=${fieldErr('file')} wide>
        <input type="file" id=${ids.file} accept="audio/*,.mp3,.wav,.ogg,.opus,.flac,.m4a,.webm" onChange=${(e) => setFile(e.target.files[0] || null)} /><//>`}
      ${kind === 'designed' && html`<${Field} id=${ids.speed} label=${t('voices.speed')} hint=${t('voices.speedHint')}>
        <input type="number" id=${ids.speed} min="0.5" max="2" step="0.05" placeholder="1.0" value=${speed} onInput=${(e) => setSpeed(e.target.value)} /><//>`}
    </div>
    ${kind === 'designed' && html`<${Field} id=${ids.instruct} label=${t('voices.instruct')} error=${fieldErr('instruct')} wide>
      <${InstructComposer} id=${ids.instruct} design=${info.design} value=${instruct} onChange=${setInstruct} /><//>`}
    <div class="form-actions">
      <span class="spacer"></span>
      <button type="button" class="btn ghost" onClick=${cancel}>${t('common.cancel')}</button>
      <button type="submit" class="btn primary" disabled=${saving}>${saving ? t('common.working') : kind === 'clone' ? t('voices.upload') : t('voices.create')}</button>
    </div>
  </form>`;
}

function Preview({ v, gid, info }) {
  const [text, setText] = useState(info.preview_text || '');
  const [url, setUrl] = useState(null);
  const [run, busy] = useRunner();
  const player = useRef(null);
  const id = useId('pv');
  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);
  const here = () => run(async () => {
    const r = await fetch(`/api/g/${gid}/voices/${v.id}/preview`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
    });
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).error || msg; } catch { /* not JSON */ }
      throw new Error(msg);
    }
    setUrl(URL.createObjectURL(await r.blob()));
    setTimeout(() => player.current && player.current.play().catch(() => {}), 50);
  });
  const inCall = () => run(() => api('POST', '/api/control', { action: 'say', guild_id: gid, text, voice_id: v.id }), t('voices.sayingInCall'));
  return html`<div class="preview">
    <label for=${id} class="field-label">${t('voices.previewLabel')}</label>
    <div class="row">
      <input type="text" id=${id} class="grow" value=${text} maxlength="300" onInput=${(e) => setText(e.target.value)} />
      <button class="btn primary" disabled=${busy || !text.trim()} onClick=${here}><${Icon} name="play" size=${14} />${busy ? t('voices.generating') : t('voices.playHere')}</button>
      <button class="btn" disabled=${busy || !text.trim()} onClick=${inCall}>${t('voices.playInCall')}</button>
    </div>
    ${url && html`<audio ref=${player} controls src=${url}></audio>`}
  </div>`;
}

function useAudioPeaks(gid, v, which, enabled) {
  const [peaks, setPeaks] = useState(null);
  useEffect(() => {
    if (!enabled) { setPeaks(null); return undefined; }
    let live = true;
    api('GET', `/api/g/${gid}/voices/${v.id}/peaks?which=${which}&n=900`)
      .then((p) => live && setPeaks(p)).catch(() => live && setPeaks(null));
    return () => { live = false; };
  }, [gid, v.id, which, v.updated_at, enabled]);
  return peaks;
}

function VoiceDetail({ v, info, gid, refresh, close }) {
  const [run, busy] = useRunner();
  const [form, setForm] = useState(() => ({
    name: v.name, gain_db: v.gain_db ?? 0, speed: v.speed ?? '', num_step: v.num_step ?? '',
    tags: (v.tags || []).join(', '), language: v.language || '', instruct: v.instruct || '',
  }));
  const [text, setText] = useState(v.ref_text || '');
  const [sel, setSel] = useState(null);
  const [newFile, setNewFile] = useState(null);
  const sourceAudio = useRef(null);
  const refAudio = useRef(null);
  const isSpeaker = v.kind === 'speaker';
  const isClone = v.kind === 'clone';
  const audioOk = !isSpeaker || v.consent === 'accepted';
  const source = useAudioPeaks(gid, v, 'source', isClone && !!v.source_name);
  const ref = useAudioPeaks(gid, v, 'ref', !!v.ref_seconds && audioOk);
  const ids = { text: useId('tx'), name: useId('n'), gain: useId('g'), speed: useId('sp'), steps: useId('st'), lang: useId('l'),
    tags: useId('tg'), start: useId('s0'), end: useId('s1'), file: useId('rf'), instruct: useId('in') };
  useEffect(() => setText(v.ref_text || ''), [v.ref_text]);
  useEffect(() => { if (source && source.auto && !sel) setSel(source.auto); }, [source]);

  const bust = encodeURIComponent(v.updated_at || '');
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });
  const save = () => run(async () => {
    const body = { name: form.name, gain_db: form.gain_db === '' ? 0 : Number(form.gain_db),
      speed: form.speed === '' ? null : Number(form.speed), num_step: form.num_step === '' ? null : Number(form.num_step),
      tags: form.tags, language: form.language };
    if (v.kind === 'designed') body.instruct = form.instruct;
    await api('PATCH', `/api/g/${gid}/voices/${v.id}`, body);
    await refresh();
  }, t('common.saved'));
  const saveText = () => run(async () => { await api('PATCH', `/api/g/${gid}/voices/${v.id}`, { ref_text: text }); await refresh(); }, t('voices.transcriptSaved'));
  const transcribe = () => run(async () => {
    const out = await api('POST', `/api/g/${gid}/voices/${v.id}/transcribe`);
    setText(out.ref_text);
    await refresh();
  }, t('voices.transcribed'));
  const build = () => run(async () => { await api('POST', `/api/g/${gid}/voices/${v.id}/build`); await refresh(); }, t('voices.buildQueued'));
  const ingest = (withSel) => run(async () => {
    const formData = new FormData();
    if (newFile) formData.append('file', newFile);
    if (withSel && sel) { formData.append('start_s', sel[0].toFixed(2)); formData.append('end_s', sel[1].toFixed(2)); }
    await api('POST', `/api/g/${gid}/voices/${v.id}/ingest`, formData);
    setNewFile(null);
    if (!withSel) setSel(null);
    await refresh();
  }, withSel ? t('voices.cutDone') : t('voices.autoDone'));
  const remove = () => run(async () => { await api('DELETE', `/api/g/${gid}/voices/${v.id}`); close(); await refresh(); }, t('voices.deleted', { name: v.name }));
  const clearCache = () => run(async () => { await api('POST', `/api/g/${gid}/voices/cache/clear`, { voice_id: v.id }); await refresh(); }, t('voices.cacheCleared'));

  const selLen = sel ? sel[1] - sel[0] : 0;
  const building = v.status === 'queued' || v.status === 'building';
  const setEdge = (i, value) => {
    if (!source || value === '' || Number.isNaN(Number(value))) return;
    const n = Math.min(source.duration, Math.max(0, Number(value)));
    const next = sel ? [...sel] : [0, Math.min(12, source.duration)];
    next[i] = n;
    setSel([Math.min(...next), Math.max(...next)]);
  };

  return html`
    <header class="page-head">
      <div class="page-title">
        <button class="btn link back" onClick=${close}><${Icon} name="chevron" size=${14} />${t('voices.back')}</button>
        <h1>${v.name} ${v.is_bot && html`<span class="tag accent">${t('voices.botVoice')}</span>`}</h1>
        <p class="page-desc">${t(`voices.kindDesc.${v.kind}`)}</p>
      </div>
      <div class="page-actions">
        <span class="tag ${statusTone(v)}">${statusLabel(v)}</span>
        <button class="btn primary" disabled=${busy || building || !audioOk} onClick=${build}>
          ${building ? t('voices.building') : v.has_prompt ? t('voices.rebuild') : t('voices.build')}</button>
        ${!v.is_bot && !isSpeaker && html`<${ConfirmButton} label=${t('common.delete')} confirmLabel=${t('voices.deleteConfirm')} disabled=${busy} onConfirm=${remove} />`}
      </div>
    </header>
    ${v.status === 'failed' && v.error && html`<p class="notice bad"><${Icon} name="warn" size=${16} />${t('voices.buildFailed', { error: v.error })}</p>`}
    ${v.needs_build && v.has_prompt && html`<p class="notice"><${Icon} name="info" size=${16} />${t('voices.changedSince')}</p>`}

    <section class="card"><${Preview} v=${v} gid=${gid} info=${info} /></section>

    ${isClone && html`<section class="card">
      <div class="card-head"><div><h2>${t('voices.referenceTitle')}</h2><p class="sub">${t('voices.referenceDesc')}</p></div></div>
      ${source ? html`<div class="wave-box">
        <div class="wave-head"><span class="muted small">${t('voices.source', { file: v.source_name, len: secs(source.duration) })}</span>
          ${sel && html`<span class="small ${selLen > 12 ? 'warn-text' : ''}">${t('voices.selection', { len: selLen.toFixed(1) })}${selLen > 12 ? ` · ${t('voices.tooLong')}` : ''}</span>`}</div>
        <${Waveform} peaks=${source.peaks} duration=${source.duration} segments=${source.segments} selection=${sel}
          onSelect=${(a, b) => setSel([a, b])} audioRef=${sourceAudio} label=${t('voices.sourceWave')} />
        <div class="legend small"><span class="swatch speech"></span>${t('voices.legendSpeech')}<span class="swatch chosen"></span>${t('voices.legendSelection')}</div>
        <div class="row">
          <label class="inline-label" for=${ids.start}>${t('voices.start')}</label>
          <input type="number" class="num" id=${ids.start} step="0.1" min="0" value=${sel ? sel[0].toFixed(1) : ''} onChange=${(e) => setEdge(0, e.target.value)} />
          <label class="inline-label" for=${ids.end}>${t('voices.end')}</label>
          <input type="number" class="num" id=${ids.end} step="0.1" min="0" value=${sel ? sel[1].toFixed(1) : ''} onChange=${(e) => setEdge(1, e.target.value)} />
          <span class="spacer"></span>
          <audio ref=${sourceAudio} controls preload="none" src=${`/api/g/${gid}/voices/${v.id}/audio/source?v=${bust}`}></audio>
        </div>
        <div class="row">
          <button class="btn primary" disabled=${busy || !sel || selLen < 0.5} onClick=${() => ingest(true)}>${t('voices.useSelection')}</button>
          <button class="btn" disabled=${busy} onClick=${() => ingest(false)}>${t('voices.pickAuto')}</button>
          <button class="btn ghost" disabled=${!sel} onClick=${() => {
            const a = sourceAudio.current; if (!a || !sel) return;
            a.currentTime = sel[0]; a.play();
            const stop = () => { if (a.currentTime >= sel[1]) { a.pause(); a.removeEventListener('timeupdate', stop); } };
            a.addEventListener('timeupdate', stop);
          }}><${Icon} name="play" size=${14} />${t('voices.playSelection')}</button>
        </div>
      </div>` : html`<p class="muted">${v.source_name ? t('common.loading') : t('voices.noSource')}</p>`}
      ${ref && html`<div class="wave-box">
        <div class="wave-head"><span class="muted small">${t('voices.inUse', { len: secs(ref.duration) })}</span></div>
        <${Waveform} peaks=${ref.peaks} duration=${ref.duration} audioRef=${refAudio} height=${56} label=${t('voices.refWave')} />
        <audio ref=${refAudio} controls preload="none" src=${`/api/g/${gid}/voices/${v.id}/audio/ref?v=${bust}`}></audio>
      </div>`}
      <div class="row">
        <label class="inline-label" for=${ids.file}>${t('voices.replaceLabel')}</label>
        <input type="file" id=${ids.file} accept="audio/*,.mp3,.wav,.ogg,.opus,.flac,.m4a,.webm" onChange=${(e) => setNewFile(e.target.files[0] || null)} />
        <button class="btn" disabled=${busy || !newFile} onClick=${() => ingest(false)}>${t('voices.replace')}</button>
      </div>
    </section>`}

    ${isSpeaker && audioOk && ref && html`<section class="card"><h2>${t('voices.theirReference')}</h2>
      <${Waveform} peaks=${ref.peaks} duration=${ref.duration} audioRef=${refAudio} height=${56} />
      <audio ref=${refAudio} controls preload="none" src=${`/api/g/${gid}/voices/${v.id}/audio/ref?v=${bust}`}></audio>
    </section>`}

    ${(isClone || isSpeaker) && html`<section class="card">
      <div class="card-head"><div><h2>${t('voices.transcriptTitle')}</h2><p class="sub">${t('voices.transcriptDesc')}</p></div></div>
      <label for=${ids.text} class="visually-hidden">${t('voices.transcriptTitle')}</label>
      <textarea id=${ids.text} rows="3" value=${text} readOnly=${isSpeaker} onInput=${(e) => setText(e.target.value)}></textarea>
      ${isClone && html`<div class="row">
        <button class="btn" disabled=${busy || !v.ref_seconds} onClick=${transcribe}>${t('voices.autoTranscribe')}</button>
        <button class="btn primary" disabled=${busy || text === (v.ref_text || '')} onClick=${saveText}>${t('voices.saveTranscript')}</button>
      </div>`}
    </section>`}

    ${!isSpeaker && html`<section class="card">
      <h2>${t('voices.settingsTitle')}</h2>
      <div class="form-grid">
        <${Field} id=${ids.name} label=${t('voices.name')}><input type="text" id=${ids.name} value=${form.name} onInput=${set('name')} /><//>
        <${Field} id=${ids.gain} label=${t('voices.gain')} hint=${t('voices.gainHint')}>
          <div class="with-unit"><input type="number" id=${ids.gain} step="0.5" min="-30" max="20" value=${form.gain_db} onInput=${set('gain_db')} /><span>dB</span></div><//>
        <${Field} id=${ids.speed} label=${t('voices.speed')} hint=${v.kind === 'designed' ? t('voices.speedDesigned') : t('voices.speedHint')}>
          <input type="number" id=${ids.speed} step="0.05" min="0.5" max="2" placeholder="1.0" value=${form.speed} onInput=${set('speed')} /><//>
        <${Field} id=${ids.steps} label=${t('voices.steps')} hint=${t('voices.stepsHint')}>
          <input type="number" id=${ids.steps} step="1" min="4" max="64" value=${form.num_step} onInput=${set('num_step')} /><//>
        <${Field} id=${ids.lang} label=${t('voices.language')} hint=${t('voices.languageHint')}>
          <input type="text" id=${ids.lang} value=${form.language} onInput=${set('language')} /><//>
        <${Field} id=${ids.tags} label=${t('voices.tags')}>
          <input type="text" id=${ids.tags} placeholder=${t('voices.tagsPlaceholder')} value=${form.tags} onInput=${set('tags')} /><//>
      </div>
      ${v.kind === 'designed' && html`<${Field} id=${ids.instruct} label=${t('voices.instruct')} hint=${t('voices.instructChanges')} wide>
        <${InstructComposer} id=${ids.instruct} design=${info.design} value=${form.instruct} onChange=${(x) => setForm({ ...form, instruct: x })} /><//>`}
      <div class="form-actions"><span class="spacer"></span><button class="btn primary" disabled=${busy} onClick=${save}>${t('voices.saveSettings')}</button></div>
    </section>`}

    <section class="card row">
      <span class="grow">${t('voices.cachedLines', { size: mb(v.cache_bytes) })}</span>
      <button class="btn" disabled=${busy || !v.cache_bytes} onClick=${clearCache}>${t('voices.clear')}</button>
    </section>`;
}

function Speakers({ info, gid, refresh, open }) {
  const [run, busy] = useRunner();
  const act = (method, path, body, okText) => run(async () => { await api(method, `/api/g/${gid}/speakers/${path}`, body); await refresh(); }, okText);
  return html`<section class="card">
    <div class="card-head"><div><h2>${t('voices.peopleTitle')}</h2><p class="sub">${t('voices.peopleDesc')}</p></div></div>
    ${!info.speakers.length ? html`<${Empty} title=${t('voices.peopleEmptyTitle')} body=${t('voices.peopleEmptyBody')} />` : html`<div class="table-wrap"><table class="table">
      <thead><tr><th>${t('voices.colPerson')}</th><th>${t('voices.colConsent')}</th><th>${t('voices.colStatus')}</th><th>${t('voices.colHeard')}</th>
        <th><span class="visually-hidden">${t('common.actions')}</span></th></tr></thead>
      <tbody>${info.speakers.map((v) => {
        const uid = S(v.owner_user_id);
        const ok = v.consent === 'accepted';
        return html`<tr key=${v.id}>
          <td><b>${v.owner || uid}</b></td>
          <td><span class="tag ${ok ? 'ok' : 'bad'}">${t(`consent.${v.consent || 'none'}`)}</span></td>
          <td><span class="tag ${statusTone(v)}" title=${v.error || ''}>${statusLabel(v)}</span></td>
          <td class="tabular">${secs(v.clip_seconds)}</td>
          <td><div class="row end">
            <button class="btn small" disabled=${!ok} onClick=${() => open(v.id)}>${t('common.open')}</button>
            <button class="btn small" disabled=${busy || !ok} onClick=${() => act('POST', `${uid}/rebuild`, undefined, t('voices.buildQueued'))}>${t('voices.rebuild')}</button>
            <${ConfirmButton} label=${t('voices.promote')} confirmLabel=${t('voices.promoteConfirm', { name: v.owner || uid })}
              cls="small" disabled=${busy || !ok} onConfirm=${() => act('POST', `${uid}/promote`, {}, t('voices.promoted'))} />
            <${ConfirmButton} label=${t('common.delete')} confirmLabel=${t('voices.deleteTheirs')} cls="small" disabled=${busy}
              onConfirm=${() => act('DELETE', uid, undefined, t('voices.deletedTheirs'))} />
          </div></td>
        </tr>`;
      })}</tbody>
    </table></div>`}
  </section>`;
}

// ---------------------------------------------------------------- sounds

export function Sounds({ data, gid, guild, missing, reload, control }) {
  const [name, setName] = useState('');
  const [file, setFile] = useState(null);
  const [err, setErr] = useState(null);
  const [making, setMaking] = useState(null);
  const [run, busy] = useRunner();
  const input = useRef(null);
  const ids = { name: useId('sn'), file: useId('sf') };
  if (!data) return html`<${Waiting} section="sounds" missing=${missing} guild=${guild} />`;
  if (!data.meta.media?.sounds) {
    return html`<div class="page"><${PageHead} section="sounds" />
      <${Empty} title=${t('sounds.noLibraryTitle')} body=${t('sounds.noLibraryBody')} /></div>`;
  }
  const limits = data.settings.effective;
  const upload = async (e) => {
    e.preventDefault();
    setErr(null);
    const form = new FormData();
    form.append('name', name);
    if (file) form.append('file', file);
    await run(async () => {
      try { await api('POST', `/api/g/${gid}/sounds`, form); } catch (error) { setErr(error); throw error; }
      setName(''); setFile(null);
      if (input.current) input.current.value = '';
      await reload();
    }, t('sounds.added'));
  };
  const patch = (s, body, okText) => run(async () => { await api('PATCH', `/api/g/${gid}/sounds/${s.id}`, body); await reload(); }, okText);
  const remove = (s) => run(async () => { await api('DELETE', `/api/g/${gid}/sounds/${s.id}`); await reload(); }, t('sounds.deleted', { name: s.name }));
  const play = (s) => run(() => api('POST', `/api/g/${gid}/sounds/${s.id}/play`), t('sounds.playing', { name: s.name }));

  return html`<div class="page">
    <${PageHead} section="sounds" />
    <form class="card" onSubmit=${upload}>
      <div class="card-head"><div><h2>${t('sounds.addTitle')}</h2>
        <p class="sub">${t('sounds.addDesc', { mb: limits['sounds.max_mb'], s: limits['sounds.max_seconds'] })}</p></div></div>
      ${err && !err.field && html`<p class="error-line">${err.message}</p>`}
      <div class="form-grid">
        <${Field} id=${ids.name} label=${t('sounds.name')} hint=${t('sounds.nameHint')} error=${err?.field === 'name' && err.message}>
          <input type="text" id=${ids.name} placeholder=${t('sounds.namePlaceholder')} value=${name} onInput=${(e) => setName(e.target.value)} /><//>
        <${Field} id=${ids.file} label=${t('sounds.file')} error=${err?.field === 'file' && err.message}>
          <input type="file" id=${ids.file} ref=${input} accept="audio/*,.mp3,.wav,.ogg,.opus,.flac,.m4a,.webm"
            onChange=${(e) => { setFile(e.target.files[0] || null); if (!name && e.target.files[0]) setName(e.target.files[0].name.replace(/\.[^.]+$/, '')); }} /><//>
      </div>
      <div class="form-actions"><span class="spacer"></span>
        <button class="btn primary" type="submit" disabled=${busy || !file}>${t('sounds.upload')}</button></div>
    </form>
    <section class="card">
      <h2>${t('sounds.listTitle', { n: data.sounds.length })}</h2>
      ${data.sounds.length === 0 ? html`<${Empty} title=${t('sounds.emptyTitle')} body=${t('sounds.emptyBody')} />`
        : html`<ul class="sound-list">${data.sounds.map((s) => html`<li key=${s.id} class="sound-row ${s.enabled ? '' : 'off'}">
          <div class="row">
            <input type="checkbox" role="switch" class="switch" checked=${s.enabled} aria-label=${t(s.enabled ? 'sounds.disableX' : 'sounds.enableX', { name: s.name })}
              onChange=${(e) => patch(s, { enabled: e.target.checked }, t(e.target.checked ? 'sounds.enabledToast' : 'sounds.disabledToast', { name: s.name }))} />
            <${Rename} value=${s.name} onCommit=${(n) => patch(s, { name: n }, t('sounds.renamed'))} />
            <span class="muted small tabular">${secs(s.duration_s)}</span>
            <audio controls preload="none" src=${`/api/g/${gid}/sounds/${s.id}/audio`}></audio>
            <span class="spacer"></span>
            <button class="btn small" disabled=${busy} onClick=${() => play(s)}><${Icon} name="play" size=${14} />${t('sounds.playInCall')}</button>
            <button class="btn small" aria-expanded=${making === s.id} onClick=${() => setMaking(making === s.id ? null : s.id)}>${t('sounds.makeReaction')}</button>
            <${ConfirmButton} label=${t('common.delete')} confirmLabel=${t('sounds.deleteConfirm')} cls="small" disabled=${busy} onConfirm=${() => remove(s)} />
          </div>
          ${making === s.id && html`<${MakeReaction} s=${s} gid=${gid} done=${async () => { setMaking(null); await reload(); }} />`}
        </li>`)}</ul>`}
    </section>
  </div>`;
}

function Rename({ value, onCommit }) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  const commit = () => { if (text.trim() && text.trim() !== value) onCommit(text.trim()); else setText(value); };
  return html`<input type="text" class="rename" value=${text} aria-label=${t('sounds.rename', { name: value })}
    onInput=${(e) => setText(e.target.value)} onBlur=${commit}
    onKeyDown=${(e) => { if (e.key === 'Enter') { e.preventDefault(); e.target.blur(); } else if (e.key === 'Escape') setText(value); }} />`;
}

function MakeReaction({ s, gid, done }) {
  const [type, setType] = useState('slash');
  const [value, setValue] = useState(s.name);
  const [run, busy] = useRunner();
  const ids = { type: useId('mt'), value: useId('mv') };
  const create = () => run(async () => {
    const words = value.split(',').map((x) => x.trim()).filter(Boolean);
    const trigger = type === 'slash' ? { type, name: value } : { type, phrases: words };
    await api('POST', `/api/g/${gid}/sounds/${s.id}/reaction`, { trigger });
    await done();
  }, t('sounds.reactionMade'));
  return html`<div class="make-reaction">
    <label for=${ids.type} class="inline-label">${t('sounds.playWhen')}</label>
    <select id=${ids.type} value=${type} onChange=${(e) => { setType(e.target.value); setValue(e.target.value === 'slash' ? s.name : ''); }}>
      <option value="slash">${t('sounds.whenSlash')}</option>
      <option value="phrase">${t('sounds.whenPhrase')}</option>
      <option value="command">${t('sounds.whenCommand')}</option>
    </select>
    <label for=${ids.value} class="visually-hidden">${t('sounds.triggerWords')}</label>
    <input type="text" id=${ids.value} class="grow" value=${value} onInput=${(e) => setValue(e.target.value)}
      placeholder=${type === 'slash' ? t('sounds.slashPlaceholder') : t('sounds.wordsPlaceholder')} />
    <button class="btn primary small" disabled=${busy || !value.trim()} onClick=${create}>${t('common.create')}</button>
  </div>`;
}

// ---------------------------------------------------------------- soundboard (Overview)

export function Soundboard({ gid, changed, control, busy, inCall, go }) {
  const [board, setBoard] = useState(null);
  const [voice, setVoice] = useState('');
  const [text, setText] = useState('');
  const ids = { text: useId('sb'), voice: useId('sv') };
  useEffect(() => {
    let live = true;
    api('GET', `/api/g/${gid}/board`).then((b) => live && setBoard(b)).catch(() => live && setBoard(false));
    return () => { live = false; };
  }, [gid, changed]);
  const say = async (e) => {
    e.preventDefault();
    const tx = text.trim();
    if (!tx) return;
    if (await control('say', { guild_id: gid, text: tx, ...(voice ? { voice_id: voice } : {}) })) setText('');
  };
  const voices = board ? board.voices : [];
  const play = async (s) => {
    try { await api('POST', `/api/g/${gid}/sounds/${s.id}/play`); toast(t('sounds.playing', { name: s.name })); } catch (err) { toast(err.message, true); }
  };
  return html`<div class="soundboard">
    <form class="say" onSubmit=${say}>
      <label for=${ids.text} class="visually-hidden">${t('overview.sayLabel')}</label>
      <input type="text" id=${ids.text} placeholder=${t('overview.sayPlaceholder')} value=${text} maxlength="300" onInput=${(e) => setText(e.target.value)} />
      ${voices.length > 0 && html`<label for=${ids.voice} class="visually-hidden">${t('overview.sayVoice')}</label>
        <select id=${ids.voice} value=${voice} onChange=${(e) => setVoice(e.target.value)}>
          <option value="">${t('overview.defaultVoice')}</option>
          ${voices.map((v) => html`<option key=${v.id} value=${S(v.id)} disabled=${v.status !== 'ready'}>
            ${v.name}${S(v.id) === S(board.bot_voice_id) ? t('pickers.botVoiceSuffix') : ''}${v.status !== 'ready' ? t('voices.stateNotBuilt') : ''}</option>`)}
        </select>`}
      <button class="btn" type="submit" disabled=${!text.trim() || busy[`say:${gid}`] || !inCall} title=${inCall ? '' : t('overview.joinFirst')}>${t('overview.say')}</button>
    </form>
    ${board && board.sounds.length > 0 ? html`<div class="pads" role="group" aria-label=${t('overview.soundboard')}>
      ${board.sounds.map((s) => html`<button key=${s.id} class="pad" disabled=${!inCall} title=${inCall ? t('sounds.playX', { name: s.name }) : t('overview.joinFirst')}
        onClick=${() => play(s)}><${Icon} name="play" size=${12} />${s.name}</button>`)}
    </div>` : board && html`<p class="muted small">${t('overview.noSounds')} <button class="btn link" onClick=${() => go('sounds')}>${t('overview.addSounds')}</button></p>`}
    ${!inCall && html`<p class="hint">${t('overview.joinFirst')}</p>`}
  </div>`;
}

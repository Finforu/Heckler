// Gags & replies: everything the bot reacts to. A simple editor for the
// everyday gag ("when someone says X, reply Y"), the full editor behind
// "Advanced", approvals for what people make in Discord, and the test bench.
import { useEffect, useRef, useState } from './vendor/hooks.js';
import {
  api, Chips, ConfirmButton, Empty, Field, html, Icon, load, PageHead, S, save, Slider, t, Tabs, toast, Toggle,
  useId, useRunner, Waiting, when,
} from './lib.js';
import { peopleNames, PersonPicker, voiceLabel, VoicePicker } from './pickers.js';

const KINDS = ['gag', 'command', 'response', 'sound'];
const clone = (x) => JSON.parse(JSON.stringify(x));

// ---------------------------------------------------------------- helpers

function triggerText(tr, names) {
  if (tr.type === 'phrase' || tr.type === 'command') return (tr.phrases || []).join(' · ');
  if (tr.type === 'swap') return `${tr.word} → ${tr.to}`;
  if (tr.type === 'slash') return `/${tr.name}`;
  const who = tr.user_id ? ` (${names.get(S(tr.user_id)) || tr.user_id})` : '';
  return t(`events.${tr.event}`) + who;
}

function stepText(s, data) {
  if (s.type === 'say') return s.text;
  if (s.type === 'sound') return `♪ ${data.sounds.find((x) => S(x.id) === S(s.sound_id))?.name || '?'}`;
  return `[${t(`builtins.${s.action}`)}]`;
}

const optionText = (o, data) => o.map((s) => stepText(s, data)).join(' → ');

// Can this reaction be shown in the simple editor without losing anything?
function isSimple(r) {
  return r.kind === 'gag' && r.triggers.length === 1 && r.triggers[0].type === 'phrase' && !(r.by_users && r.by_users.length)
    && r.options.every((o) => o.length === 1 && o[0].type === 'say' && !o[0].voice_id);
}

function newReaction(kind) {
  const base = { kind, name: '', enabled: true, status: 'approved', chance: 1, cooldown_s: null,
    voice_id: null, by_users: null, options: [[{ type: 'say', text: '' }]] };
  if (kind === 'command') return { ...base, triggers: [{ type: 'command', phrases: [] }], options: [[{ type: 'builtin', action: 'stop' }]] };
  if (kind === 'response') return { ...base, triggers: [{ type: 'event', event: 'hello' }] };
  if (kind === 'sound') return { ...base, triggers: [{ type: 'slash', name: '' }], options: [[{ type: 'sound', sound_id: null }]] };
  return { ...base, triggers: [{ type: 'phrase', phrases: [] }] };
}

function payload(r) {
  return {
    kind: r.kind, name: r.name, enabled: r.enabled, status: r.status, chance: r.chance,
    cooldown_s: r.cooldown_s === '' ? null : r.cooldown_s, voice_id: r.voice_id,
    by_users: r.by_users && r.by_users.length ? r.by_users : null, triggers: r.triggers, options: r.options,
  };
}

async function saveReaction(gid, id, r) {
  if (id === null) return api('POST', `/api/g/${gid}/reactions`, payload(r));
  return api('PATCH', `/api/g/${gid}/reactions/${id}`, payload(r));
}

// ---------------------------------------------------------------- list

function ReactionRow({ r, data, first, last, busy, onMove, onToggle, onEdit, onDelete }) {
  const names = peopleNames(data);
  return html`<li class="rrow ${r.enabled ? '' : 'off'}">
    <input type="checkbox" role="switch" class="switch" checked=${r.enabled} disabled=${busy}
      aria-label=${t(r.enabled ? 'gags.disableX' : 'gags.enableX', { name: r.name })} onChange=${(e) => onToggle(e.target.checked)} />
    <div class="rbody">
      <div class="rname">
        <button class="link-title" onClick=${onEdit}>${r.name}</button>
        ${r.status !== 'approved' && html`<span class="tag ${r.status === 'pending' ? 'warn' : 'bad'}">${t(`gags.status.${r.status}`)}</span>`}
      </div>
      <div class="rtriggers">${r.triggers.map((tr, k) => html`<span key=${k} class="trigger-chip">
        <span class="muted">${t(`triggers.short.${tr.type}`)}</span> ${triggerText(tr, names)}</span>`)}</div>
      <p class="rsays" title=${r.options.map((o) => optionText(o, data)).join('\n')}>
        ${r.options.length > 1 && html`<span class="muted">${t('gags.oneOf', { n: r.options.length })}</span> `}
        ${optionText(r.options[0] || [], data)}</p>
      <div class="rmeta">
        ${r.chance < 1 && html`<span class="tag">${t('gags.chanceTag', { n: Math.round(r.chance * 100) })}</span>`}
        ${r.cooldown_s !== null && html`<span class="tag">${t('gags.cooldownTag', { n: r.cooldown_s })}</span>`}
        ${r.voice_id !== null && html`<span class="tag">${t('gags.voiceTag', { name: voiceLabel(data, r.voice_id) })}</span>`}
        ${r.by_users && html`<span class="tag">${t('gags.onlyTag', { names: r.by_users.map((u) => names.get(S(u)) || u).join(', ') })}</span>`}
        <span class="muted small">${t('gags.by', { name: r.creator })}</span>
      </div>
    </div>
    <div class="ractions">
      <button class="btn icon ghost" aria-label=${t('gags.moveUp', { name: r.name })} title=${t('gags.moveUpTitle')}
        disabled=${busy || first} onClick=${() => onMove(-1)}><${Icon} name="up" size=${16} /></button>
      <button class="btn icon ghost" aria-label=${t('gags.moveDown', { name: r.name })} title=${t('gags.moveDownTitle')}
        disabled=${busy || last} onClick=${() => onMove(1)}><${Icon} name="down" size=${16} /></button>
      <button class="btn small" onClick=${onEdit}><${Icon} name="edit" size=${14} />${t('common.edit')}</button>
      <${ConfirmButton} label=${t('common.delete')} confirmLabel=${t('gags.deleteConfirm')} cls="small" disabled=${busy} onConfirm=${onDelete} />
    </div>
  </li>`;
}

function ReactionList({ kind, data, gid, reload, edit }) {
  const [run, busy] = useRunner();
  const all = data.reactions;
  const rows = all.filter((r) => r.kind === kind);
  const move = (r, dir) => run(async () => {
    const i = rows.indexOf(r), j = i + dir;
    if (j < 0 || j >= rows.length) return;
    const ids = all.map((x) => x.id);
    const a = ids.indexOf(r.id), b = ids.indexOf(rows[j].id);
    [ids[a], ids[b]] = [ids[b], ids[a]];
    await api('POST', `/api/g/${gid}/reactions/order`, { ids });
    await reload();
  });
  const patch = (r, fields, okText) => run(async () => { await api('PATCH', `/api/g/${gid}/reactions/${r.id}`, fields); await reload(); }, okText);
  const remove = (r) => run(async () => { await api('DELETE', `/api/g/${gid}/reactions/${r.id}`); await reload(); }, t('gags.deleted', { name: r.name }));

  return html`<section class="card">
    <div class="card-head">
      <div><h2>${t(`kinds.${kind}.title`)}</h2><p class="sub">${t(`kinds.${kind}.desc`)}</p></div>
      <button class="btn primary" onClick=${() => edit(null, newReaction(kind))}><${Icon} name="plus" size=${16} />${t(`kinds.${kind}.new`)}</button>
    </div>
    ${rows.length === 0 ? html`<${Empty} title=${t(`kinds.${kind}.emptyTitle`)} body=${t(`kinds.${kind}.emptyBody`)}
        action=${html`<button class="btn primary" onClick=${() => edit(null, newReaction(kind))}>${t(`kinds.${kind}.new`)}</button>`} />`
      : html`<p class="hint">${t('gags.orderHint')}</p>
        <ol class="rlist">${rows.map((r, i) => html`<${ReactionRow} key=${r.id} r=${r} data=${data} first=${i === 0} last=${i === rows.length - 1}
          busy=${busy} onMove=${(d) => move(r, d)} onToggle=${(v) => patch(r, { enabled: v }, v ? t('gags.enabled', { name: r.name }) : t('gags.disabled', { name: r.name }))}
          onEdit=${() => edit(r.id, clone(r))} onDelete=${() => remove(r)} />`)}</ol>`}
  </section>`;
}

// ---------------------------------------------------------------- simple editor

function Placeholders({ list }) {
  return html`<p class="hint">${t('gags.placeholders')} ${list.map((p) => html`<code key=${p} title=${t(`placeholders.${p}`)}>{${p}}</code> `)}</p>`;
}

function SimpleEditor({ r, setR, data, err }) {
  const ids = { name: useId('name'), phrases: useId('ph'), voice: useId('vc'), chance: useId('ch'), cool: useId('cd') };
  const replies = r.options.map((o) => o[0].text);
  const setReplies = (list) => setR({ ...r, options: list.map((text) => [{ type: 'say', text }]) });
  const bad = (f) => err && err.field && (err.field === f || err.field.startsWith(`${f}.`));
  const cooldownDefault = data.settings.effective['gags.cooldown_s'];
  return html`<div class="form-grid simple">
    <${Field} id=${ids.phrases} label=${t('simple.when')} hint=${t('simple.whenHint')} error=${bad('triggers') && err.message} wide>
      <${Chips} id=${ids.phrases} values=${r.triggers[0].phrases} placeholder=${t('simple.whenPlaceholder')} invalid=${bad('triggers')}
        onChange=${(v) => setR({ ...r, triggers: [{ type: 'phrase', phrases: v }] })} />
    <//>
    <div class="field wide ${bad('options') ? 'has-error' : ''}">
      <span class="field-label">${t('simple.reply')}</span>
      <ol class="reply-list">${replies.map((text, i) => html`<li key=${i}>
        <input type="text" value=${text} placeholder=${i === 0 ? t('simple.replyPlaceholder') : t('simple.replyMore')}
          class=${bad(`options.${i}`) ? 'invalid' : ''} aria-label=${t('simple.replyN', { n: i + 1 })}
          onInput=${(e) => setReplies(replies.map((x, j) => (j === i ? e.target.value : x)))} />
        ${replies.length > 1 && html`<button type="button" class="btn icon ghost" aria-label=${t('simple.removeReply', { n: i + 1 })}
          onClick=${() => setReplies(replies.filter((_, j) => j !== i))}><${Icon} name="close" size=${16} /></button>`}
      </li>`)}</ol>
      <div class="row">
        <button type="button" class="btn small" onClick=${() => setReplies([...replies, ''])}><${Icon} name="plus" size=${14} />${t('simple.addReply')}</button>
        ${replies.length > 1 && html`<span class="hint">${t('simple.randomHint')}</span>`}
      </div>
      ${bad('options') ? html`<span class="field-error">${err.message}</span>` : html`<${Placeholders} list=${['name']} />`}
    </div>
    <${Field} id=${ids.voice} label=${t('simple.voice')} hint=${t('simple.voiceHint')}>
      <${VoicePicker} id=${ids.voice} data=${data} value=${r.voice_id} inheritLabel=${t('pickers.serverDefault')}
        onChange=${(v) => setR({ ...r, voice_id: v })} invalid=${bad('voice_id')} />
    <//>
    <${Field} id=${ids.chance} label=${t('simple.chance')} hint=${t('simple.chanceHint')}>
      <${Slider} id=${ids.chance} value=${Math.round((r.chance ?? 1) * 100)} min=${0} max=${100} step=${5}
        format=${(v) => `${v}%`} onInput=${(v) => setR({ ...r, chance: v / 100 })} />
    <//>
    <${Field} id=${ids.cool} label=${t('simple.cooldown')} hint=${t('simple.cooldownHint', { n: cooldownDefault })} error=${bad('cooldown_s') && err.message}>
      <div class="with-unit"><input type="number" id=${ids.cool} min="0" step="1" value=${r.cooldown_s ?? ''}
        placeholder=${t('simple.cooldownPlaceholder', { n: cooldownDefault })}
        onInput=${(e) => setR({ ...r, cooldown_s: e.target.value === '' ? null : Number(e.target.value) })} /><span>${t('units.seconds')}</span></div>
    <//>
    <${Field} id=${ids.name} label=${t('simple.name')} hint=${t('simple.nameHint')} error=${bad('name') && err.message}>
      <input type="text" id=${ids.name} value=${r.name} placeholder=${r.triggers[0].phrases[0] || t('simple.namePlaceholder')}
        onInput=${(e) => setR({ ...r, name: e.target.value })} />
    <//>
  </div>`;
}

// ---------------------------------------------------------------- advanced editor

function defaultTrigger(type) {
  if (type === 'swap') return { type, word: '', to: '', also: [], connectors: ['de', 'e', 'y'] };
  if (type === 'event') return { type, event: 'hello' };
  if (type === 'slash') return { type, name: '' };
  return { type, phrases: [] };
}

function TriggerEditor({ tr, data, onChange }) {
  const set = (fields) => onChange({ ...tr, ...fields });
  return html`<div class="trigger">
    <select value=${tr.type} aria-label=${t('advanced.triggerType')} onChange=${(e) => onChange(defaultTrigger(e.target.value))}>
      ${data.meta.trigger_types.map((k) => html`<option key=${k} value=${k}>${t(`triggers.${k}`)}</option>`)}
    </select>
    ${(tr.type === 'phrase' || tr.type === 'command') && html`<${Chips} values=${tr.phrases} label=${t('advanced.phrases')}
      placeholder=${tr.type === 'command' ? t('advanced.commandPlaceholder') : t('advanced.phrasePlaceholder')}
      onChange=${(v) => set({ phrases: v })} />`}
    ${tr.type === 'swap' && html`<div class="swap">
      <input type="text" placeholder=${t('advanced.swapWord')} aria-label=${t('advanced.swapWord')} value=${tr.word} onInput=${(e) => set({ word: e.target.value })} />
      <span class="muted">→</span>
      <input type="text" placeholder=${t('advanced.swapTo')} aria-label=${t('advanced.swapTo')} value=${tr.to} onInput=${(e) => set({ to: e.target.value })} />
      <label class="inline-label">${t('advanced.swapAlso')}<${Chips} values=${tr.also} label=${t('advanced.swapAlso')} placeholder=${t('advanced.swapAlsoPlaceholder')} onChange=${(v) => set({ also: v })} /></label>
      <label class="inline-label">${t('advanced.swapConnectors')}<${Chips} values=${tr.connectors} label=${t('advanced.swapConnectors')} onChange=${(v) => set({ connectors: v })} /></label>
    </div>`}
    ${tr.type === 'event' && html`<div class="row">
      <select value=${tr.event} aria-label=${t('advanced.event')} onChange=${(e) => set({ event: e.target.value })}>
        ${data.meta.events.map((k) => html`<option key=${k} value=${k}>${t(`events.${k}`)}</option>`)}
      </select>
      <${PersonPicker} data=${data} value=${tr.user_id} anyLabel=${t('advanced.forEveryone')}
        onChange=${(u) => { const n = { ...tr }; if (u) n.user_id = u; else delete n.user_id; onChange(n); }} />
    </div>`}
    ${tr.type === 'slash' && html`<div class="with-unit prefix"><span>/sound</span>
      <input type="text" placeholder=${t('advanced.slashPlaceholder')} aria-label=${t('advanced.slashName')} value=${tr.name} onInput=${(e) => set({ name: e.target.value })} /></div>`}
  </div>`;
}

function StepEditor({ s, data, onChange }) {
  const set = (fields) => onChange({ ...s, ...fields });
  const retype = (type) => {
    if (type === 'say') onChange({ type, text: '' });
    else if (type === 'sound') onChange({ type, sound_id: data.sounds[0] ? S(data.sounds[0].id) : null });
    else onChange({ type, action: data.meta.builtins[0] });
  };
  return html`<div class="step-body">
    <select value=${s.type} aria-label=${t('advanced.stepType')} onChange=${(e) => retype(e.target.value)}>
      ${data.meta.step_types.map((k) => html`<option key=${k} value=${k}>${t(`steps.${k}`)}</option>`)}
    </select>
    ${s.type === 'say' && html`<input type="text" class="grow" placeholder=${t('advanced.sayPlaceholder')} aria-label=${t('advanced.sayText')}
      value=${s.text} onInput=${(e) => set({ text: e.target.value })} />
      <${VoicePicker} data=${data} value=${s.voice_id} inheritLabel=${t('pickers.inheritReaction')}
        onChange=${(v) => { const n = { ...s }; if (v) n.voice_id = v; else delete n.voice_id; onChange(n); }} />`}
    ${s.type === 'sound' && (data.sounds.length
      ? html`<select class="grow" value=${S(s.sound_id)} aria-label=${t('advanced.sound')} onChange=${(e) => set({ sound_id: e.target.value })}>
          ${!s.sound_id && html`<option value="">${t('advanced.pickSound')}</option>`}
          ${data.sounds.map((x) => html`<option key=${x.id} value=${S(x.id)}>${x.name}${x.enabled ? '' : t('advanced.soundOff')}</option>`)}
        </select>`
      : html`<span class="muted grow">${t('advanced.noSounds')}</span>`)}
    ${s.type === 'builtin' && html`<select value=${s.action} aria-label=${t('advanced.action')} onChange=${(e) => set({ action: e.target.value })}>
      ${data.meta.builtins.map((k) => html`<option key=${k} value=${k}>${t(`builtins.${k}`)}</option>`)}
    </select>`}
  </div>`;
}

// Helper replies that get {result}: the time, the coin, the dice, who was picked, the time left.
const RESULT_EVENTS = ['timer_left', 'time_now', 'coin_result', 'dice_result', 'pick_result'];

function placeholdersFor(triggers) {
  const out = new Set(['name']);
  for (const tr of triggers) {
    if (tr.type === 'swap') ['subject', 'connector', 'to'].forEach((p) => out.add(p));
    if (tr.type === 'command' || (tr.type === 'event' && (tr.event.startsWith('timer') || tr.event === 'alarm_set'))) {
      ['said', 'message'].forEach((p) => out.add(p));
    }
    if (tr.type === 'event' && RESULT_EVENTS.includes(tr.event)) out.add('result');
  }
  return [...out];
}

function AdvancedEditor({ r, setR, data, err }) {
  const ids = { name: useId('an'), kind: useId('ak'), status: useId('as'), chance: useId('ac'), cool: useId('acd'), voice: useId('av'), by: useId('ab') };
  const set = (fields) => setR({ ...r, ...fields });
  const bad = (f) => err && err.field && (err.field === f || err.field.startsWith(`${f}.`));
  const names = peopleNames(data);
  const byIds = (r.by_users || []).map(S);
  const setTrigger = (i, tr) => set({ triggers: r.triggers.map((x, j) => (j === i ? tr : x)) });
  const setOption = (i, o) => set({ options: r.options.map((x, j) => (j === i ? o : x)) });

  return html`<div class="advanced">
    <div class="form-grid">
      <${Field} id=${ids.name} label=${t('advanced.name')} error=${bad('name') && err.message}>
        <input type="text" id=${ids.name} value=${r.name} required onInput=${(e) => set({ name: e.target.value })} /><//>
      <${Field} id=${ids.kind} label=${t('advanced.kind')} hint=${t('advanced.kindHint')}>
        <select id=${ids.kind} value=${r.kind} onChange=${(e) => set({ kind: e.target.value })}>
          ${data.meta.kinds.map((k) => html`<option key=${k} value=${k}>${t(`kinds.${k}.one`)}</option>`)}</select><//>
      <${Field} id=${ids.status} label=${t('advanced.status')}>
        <select id=${ids.status} value=${r.status} onChange=${(e) => set({ status: e.target.value })}>
          ${data.meta.statuses.map((k) => html`<option key=${k} value=${k}>${t(`gags.status.${k}`)}</option>`)}</select><//>
      <${Field} id=${ids.chance} label=${t('simple.chance')}>
        <${Slider} id=${ids.chance} value=${Math.round((r.chance ?? 1) * 100)} step=${5} format=${(v) => `${v}%`}
          onInput=${(v) => set({ chance: v / 100 })} /><//>
      <${Field} id=${ids.cool} label=${t('simple.cooldown')} error=${bad('cooldown_s') && err.message}>
        <div class="with-unit"><input type="number" id=${ids.cool} min="0" step="1" value=${r.cooldown_s ?? ''}
          placeholder=${t('simple.cooldownPlaceholder', { n: data.settings.effective['gags.cooldown_s'] })}
          onInput=${(e) => set({ cooldown_s: e.target.value === '' ? null : Number(e.target.value) })} /><span>${t('units.seconds')}</span></div><//>
      <${Field} id=${ids.voice} label=${t('advanced.defaultVoice')} hint=${t('advanced.defaultVoiceHint')}>
        <${VoicePicker} id=${ids.voice} data=${data} value=${r.voice_id} invalid=${bad('voice_id')}
          inheritLabel=${t('pickers.serverDefault')} onChange=${(v) => set({ voice_id: v })} /><//>
      <${Field} id=${ids.by} label=${t('advanced.onlyFor')} hint=${t('advanced.onlyForHint')} wide>
        <div class="row">
          <${Chips} id=${ids.by} values=${byIds.map((u) => (names.get(u) ? `${names.get(u)} · ${u}` : u))}
            placeholder=${t('advanced.onlyForPlaceholder')} invalid=${bad('by_users')}
            onChange=${(vals) => set({ by_users: vals.map((v) => (v.includes(' · ') ? v.split(' · ').pop() : v)) })} />
          <${PersonPicker} data=${data} value="" anyLabel=${t('advanced.addPerson')}
            onChange=${(u) => u && !byIds.includes(u) && set({ by_users: [...byIds, u] })} />
        </div><//>
      <div class="field"><span class="field-label">${t('advanced.enabled')}</span>
        <${Toggle} checked=${r.enabled} label=${r.enabled ? t('advanced.on') : t('advanced.off')} onChange=${(v) => set({ enabled: v })} /></div>
    </div>

    <h3>${t('advanced.triggers')} <span class="hint">${t('advanced.triggersHint')}</span></h3>
    ${r.triggers.map((tr, i) => html`<div key=${i} class="block ${bad(`triggers.${i}`) ? 'invalid' : ''}">
      <${TriggerEditor} tr=${tr} data=${data} onChange=${(n) => setTrigger(i, n)} />
      <button type="button" class="btn icon ghost" aria-label=${t('advanced.removeTrigger')} disabled=${r.triggers.length === 1}
        onClick=${() => set({ triggers: r.triggers.filter((_, j) => j !== i) })}><${Icon} name="close" size=${16} /></button>
      ${bad(`triggers.${i}`) && html`<span class="field-error block-error">${err.message}</span>`}
    </div>`)}
    <button type="button" class="btn small" onClick=${() => set({ triggers: [...r.triggers, defaultTrigger('phrase')] })}>
      <${Icon} name="plus" size=${14} />${t('advanced.addTrigger')}</button>

    <h3>${t('advanced.options')} <span class="hint">${t('advanced.optionsHint')}</span></h3>
    <${Placeholders} list=${placeholdersFor(r.triggers)} />
    ${r.options.map((o, i) => html`<div key=${i} class="block option ${bad(`options.${i}`) ? 'invalid' : ''}">
      <div class="option-steps">
        <span class="eyebrow">${t('advanced.optionN', { n: i + 1 })}</span>
        ${o.map((s, k) => html`<div key=${k} class="step">
          <${StepEditor} s=${s} data=${data} onChange=${(ns) => setOption(i, o.map((x, j) => (j === k ? ns : x)))} />
          <button type="button" class="btn icon ghost" aria-label=${t('advanced.removeStep')} disabled=${o.length === 1}
            onClick=${() => setOption(i, o.filter((_, j) => j !== k))}><${Icon} name="close" size=${14} /></button>
        </div>`)}
        <button type="button" class="btn small ghost" onClick=${() => setOption(i, [...o, { type: 'say', text: '' }])}>
          <${Icon} name="plus" size=${14} />${t('advanced.addStep')}</button>
        ${bad(`options.${i}`) && html`<span class="field-error">${err.message}</span>`}
      </div>
      <button type="button" class="btn icon ghost" aria-label=${t('advanced.removeOption')} disabled=${r.options.length === 1}
        onClick=${() => set({ options: r.options.filter((_, j) => j !== i) })}><${Icon} name="close" size=${16} /></button>
    </div>`)}
    <button type="button" class="btn small" onClick=${() => set({ options: [...r.options, [{ type: 'say', text: '' }]] })}>
      <${Icon} name="plus" size=${14} />${t('advanced.addOption')}</button>
  </div>`;
}

// ---------------------------------------------------------------- editor frame

function Editor({ data, gid, id, initial, onDone, onCancel }) {
  const [r, setR] = useState(() => clone(initial));
  const simpleOk = isSimple(r);
  const [advanced, setAdvanced] = useState(() => !isSimple(initial) || load('heckler.gags.advanced', false));
  const [err, setErr] = useState(null);
  const [saving, setSaving] = useState(false);
  const top = useRef(null);
  useEffect(() => { if (top.current) top.current.scrollIntoView({ block: 'start' }); }, []);

  const submit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setErr(null);
    let body = r;
    if (!advanced && r.kind === 'gag') {
      if (!r.triggers[0].phrases.length) {
        setErr({ message: t('simple.needWords'), field: 'triggers.0' });
        setSaving(false);
        return;
      }
      if (!r.options.some((o) => o[0].text.trim())) {
        setErr({ message: t('simple.needReply'), field: 'options.0' });
        setSaving(false);
        return;
      }
      body = { ...r, name: r.name.trim() || (r.triggers[0].phrases[0] || ''),
        options: r.options.filter((o) => o[0].text.trim()).length ? r.options.filter((o) => o[0].text.trim()) : r.options };
    }
    try {
      await saveReaction(gid, id, body);
      toast(id === null ? t('gags.added', { name: body.name }) : t('gags.saved', { name: body.name }));
      await onDone();
    } catch (error) {
      setErr({ message: error.message, field: error.field });
    } finally {
      setSaving(false);
    }
  };

  return html`<form class="card editor" ref=${top} onSubmit=${submit}
    onKeyDown=${(e) => { if (e.key === 'Escape') onCancel(); }}>
    <div class="card-head">
      <div>
        <h2>${id === null ? t(`kinds.${r.kind}.new`) : t('editor.edit', { name: initial.name })}</h2>
        <p class="sub">${advanced ? t('editor.advancedSub') : t('editor.simpleSub')}</p>
      </div>
      ${r.kind === 'gag' && html`<${Toggle} label=${t('editor.advanced')} checked=${advanced} disabled=${advanced && !simpleOk}
        hint=${advanced && !simpleOk ? t('editor.advancedOnly') : ''}
        onChange=${(v) => { setAdvanced(v); save('heckler.gags.advanced', v); }} />`}
    </div>
    ${err && !err.field && html`<p class="error-line" role="alert"><${Icon} name="warn" size=${16} />${err.message}</p>`}
    ${advanced || r.kind !== 'gag' ? html`<${AdvancedEditor} r=${r} setR=${setR} data=${data} err=${err} />`
      : html`<${SimpleEditor} r=${r} setR=${setR} data=${data} err=${err} />`}
    <div class="form-actions">
      <${Toggle} label=${t('editor.enabled')} checked=${r.enabled} onChange=${(v) => setR({ ...r, enabled: v })} />
      <span class="spacer"></span>
      <button type="button" class="btn ghost" onClick=${onCancel}>${t('common.cancel')}</button>
      <button type="submit" class="btn primary" disabled=${saving}>${saving ? t('common.saving') : t('common.save')}</button>
    </div>
  </form>`;
}

// ---------------------------------------------------------------- approvals

function Approvals({ data, gid, reload }) {
  const [run, busy] = useRunner();
  const pending = data.reactions.filter((r) => r.status === 'pending');
  const decide = (r, status) => run(async () => {
    await api('PATCH', `/api/g/${gid}/reactions/${r.id}`, { status });
    await reload();
  }, t(status === 'approved' ? 'approvals.approved' : 'approvals.rejected', { name: r.name }));
  return html`<section class="card">
    <div class="card-head"><div><h2>${t('approvals.title')}</h2><p class="sub">${t('approvals.desc')}</p></div></div>
    ${pending.length === 0 ? html`<${Empty} title=${t('approvals.emptyTitle')} body=${t('approvals.emptyBody')} />` : html`<ul class="plain">
      ${pending.map((r) => html`<li key=${r.id} class="approval">
        <div class="grow">
          <div><b>${r.name}</b> <span class="tag">${t(`kinds.${r.kind}.one`)}</span>
            <span class="muted small">${t('approvals.byOn', { name: r.creator, date: when(r.created_at) })}</span></div>
          <p class="small">${r.triggers.map((tr) => triggerText(tr, peopleNames(data))).join(' · ')}
            <span class="muted"> → </span>${r.options.map((o) => optionText(o, data)).join(' | ')}</p>
        </div>
        <button class="btn primary" disabled=${busy} onClick=${() => decide(r, 'approved')}>${t('approvals.approve')}</button>
        <button class="btn" disabled=${busy} onClick=${() => decide(r, 'rejected')}>${t('approvals.reject')}</button>
      </li>`)}
    </ul>`}
  </section>`;
}

// ---------------------------------------------------------------- test bench

function Bench({ data, gid, reload }) {
  const [text, setText] = useState('');
  const [user, setUser] = useState('');
  const [result, setResult] = useState(null);
  const [run, busy] = useRunner();
  const ids = { text: useId('bt'), who: useId('bw') };
  if (!data.meta.test_bench) {
    return html`<section class="card"><${Empty} title=${t('bench.title')} body=${t('bench.unavailable')} /></section>`;
  }
  const test = (e) => {
    if (e) e.preventDefault();
    if (!text.trim()) return;
    run(async () => setResult(await api('POST', `/api/g/${gid}/test`, { text, user_id: user || null })));
  };
  const addPhrase = (item) => run(async () => {
    await api('POST', `/api/g/${gid}/reactions/${item.reaction_id}/phrase`, { trigger: item.trigger, phrase: item.heard });
    await reload();
    test();
  }, t('bench.addedPhrase', { phrase: item.heard }));
  const results = result?.results || [];
  const matches = results.filter((x) => !x.near_miss);
  const best = new Map();
  for (const x of results.filter((r) => r.near_miss)) {
    const key = `${x.reaction_id}|${JSON.stringify(x.trigger)}|${x.heard}`;
    if (!best.has(key) || (x.similarity ?? 0) > (best.get(key).similarity ?? 0)) best.set(key, x);
  }
  const near = [...best.values()];
  const firing = matches.find((x) => x.would_fire);

  return html`<section class="card">
    <div class="card-head"><div><h2>${t('bench.title')}</h2><p class="sub">${t('bench.desc')}</p></div></div>
    <form class="bench" onSubmit=${test}>
      <label class="visually-hidden" for=${ids.text}>${t('bench.sentence')}</label>
      <input type="text" id=${ids.text} class="grow" placeholder=${t('bench.placeholder')} value=${text} onInput=${(e) => setText(e.target.value)} />
      <label class="visually-hidden" for=${ids.who}>${t('bench.who')}</label>
      <${PersonPicker} id=${ids.who} data=${data} value=${user} anyLabel=${t('bench.anyone')} onChange=${(u) => setUser(u || '')} />
      <button class="btn primary" type="submit" disabled=${busy || !text.trim()}>${t('bench.test')}</button>
    </form>
    ${result && html`<div class="bench-out" aria-live="polite">
      <p class="verdict ${firing ? 'ok' : ''}">${firing ? t('bench.fires', { name: firing.name }) : t('bench.nothing')}</p>
      ${matches.length > 0 && html`<ul class="plain">${matches.map((x, i) => html`<li key=${i} class="bench-row">
        <span class="tag ${x.would_fire ? 'ok' : 'warn'}">${x.would_fire ? t('bench.wouldFire') : t('bench.blocked')}</span>
        <b>${x.name}</b>
        ${x.trigger && html`<span class="muted small">${triggerText(x.trigger, peopleNames(data))}</span>`}
        ${x.reason && html`<span class="small">${t(`bench.reasons.${x.reason}`) || x.reason}</span>`}
        ${x.chance !== undefined && x.chance < 1 && html`<span class="tag">${t('gags.chanceTag', { n: Math.round(x.chance * 100) })}</span>`}
      </li>`)}</ul>`}
      ${near.length > 0 && html`<h3>${t('bench.nearMisses')}</h3>
        <ul class="plain">${near.map((x, i) => html`<li key=${i} class="bench-row">
          <span>${t('bench.heardClose', { heard: x.heard, word: x.trigger_word, name: x.name })}</span>
          <button class="btn small" disabled=${busy} onClick=${() => addPhrase(x)}>${t('bench.addPhrase', { phrase: x.heard })}</button>
        </li>`)}</ul>`}
    </div>`}
  </section>`;
}

// ---------------------------------------------------------------- page

export function Gags({ data, gid, guild, missing, reload }) {
  const [tab, setTab] = useState(() => load('heckler.gags.tab', 'gag'));
  const [editing, setEditing] = useState(null); // {id, draft}
  useEffect(() => save('heckler.gags.tab', tab), [tab]);
  useEffect(() => setEditing(null), [gid]);
  if (!data) return html`<${Waiting} section="gags" missing=${missing} guild=${guild} />`;

  const pending = data.reactions.filter((r) => r.status === 'pending').length;
  const count = (k) => data.reactions.filter((r) => r.kind === k).length;
  const items = [
    ...KINDS.map((k) => [k, `${t(`kinds.${k}.tab`)}`, null]),
    ['approvals', t('approvals.tab'), pending],
    ['bench', t('bench.tab'), null],
  ];
  return html`<div class="page">
    <${PageHead} section="gags" />
    <${Tabs} label=${t('page.gags.title')} value=${tab} onChange=${(k) => { setTab(k); setEditing(null); }}
      items=${items.map(([k, label, badge]) => [k, KINDS.includes(k) ? html`${label} <span class="count">${count(k)}</span>` : label, badge])} />
    ${editing ? html`<${Editor} key=${editing.id ?? 'new'} data=${data} gid=${gid} id=${editing.id} initial=${editing.draft}
        onDone=${async () => { setEditing(null); await reload(); }} onCancel=${() => setEditing(null)} />`
      : KINDS.includes(tab) ? html`<${ReactionList} kind=${tab} data=${data} gid=${gid} reload=${reload}
          edit=${(id, draft) => setEditing({ id, draft })} />`
        : tab === 'approvals' ? html`<${Approvals} data=${data} gid=${gid} reload=${reload} />`
          : html`<${Bench} data=${data} gid=${gid} reload=${reload} />`}
  </div>`;
}

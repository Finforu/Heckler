// Settings: plain-language groups for the selected server, with every raw key
// (and the global defaults) under "Advanced".
import { useState } from './vendor/hooks.js';
import {
  api, Chips, Commit, html, Icon, load, PageHead, S, save, Slider, t, Toggle, toast, useId, useRunner, Waiting,
} from './lib.js';
import { privacyKey } from './overview.js';
import { voiceLabel, VoicePicker } from './pickers.js';

function languageName(code) {
  try {
    const name = new Intl.DisplayNames([navigator.language || 'en', 'en'], { type: 'language' }).of(code);
    return name && name !== code ? `${name} (${code})` : code;
  } catch { return code; }
}

function useSetting(data, gid, reload) {
  const [run, busy] = useRunner();
  const put = (key, value, target = gid) => run(async () => {
    await api('PUT', `/api/g/${target}/settings/${key}`, { value });
    await reload();
  }, t('common.saved'));
  const reset = (key, target = gid) => run(async () => {
    await api('DELETE', `/api/g/${target}/settings/${key}`);
    await reload();
  }, t('settings.resetDone'));
  return { put, reset, busy };
}

// One setting: label, help, the control, and whether this server overrides the default.
function Setting({ k, data, s, label, hint, children }) {
  const own = k in data.settings.own;
  const id = `set-${k.replace(/\W/g, '-')}`;
  return html`<div class="setting">
    <div class="setting-text">
      <label class="setting-label" for=${id}>${label}</label>
      ${hint && html`<p class="hint">${hint}</p>`}
    </div>
    <div class="setting-control">
      ${children(id)}
      <span class="setting-source">${own
        ? html`<span class="tag accent">${t('settings.thisServer')}</span>
            <button class="btn link" disabled=${s.busy} onClick=${() => s.reset(k)}>${t('settings.reset')}</button>`
        : html`<span class="muted small">${t('settings.default')}</span>`}</span>
    </div>
  </div>`;
}

function Group({ title, desc, children, id }) {
  return html`<section class="card settings-group" id=${id} aria-labelledby=${`${id}-title`}>
    <div class="card-head"><div><h2 id=${`${id}-title`}>${title}</h2>${desc && html`<p class="sub">${desc}</p>`}</div></div>
    <div class="settings-list">${children}</div>
  </section>`;
}

function NumberSetting({ k, data, s, unit, min = 0, step = 1 }) {
  return (id) => html`<div class="with-unit"><${Commit} id=${id} type="number" min=${min} step=${step}
    value=${data.settings.effective[k]} onCommit=${(v) => v !== null && s.put(k, v)} />${unit && html`<span>${unit}</span>`}</div>`;
}

// A channel of this server the bot can post in (ids are strings), or the default.
function ChannelPicker({ id, value, channels, defaultLabel, onChange }) {
  const list = channels || [];
  const current = S(value);
  const gone = current && !list.some((c) => c.id === current);
  return html`<select id=${id} value=${current} onChange=${(e) => onChange(e.target.value || null)}>
    <option value="">${defaultLabel}</option>
    ${gone && html`<option value=${current}>${t('settings.missingChannel', { id: current })}</option>`}
    ${list.map((c) => html`<option key=${c.id} value=${c.id}>${c.kind === 'voice' ? t('settings.voiceChat', { name: c.name }) : `#${c.name}`}</option>`)}
  </select>`;
}

function Llm({ data, s, status }) {
  const e = data.settings.effective;
  const llm = status?.bot?.llm;
  return html`<${Group} id="llm" title=${t('settings.llm.title')} desc=${t('settings.llm.desc')}>
    ${llm ? html`<p class="hint llm-note">${t('settings.llm.using', { model: llm.model, service: llm.service })}
        ${' '}${llm.cloud ? t('settings.llm.cloudNote', { service: llm.service }) : t('settings.llm.localNote')}</p>`
      : html`<p class="hint llm-note">${t('settings.llm.notConfigured')}</p>`}
    <${Setting} k="llm.enabled" data=${data} s=${s} label=${t('settings.llmEnabled')} hint=${t('settings.llmEnabledHint')}>
      ${(id) => html`<${Toggle} id=${id} checked=${e['llm.enabled']} disabled=${!llm} onChange=${(v) => s.put('llm.enabled', v)} />`}<//>
    <${Setting} k="llm.persona" data=${data} s=${s} label=${t('settings.llmPersona')} hint=${t('settings.llmPersonaHint')}>
      ${(id) => html`<${Commit} id=${id} value=${e['llm.persona']} placeholder=${t('settings.llmPersonaPlaceholder')}
        onCommit=${(v) => s.put('llm.persona', v)} />`}<//>
    <${Setting} k="llm.cooldown_s" data=${data} s=${s} label=${t('settings.llmCooldown')} hint=${t('settings.llmCooldownHint')}>
      ${NumberSetting({ k: 'llm.cooldown_s', data, s, unit: t('units.seconds') })}<//>
  <//>`;
}

function Privacy({ data, gid, guild, control, busy, go }) {
  const on = !!(guild?.toggles?.transcripts ?? data.settings.effective['transcripts.enabled']);
  const [reviewed, setReviewed] = useState(() => !!load(privacyKey(gid), false));
  const [confirming, setConfirming] = useState(false);
  const flip = async (value) => {
    setConfirming(false);
    await control('toggle', { guild_id: gid, name: 'transcripts', value });
  };
  return html`<section class="card settings-group privacy" id="privacy" aria-labelledby="privacy-title">
    <div class="card-head"><div><h2 id="privacy-title">${t('settings.privacy.title')}</h2><p class="sub">${t('settings.privacy.desc')}</p></div></div>
    <div class="setting">
      <div class="setting-text">
        <span class="setting-label">${t('settings.privacy.transcripts')}</span>
        <p class="hint">${on ? t('settings.privacy.transcriptsOnHint') : t('settings.privacy.transcriptsOffHint')}</p>
      </div>
      <div class="setting-control">
        ${confirming ? html`<div class="confirm-box" role="alertdialog" aria-labelledby="tr-confirm">
            <p id="tr-confirm">${t('settings.privacy.confirmOn')}</p>
            <div class="row"><button class="btn primary" onClick=${() => flip(true)}>${t('settings.privacy.turnOn')}</button>
              <button class="btn ghost" onClick=${() => setConfirming(false)}>${t('common.cancel')}</button></div>
          </div>`
          : html`<${Toggle} checked=${on} disabled=${busy[`toggle:${gid}`]} label=${on ? t('settings.privacy.on') : t('settings.privacy.off')}
              onChange=${(v) => (v ? setConfirming(true) : flip(false))} />`}
      </div>
    </div>
    <ul class="kept small">
      <li><${Icon} name="info" size=${16} /><span>${t('settings.privacy.point1')}</span></li>
      <li><${Icon} name="info" size=${16} /><span>${t('settings.privacy.point2')}</span></li>
      <li><${Icon} name="info" size=${16} /><span>${t('settings.privacy.point3')}</span></li>
    </ul>
    <div class="card-actions">
      <button class="btn" onClick=${() => go('people')}>${t('settings.privacy.people')}</button>
      ${reviewed ? html`<span class="tag ok"><${Icon} name="check" size=${14} />${t('settings.privacy.reviewed')}</span>`
        : html`<button class="btn primary" onClick=${() => { save(privacyKey(gid), true); setReviewed(true); toast(t('settings.privacy.reviewedToast')); }}>
            ${t('settings.privacy.markReviewed')}</button>`}
    </div>
  </section>`;
}

// ---------------------------------------------------------------- advanced (raw keys)

function show(value, data, key) {
  if (key === 'voice.default' || key === 'voice.bot') return value === null || value === undefined ? t('settings.botVoice') : voiceLabel(data, value);
  if (value === null || value === undefined || value === '') return '—';
  if (Array.isArray(value)) return value.length ? value.join(', ') : t('settings.none');
  if (typeof value === 'boolean') return value ? t('settings.on') : t('settings.off');
  return String(value);
}

function RawEditor({ k, value, fallback, data, onSave }) {
  const def = data.settings.defaults[k];
  if (data.meta.controller_settings?.includes(k)) return html`<span class="muted small">${t('settings.viaToggle')}</span>`;
  if (k === 'voice.default' || k === 'voice.bot') {
    return html`<${VoicePicker} data=${data} value=${value} speaker=${k === 'voice.default'} globalOnly=${k === 'voice.bot'}
      inheritLabel=${k === 'voice.bot' ? t('settings.notSet') : t('settings.botVoice')} onChange=${onSave} />`;
  }
  if (typeof def === 'boolean') {
    return html`<input type="checkbox" role="switch" class="switch" aria-label=${k} checked=${!!(value ?? fallback)} onChange=${(e) => onSave(e.target.checked)} />`;
  }
  if (typeof def === 'number') return html`<${Commit} type="number" cls="num" value=${value} placeholder=${S(fallback)} label=${k} onCommit=${(v) => v !== null && onSave(v)} />`;
  if (Array.isArray(def)) return html`<${Chips} values=${value ?? []} label=${k} placeholder=${(fallback || []).join(', ') || t('settings.add')} onChange=${onSave} />`;
  return html`<${Commit} value=${value} placeholder=${S(fallback) || t('settings.defaultPlaceholder')} label=${k} onCommit=${onSave} />`;
}

function Advanced({ data, gid, s }) {
  const { defaults, own, global, effective } = data.settings;
  const globalOnly = data.meta.global_only_settings || [];
  return html`<details class="card advanced-settings">
    <summary><span><b>${t('settings.advanced.title')}</b><span class="hint">${t('settings.advanced.desc')}</span></span></summary>
    <div class="table-wrap"><table class="table settings-table">
      <thead><tr><th>${t('settings.advanced.key')}</th><th>${t('settings.advanced.effective')}</th>
        <th>${t('settings.advanced.server')}</th><th>${t('settings.advanced.global')}</th></tr></thead>
      <tbody>${Object.keys(defaults).map((k) => {
        const isOwn = k in own, isGlobal = k in global;
        return html`<tr key=${k}>
          <td><code>${k}</code></td>
          <td>${show(effective[k], data, k)}<div><span class="tag ${isOwn ? 'accent' : ''}">${isOwn ? t('settings.thisServer') : isGlobal ? t('settings.fromGlobal') : t('settings.builtIn')}</span></div></td>
          <td>${globalOnly.includes(k) ? html`<span class="muted small">${t('settings.globalOnly')}</span>` : html`<div class="row">
            <${RawEditor} k=${k} value=${isOwn ? own[k] : null} fallback=${isGlobal ? global[k] : defaults[k]} data=${data} onSave=${(v) => s.put(k, v)} />
            ${isOwn && html`<button class="btn link" onClick=${() => s.reset(k)}>${t('settings.reset')}</button>`}</div>`}</td>
          <td><div class="row">
            <${RawEditor} k=${k} value=${isGlobal ? global[k] : null} fallback=${defaults[k]} data=${data} onSave=${(v) => s.put(k, v, 0)} />
            ${isGlobal && html`<button class="btn link" onClick=${() => s.reset(k, 0)}>${t('settings.reset')}</button>`}</div></td>
        </tr>`;
      })}</tbody>
    </table></div>
  </details>`;
}

// ---------------------------------------------------------------- page

export function Settings({ data, gid, guild, missing, reload, control, busy, go, status }) {
  const s = useSetting(data, gid, reload);
  const intensityId = useId('int');
  const [intensity, setIntensity] = useState(null);
  const soundCapId = useId('cap');
  const [soundCap, setSoundCap] = useState(null);
  if (!data) return html`<${Waiting} section="settings" missing=${missing} guild=${guild} />`;
  const e = data.settings.effective;
  const langs = data.meta.languages || [];
  const packs = data.meta.starter_packs || [];
  const known = (k) => k in data.settings.defaults;

  return html`<div class="page">
    <${PageHead} section="settings" desc=${t('page.settings.descFor', { server: guild?.name || '' })} />

    <${Group} id="general" title=${t('settings.general.title')} desc=${t('settings.general.desc')}>
      ${known('language') && html`<${Setting} k="language" data=${data} s=${s} label=${t('settings.language')} hint=${t('settings.languageHint')}>
        ${(id) => html`<select id=${id} value=${S(e.language)} onChange=${(ev) => s.put('language', ev.target.value || null)}>
          <option value="">${t('settings.languageDefault', { lang: languageName(data.meta.default_language || 'en') })}</option>
          ${langs.map((l) => html`<option key=${l} value=${l}>${languageName(l)}</option>`)}
        </select>`}<//>`}
      <${Setting} k="bot.name" data=${data} s=${s} label=${t('settings.botName')} hint=${t('settings.botNameHint')}>
        ${(id) => html`<${Commit} id=${id} value=${data.settings.own['bot.name'] ?? ''} placeholder=${status?.bot?.name || ''}
          onCommit=${(v) => s.put('bot.name', v || null)} />`}<//>
      <${Setting} k="bot.wake_words" data=${data} s=${s} label=${t('settings.wakeWords')} hint=${t('settings.wakeWordsHint')}>
        ${(id) => html`<${Chips} id=${id} values=${e['bot.wake_words'] || []} placeholder=${t('settings.wakeWordsPlaceholder')}
          onChange=${(v) => s.put('bot.wake_words', v)} />`}<//>
    <//>

    <${Privacy} data=${data} gid=${gid} guild=${guild} control=${control} busy=${busy} go=${go} />

    <${Group} id="gags" title=${t('settings.gags.title')} desc=${t('settings.gags.desc')}>
      <${Setting} k="gags.cooldown_s" data=${data} s=${s} label=${t('settings.cooldown')} hint=${t('settings.cooldownHint')}>
        ${NumberSetting({ k: 'gags.cooldown_s', data, s, unit: t('units.seconds') })}<//>
      <${Setting} k="gags.intensity" data=${data} s=${s} label=${t('settings.intensity')} hint=${t('settings.intensityHint')}>
        ${() => html`<${Slider} id=${intensityId} min=${0} max=${200} step=${10} value=${intensity ?? Math.round((e['gags.intensity'] ?? 1) * 100)}
          format=${(v) => `${v}%`} onInput=${setIntensity} onChange=${(v) => { setIntensity(null); s.put('gags.intensity', v / 100); }} />`}<//>
      <${Setting} k="user_content.needs_approval" data=${data} s=${s} label=${t('settings.approval')} hint=${t('settings.approvalHint')}>
        ${(id) => html`<${Toggle} id=${id} checked=${e['user_content.needs_approval']} onChange=${(v) => s.put('user_content.needs_approval', v)} />`}<//>
    <//>

    <${Group} id="quotas" title=${t('settings.quotas.title')} desc=${t('settings.quotas.desc')}>
      ${['gags', 'sounds', 'voices'].map((r) => html`<${Setting} key=${r} k=${`quota.${r}`} data=${data} s=${s}
        label=${t(`settings.quota.${r}`)} hint=${t(`settings.quota.${r}Hint`)}>
        ${NumberSetting({ k: `quota.${r}`, data, s, unit: t('settings.perPerson') })}<//>`)}
    <//>

    <${Group} id="media" title=${t('settings.media.title')} desc=${t('settings.media.desc')}>
      <${Setting} k="voice.default" data=${data} s=${s} label=${t('settings.defaultVoice')} hint=${t('settings.defaultVoiceHint')}>
        ${(id) => html`<${VoicePicker} id=${id} data=${data} value=${e['voice.default']} inheritLabel=${t('settings.botVoice')}
          onChange=${(v) => s.put('voice.default', v)} />`}<//>
      ${known('sounds.max_volume') && html`<${Setting} k="sounds.max_volume" data=${data} s=${s} label=${t('settings.soundMaxVolume')} hint=${t('settings.soundMaxVolumeHint')}>
        ${() => html`<${Slider} id=${soundCapId} min=${0} max=${400} step=${10} value=${soundCap ?? Math.round(e['sounds.max_volume'] ?? 100)}
          format=${(v) => `${v}%`} onInput=${setSoundCap} onChange=${(v) => { setSoundCap(null); s.put('sounds.max_volume', v); }} />`}<//>`}
      <${Setting} k="sounds.max_seconds" data=${data} s=${s} label=${t('settings.soundSeconds')} hint=${t('settings.soundSecondsHint')}>
        ${NumberSetting({ k: 'sounds.max_seconds', data, s, unit: t('units.seconds'), min: 1 })}<//>
      <${Setting} k="sounds.max_mb" data=${data} s=${s} label=${t('settings.soundMb')}>
        ${NumberSetting({ k: 'sounds.max_mb', data, s, unit: 'MB', min: 1 })}<//>
      <${Setting} k="voices.max_mb" data=${data} s=${s} label=${t('settings.voiceMb')}>
        ${NumberSetting({ k: 'voices.max_mb', data, s, unit: 'MB', min: 1 })}<//>
      <${Setting} k="voices.preview_text" data=${data} s=${s} label=${t('settings.previewText')} hint=${t('settings.previewTextHint')}>
        ${(id) => html`<${Commit} id=${id} value=${e['voices.preview_text']} onCommit=${(v) => s.put('voices.preview_text', v)} />`}<//>
    <//>

    ${known('notices.channel_id') && html`<${Group} id="messages" title=${t('settings.messages.title')} desc=${t('settings.messages.desc')}>
      <${Setting} k="notices.channel_id" data=${data} s=${s} label=${t('settings.noticesChannel')} hint=${t('settings.noticesChannelHint')}>
        ${(id) => html`<${ChannelPicker} id=${id} value=${data.settings.own['notices.channel_id']} channels=${guild?.text_channels}
          defaultLabel=${t('settings.noticesDefault')} onChange=${(v) => s.put('notices.channel_id', v)} />`}<//>
      <${Setting} k="time.zone" data=${data} s=${s} label=${t('settings.timeZone')}
        hint=${t('settings.timeZoneHint', { zone: status?.bot?.time_zone || '?' })}>
        ${(id) => html`<${Commit} id=${id} value=${data.settings.own['time.zone'] ?? ''} placeholder=${status?.bot?.time_zone || 'Europe/Madrid'}
          onCommit=${(v) => s.put('time.zone', v || null)} />`}<//>
    <//>`}

    ${known('llm.enabled') && html`<${Llm} data=${data} s=${s} status=${status} />`}

    ${known('content.starter_pack') && html`<${Group} id="starter" title=${t('settings.starter.title')} desc=${t('settings.starter.desc')}>
      <${Setting} k="content.starter_pack" data=${data} s=${s} label=${t('settings.starterPack')} hint=${t('settings.starterPackHint')}>
        ${(id) => html`<select id=${id} value=${S(e['content.starter_pack'])} onChange=${(ev) => s.put('content.starter_pack', ev.target.value || null)}>
          <option value="">${t('settings.starterAuto')}</option>
          ${packs.map((p) => html`<option key=${p} value=${p}>${p}</option>`)}
          <option value="none">${t('settings.starterNone')}</option>
        </select>`}<//>
    <//>`}

    <${Group} id="admins" title=${t('settings.admins.title')} desc=${t('settings.admins.desc')}>
      <${Setting} k="admin.role_id" data=${data} s=${s} label=${t('settings.adminRole')} hint=${t('settings.adminRoleHint')}>
        ${(id) => html`<${Commit} id=${id} value=${data.settings.own['admin.role_id'] ?? ''} placeholder=${t('settings.idPlaceholder')}
          onCommit=${(v) => s.put('admin.role_id', v || null)} />`}<//>
      <${Setting} k="admin.channel_id" data=${data} s=${s} label=${t('settings.adminChannel')} hint=${t('settings.adminChannelHint')}>
        ${(id) => html`<${ChannelPicker} id=${id} value=${data.settings.own['admin.channel_id']} channels=${guild?.text_channels}
          defaultLabel=${t('settings.adminChannelDefault')} onChange=${(v) => s.put('admin.channel_id', v)} />`}<//>
    <//>

    <${Advanced} data=${data} gid=${gid} s=${s} />
  </div>`;
}

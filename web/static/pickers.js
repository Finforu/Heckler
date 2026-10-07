// Voice and person pickers shared by the editors.
import { html, S, t } from './lib.js';

export function voiceState(v) {
  if (v.status === 'ready') return '';
  if (v.status === 'queued' || v.status === 'building') return t('voices.stateBuilding');
  if (v.status === 'failed') return t('voices.stateFailed');
  return t('voices.stateNotBuilt');
}

export function voiceLabel(data, id) {
  if (id === null || id === undefined || id === '') return t('pickers.inherit');
  if (id === data.meta.speaker) return t('pickers.speakerShort');
  const v = data.voices.find((x) => S(x.id) === S(id));
  return v ? v.name : `#${id}`;
}

// inheritLabel: what "no choice" means here. speaker: offer the speaker's own voice.
// globalOnly: only voices every server shares (the bot's own voice setting).
export function VoicePicker({ data, value, onChange, id, inheritLabel, speaker = true, globalOnly = false, invalid }) {
  const botVoice = S(data.settings.global['voice.bot']);
  return html`<select id=${id} class=${invalid ? 'invalid' : ''} value=${S(value)} onChange=${(e) => onChange(e.target.value || null)}>
    <option value="">${inheritLabel || t('pickers.inherit')}</option>
    ${speaker && html`<option value=${data.meta.speaker}>${t('pickers.speaker')}</option>`}
    ${data.voices.filter((v) => v.kind !== 'speaker' && (!globalOnly || v.guild_id === null)).map((v) => html`<option key=${v.id} value=${S(v.id)}>
      ${v.name}${S(v.id) === botVoice ? t('pickers.botVoiceSuffix') : ''}${voiceState(v)}</option>`)}
  </select>`;
}

export function PersonPicker({ data, value, onChange, id, anyLabel }) {
  return html`<select id=${id} value=${S(value)} onChange=${(e) => onChange(e.target.value || null)}>
    <option value="">${anyLabel || t('pickers.anyone')}</option>
    ${data.people.map((p) => html`<option key=${S(p.user_id)} value=${S(p.user_id)}>
      ${p.nickname || p.display_name || S(p.user_id)}</option>`)}
  </select>`;
}

export function peopleNames(data) {
  const m = new Map();
  for (const p of data.people) m.set(S(p.user_id), p.nickname || p.display_name || S(p.user_id));
  return m;
}

// People & privacy: nicknames, both kinds of consent, deleting someone's data,
// and quotas with the requests people send from Discord.
import { useEffect, useState } from './vendor/hooks.js';
import {
  api, Commit, ConfirmButton, Empty, html, Icon, load, num, PageHead, S, save, t, Tabs, useRunner, Waiting, when,
} from './lib.js';

const tone = (status) => (status === 'accepted' ? 'ok' : status === 'revoked' || status === 'declined' ? 'bad' : status === 'pending' ? 'warn' : 'muted');

function initials(name) {
  return (name || '?').replace(/[^\p{L}\p{N} ]/gu, '').trim().split(/\s+/).map((w) => w[0]).join('').slice(0, 2).toUpperCase() || '?';
}

function Person({ p, gid, purposes, reload, transcriptsOn }) {
  const [open, setOpen] = useState(false);
  const [run, busy] = useRunner();
  const name = p.nickname || p.display_name || S(p.user_id);
  const consent = p.consent || {};
  const act = (fn, okText) => run(async () => { const out = await fn(); await reload(); return out; }, okText);
  const panel = `person-${S(p.user_id)}`;
  return html`<li class="person">
    <div class="person-main">
      <span class="avatar" aria-hidden="true">${initials(name)}</span>
      <div class="person-names">
        <span class="person-name">${p.display_name || S(p.user_id)}</span>
        <span class="muted small">${p.last_seen ? t('people.lastSeen', { when: when(p.last_seen) }) : t('people.neverSeen')}</span>
      </div>
      <div class="person-nick">
        <label class="eyebrow" for=${`nick-${S(p.user_id)}`}>${t('people.nickname')}</label>
        <${Commit} id=${`nick-${S(p.user_id)}`} value=${p.nickname} placeholder=${t('people.nicknamePlaceholder')}
          onCommit=${(v) => act(() => api('PATCH', `/api/g/${gid}/people/${p.user_id}`, { nickname: v || null }), t('people.nicknameSaved'))} />
      </div>
      <div class="person-consent">
        ${purposes.map((purpose) => html`<span key=${purpose} class="consent-chip">
          <span class="eyebrow">${t(`people.purpose.${purpose}`)}</span>
          <span class="tag ${tone(consent[purpose])}">${t(`consent.${consent[purpose] || 'none'}`)}</span>
        </span>`)}
      </div>
      <button class="btn small" aria-expanded=${open} aria-controls=${panel} onClick=${() => setOpen(!open)}>
        ${t('people.manage')}<${Icon} name="chevron" size=${14} /></button>
    </div>
    ${open && html`<div class="person-actions" id=${panel}>
      ${purposes.map((purpose) => html`<div key=${purpose} class="action-row">
        <div class="grow"><b>${t(`people.revoke.${purpose}.title`)}</b><p class="hint">${t(`people.revoke.${purpose}.desc`)}</p></div>
        <${ConfirmButton} label=${t('people.revokeButton')} confirmLabel=${t('people.revokeConfirm')} cls="small"
          disabled=${busy || !['accepted', 'pending'].includes(consent[purpose])}
          onConfirm=${() => act(() => api('POST', `/api/g/${gid}/people/${p.user_id}/revoke-consent`, { purpose }),
            t('people.revoked', { name, purpose: t(`people.purpose.${purpose}`) }))} />
      </div>`)}
      <div class="action-row">
        <div class="grow"><b>${t('people.deleteText.title')}</b><p class="hint">${t('people.deleteText.desc')}</p></div>
        <${ConfirmButton} label=${t('people.deleteText.button')} confirmLabel=${t('people.deleteText.confirm')} cls="small" icon="trash"
          disabled=${busy} onConfirm=${() => act(async () => {
            const out = await api('POST', `/api/g/${gid}/people/${p.user_id}/delete-text`);
            return out;
          }, t('people.deleteText.done', { name }))} />
      </div>
      <div class="action-row">
        <div class="grow"><b>${t('people.deleteHistory.title')}</b><p class="hint">${t('people.deleteHistory.desc')}</p></div>
        <${ConfirmButton} label=${t('people.deleteHistory.button')} confirmLabel=${t('people.deleteHistory.confirm')} cls="small" icon="trash"
          disabled=${busy} onConfirm=${() => act(() => api('DELETE', `/api/g/${gid}/history/${p.user_id}`), t('people.deleteHistory.done', { name }))} />
      </div>
      ${!transcriptsOn && html`<p class="hint">${t('people.transcriptsOffNote')}</p>`}
    </div>`}
  </li>`;
}

function PeopleTab({ data, gid, reload, guild }) {
  const [query, setQuery] = useState('');
  const purposes = data.meta.consent_purposes || ['voice'];
  const transcriptsOn = !!(guild?.toggles?.transcripts ?? data.settings.effective['transcripts.enabled']);
  const q = query.trim().toLowerCase();
  const people = data.people.filter((p) => !q || `${p.display_name} ${p.nickname || ''}`.toLowerCase().includes(q));
  return html`<div class="stack">
    <section class="card privacy-summary">
      <h2>${t('people.keptTitle')}</h2>
      <ul class="kept">
        <li><${Icon} name="voices" size=${18} /><span>${t('people.keptVoice')}</span></li>
        <li><${Icon} name="live" size=${18} /><span>${transcriptsOn ? t('people.keptTextOn') : t('people.keptTextOff')}</span></li>
        <li><${Icon} name="stats" size=${18} /><span>${t('people.keptHistory')}</span></li>
        <li><${Icon} name="trash" size=${18} /><span>${t('people.keptDelete')}</span></li>
      </ul>
    </section>
    <section class="card">
      <div class="card-head">
        <div><h2>${t('people.listTitle', { n: data.people.length })}</h2><p class="sub">${t('people.listDesc')}</p></div>
        ${data.people.length > 6 && html`<div class="search"><${Icon} name="search" size=${16} />
          <input type="search" placeholder=${t('people.search')} aria-label=${t('people.search')} value=${query} onInput=${(e) => setQuery(e.target.value)} /></div>`}
      </div>
      ${data.people.length === 0 ? html`<${Empty} title=${t('people.emptyTitle')} body=${t('people.emptyBody')} />`
        : html`<ul class="people">${people.map((p) => html`<${Person} key=${S(p.user_id)} p=${p} gid=${gid} purposes=${purposes}
            reload=${reload} transcriptsOn=${transcriptsOn} />`)}</ul>`}
    </section>
  </div>`;
}

// ---------------------------------------------------------------- quotas

function QuotasTab({ data, gid, reload, go }) {
  const [run, busy] = useRunner();
  const [amounts, setAmounts] = useState({});
  const res = data.meta.quota_resources;
  const requests = data.requests.filter((q) => q.status === 'pending');
  const past = data.requests.filter((q) => q.status !== 'pending').slice(0, 10);
  const setOverride = (uid, r, v) => run(async () => { await api('PUT', `/api/g/${gid}/quota/${uid}/${r}`, { limit: v }); await reload(); }, t('common.saved'));
  const decide = (q, approve) => run(async () => {
    const amount = amounts[q.id] ?? q.amount;
    await api('POST', `/api/g/${gid}/quota-requests/${q.id}/decide`, { approve, amount: approve ? amount : undefined });
    await reload();
  }, approve ? t('quotas.approved', { name: q.name }) : t('quotas.denied', { name: q.name }));

  return html`<div class="stack">
    <section class="card">
      <div class="card-head"><div><h2>${t('quotas.requestsTitle')}</h2><p class="sub">${t('quotas.requestsDesc')}</p></div></div>
      ${requests.length === 0 ? html`<${Empty} title=${t('quotas.noRequests')} />` : html`<ul class="plain">
        ${requests.map((q) => html`<li key=${q.id} class="approval">
          <div class="grow">
            <div>${t('quotas.wants', { name: q.name, n: q.amount, what: t(`quotas.res.${q.resource}`) })}</div>
            <p class="muted small">${t('quotas.uses', { used: q.used, limit: q.limit })} · ${when(q.created_at)}</p>
            ${q.reason && html`<p class="quote">“${q.reason}”</p>`}
          </div>
          <label class="inline-label" for=${`grant-${q.id}`}>${t('quotas.grant')}</label>
          <input type="number" class="num" id=${`grant-${q.id}`} min="1" value=${amounts[q.id] ?? q.amount}
            onInput=${(e) => setAmounts({ ...amounts, [q.id]: Number(e.target.value) })} />
          <button class="btn primary" disabled=${busy} onClick=${() => decide(q, true)}>${t('quotas.approve')}</button>
          <button class="btn" disabled=${busy} onClick=${() => decide(q, false)}>${t('quotas.deny')}</button>
        </li>`)}
      </ul>`}
      ${past.length > 0 && html`<details class="past"><summary>${t('quotas.decided', { n: past.length })}</summary>
        <ul class="plain small">${past.map((q) => html`<li key=${q.id}>
          ${t('quotas.pastLine', { name: q.name, n: q.amount, what: t(`quotas.res.${q.resource}`) })}
          <span class="tag ${q.status === 'approved' ? 'ok' : 'bad'}">${t(`quotas.status.${q.status}`)}</span>
          <span class="muted">${when(q.decided_at)}</span></li>`)}</ul></details>`}
    </section>
    <section class="card">
      <div class="card-head">
        <div><h2>${t('quotas.limitsTitle')}</h2>
          <p class="sub">${t('quotas.limitsDesc', { gags: data.quotas.defaults.gags, sounds: data.quotas.defaults.sounds, voices: data.quotas.defaults.voices })}</p></div>
        <button class="btn small" onClick=${() => go('settings')}>${t('quotas.changeDefaults')}</button>
      </div>
      ${data.quotas.users.length === 0 ? html`<${Empty} title=${t('quotas.nobody')} /> ` : html`<div class="table-wrap"><table class="table">
        <thead><tr><th>${t('quotas.person')}</th>${res.map((r) => html`<th key=${r}>${t(`quotas.res.${r}`)}</th>`)}</tr></thead>
        <tbody>${data.quotas.users.map((u) => html`<tr key=${S(u.user_id)}>
          <td>${u.name}</td>
          ${res.map((r) => html`<td key=${r}><div class="quota-cell">
            <span class="tabular ${u[r].used >= u[r].limit ? 'warn-text' : ''}">${num(u[r].used)} / ${num(u[r].limit)}</span>
            <${Commit} type="number" cls="num" min="0" value=${u[r].override} placeholder=${t('quotas.default')}
              label=${t('quotas.overrideFor', { name: u.name, what: t(`quotas.res.${r}`) })} onCommit=${(v) => setOverride(S(u.user_id), r, v)} />
          </div></td>`)}
        </tr>`)}</tbody>
      </table></div>
      <p class="hint">${t('quotas.overrideHint')}</p>`}
    </section>
  </div>`;
}

export function People({ data, gid, guild, missing, reload, go }) {
  const [tab, setTab] = useState(() => load('heckler.people.tab', 'people'));
  useEffect(() => save('heckler.people.tab', tab), [tab]);
  if (!data) return html`<${Waiting} section="people" missing=${missing} guild=${guild} />`;
  const pending = data.requests.filter((q) => q.status === 'pending').length;
  return html`<div class="page">
    <${PageHead} section="people" />
    <${Tabs} label=${t('page.people.title')} value=${tab} onChange=${setTab}
      items=${[['people', t('people.tab')], ['quotas', t('quotas.tab'), pending]]} />
    ${tab === 'people' ? html`<${PeopleTab} data=${data} gid=${gid} reload=${reload} guild=${guild} />`
      : html`<${QuotasTab} data=${data} gid=${gid} reload=${reload} go=${go} />`}
  </div>`;
}

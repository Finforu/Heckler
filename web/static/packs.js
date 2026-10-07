// Packs: export this server's content as a .zip, or load a pack into it.
import { useRef, useState } from './vendor/hooks.js';
import { api, ConfirmButton, html, Icon, PageHead, t, Toggle, useId, useRunner, Waiting } from './lib.js';

export function Packs({ data, gid, guild, missing, reload }) {
  const [personal, setPersonal] = useState(false);
  const [mode, setMode] = useState('merge');
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [run, busy] = useRunner();
  const input = useRef(null);
  const ids = { file: useId('pk') };
  if (!data) return html`<${Waiting} section="packs" missing=${missing} guild=${guild} />`;
  const doImport = () => run(async () => {
    const form = new FormData();
    form.append('mode', mode);
    form.append('file', file);
    const out = await api('POST', `/api/g/${gid}/pack/import`, form);
    setResult(out);
    setFile(null);
    if (input.current) input.current.value = '';
    await reload();
  }, t('packs.imported'));

  return html`<div class="page">
    <${PageHead} section="packs" />
    <div class="two-col">
      <section class="card">
        <div class="card-head"><div><h2>${t('packs.exportTitle')}</h2><p class="sub">${t('packs.exportDesc')}</p></div></div>
        <${Toggle} label=${t('packs.personal')} hint=${t('packs.personalHint')} checked=${personal} onChange=${setPersonal} />
        <div class="card-actions">
          <a class="btn primary" href=${`/api/g/${gid}/pack/export?personal=${personal ? 1 : 0}`} download>
            <${Icon} name="down" size=${16} />${t('packs.download')}</a>
        </div>
      </section>
      <section class="card">
        <div class="card-head"><div><h2>${t('packs.importTitle')}</h2><p class="sub">${t('packs.importDesc')}</p></div></div>
        <div class="field">
          <label class="field-label" for=${ids.file}>${t('packs.file')}</label>
          <input type="file" id=${ids.file} ref=${input} accept=".yaml,.yml,.zip" onChange=${(e) => setFile(e.target.files[0] || null)} />
        </div>
        <fieldset class="radio-cards">
          <legend class="field-label">${t('packs.how')}</legend>
          <label class=${mode === 'merge' ? 'on' : ''}><input type="radio" name="pack-mode" checked=${mode === 'merge'} onChange=${() => setMode('merge')} />
            <span><b>${t('packs.merge')}</b><span class="hint">${t('packs.mergeHint')}</span></span></label>
          <label class=${mode === 'replace' ? 'on' : ''}><input type="radio" name="pack-mode" checked=${mode === 'replace'} onChange=${() => setMode('replace')} />
            <span><b>${t('packs.replace')}</b><span class="hint">${t('packs.replaceHint')}</span></span></label>
        </fieldset>
        <div class="card-actions">
          ${mode === 'replace'
            ? html`<${ConfirmButton} label=${t('packs.import')} confirmLabel=${t('packs.replaceConfirm')} disabled=${!file || busy} onConfirm=${doImport} />`
            : html`<button class="btn primary" disabled=${!file || busy} onClick=${doImport}>${t('packs.import')}</button>`}
        </div>
        ${result && html`<div class="import-result" role="status">
          <p><${Icon} name="check" size=${16} />${t('packs.result', result)}</p>
          ${result.warnings.length > 0 && html`<p class="hint">${t('packs.warnings', { n: result.warnings.length })}</p>
            <ul class="warnings">${result.warnings.map((w, i) => html`<li key=${i}>${w}</li>`)}</ul>`}
        </div>`}
      </section>
    </div>
  </div>`;
}

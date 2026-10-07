Vendored front-end libraries (no build step; served as-is).

  preact.js  preact 10.27.2  dist/preact.module.js        MIT         LICENSE-preact
  hooks.js   preact 10.27.2  hooks/dist/hooks.module.js   MIT         LICENSE-preact
  htm.js     htm 3.1.1       dist/htm.module.js           Apache-2.0  LICENSE-htm
  fonts/bricolage-grotesque-latin-wght.woff2
             Bricolage Grotesque (variable, latin)          OFL-1.1     fonts/OFL-bricolage-grotesque.txt
             from https://cdn.jsdelivr.net/fontsource/fonts/bricolage-grotesque:vf@latest/latin-wght-normal.woff2

Source: https://cdn.jsdelivr.net/npm/<package>@<version>/<path>

Local changes (the only ones):
  - hooks.js: the bare import `from"preact"` became `from"./preact.js"`,
    so no import map is needed (keeps the page CSP free of inline scripts).
  - the trailing `//# sourceMappingURL=` comments were removed (maps not vendored).

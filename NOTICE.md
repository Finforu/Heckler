# Third-party notices

Heckler is licensed under the GNU Affero General Public License v3.0 (see
`LICENSE`). It includes, or downloads at run time, the following third-party
work, under its own licenses.

## Included in this repository

### OmniVoice inference code (`omnivoice/`)
- Unmodified inference code from OmniVoice 0.2.1 by k2-fsa,
  <https://github.com/k2-fsa/OmniVoice>.
- Apache License 2.0; the full text is in `omnivoice/LICENSE`. Details are in
  `omnivoice/NOTICE.md`.

### Preact 10.27.2 (`web/static/vendor/preact.js`, `hooks.js`)
- Copyright (c) 2015-present Jason Miller.
- MIT License; the full text is in `web/static/vendor/LICENSE-preact`.

### htm 3.1.1 (`web/static/vendor/htm.js`)
- By Jason Miller.
- Apache License 2.0; the full text is in `web/static/vendor/LICENSE-htm`.

## Downloaded at first run (not included)

### OmniVoice model weights (`k2-fsa/OmniVoice`)
- Licensed **CC-BY-NC**: non-commercial use only.
- The model card strictly prohibits unauthorized voice cloning,
  impersonation, fraud and scams.
- Citation: Zhu et al., *OmniVoice*, arXiv:2604.00688 (2026).

### OmniVoice audio tokenizer (Boson Higgs Audio 2)
Licensed under the Boson Higgs Audio 2 Community License, which is based on
the Meta Llama 3 Community License. Its required attribution (section 1.b.i,
verbatim):

> Built with Higgs Materials licensed from Boson AI USA, Inc., Copyright Boson AI USA, Inc., All Rights Reserved and Meta Llama 3 licensed under the Meta Llama 3 Community License, Copyright Meta Platforms, Inc., All Right Reserved. based on Meta Llama 3

and its notice text (section 1.b.iii):

> Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
> Boson Higgs Audio 2 is licensed under the Boson Community License, Copyright © Boson AI USA, Inc. All Rights Reserved.

Use must also follow the Meta Llama 3 Acceptable Use Policy
(<https://llama.meta.com/llama3/use-policy>).

### Whisper (`deepdml/faster-whisper-large-v3-turbo-ct2`)
- OpenAI Whisper large-v3-turbo, converted to CTranslate2.
- MIT License.

### Parakeet TDT 0.6B v3 (NVIDIA, via sherpa-onnx int8)
- CC-BY-4.0. Verify this on the model page before redistributing.

### Python dependencies
Installed from PyPI under their own licenses; see `requirements.txt`.

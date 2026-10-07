Unmodified inference code from OmniVoice 0.2.1 by k2-fsa
(https://github.com/k2-fsa/OmniVoice, PyPI `omnivoice`), Apache License 2.0
(see LICENSE). Only `__init__.py`, `models/` and `utils/` are included. The
training, evaluation and CLI code is left out because the bot doesn't need it,
and installing the full package would pull in gradio and other heavy
dependencies.

The model weights (`k2-fsa/OmniVoice` on Hugging Face) are NOT included. They
are downloaded at first run, and they carry their own licenses:
- the OmniVoice model: CC-BY-NC (non-commercial use only)
- its audio tokenizer: the Boson Higgs Audio 2 Community License

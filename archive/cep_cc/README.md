# CEP-CC Split Notice

CEP-CC has been split out into its own project:

```text
F:\cep-cc
```

CEP-CC stands for:

> Compressed Emergent Protocol under Continuous Communication

It is no longer part of the `unified-sel` mainline.

Current project boundaries:

- `unified-sel`: Capability Router, TopoMem OBD, and boundary-local amplification.
- `cognitive-execution-engine`: execution kernel / reality commitment layer.
- `cep-cc`: continuous communication protocol emergence research.

The original CEP-CC files are still present in this working tree for now to
avoid destructive cleanup before an explicit archive/delete pass.

Validation after split:

```powershell
cd F:\cep-cc
python -m pytest tests\test_cep_cc_protocol.py -q
```

Result on 2026-04-22:

```text
33 passed, 1 pytest cache warning
```


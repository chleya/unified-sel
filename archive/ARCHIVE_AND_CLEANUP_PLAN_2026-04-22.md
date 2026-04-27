# Unified-SEL Archive And Cleanup Plan

Date: 2026-04-22

Goal: reduce `unified-sel` to the validated mainline while preserving enough provenance for research claims.

No destructive cleanup should happen until each batch below is explicitly approved.

---

## Mainline To Keep

Keep these in the active project root:

| Path | Reason |
|---|---|
| `core/capability_benchmark.py` | Capability Router engine |
| `core/llm_solver.py` | real/simulated LLM solver adapters |
| `experiments/capability/` | current tool/product line |
| `data/capability_boundary_bench/` | public/eval benchmark data |
| `double_helix/` | boundary-local amplification paper experiments |
| `topomem/health_controller.py` | TopoMem OBD candidate |
| `topomem/topology.py` | topology and geometry health signals |
| `topomem/memory.py` | memory substrate used by TopoMem |
| `tests/smoke_test.py` | repo-level validation |
| `tests/test_llm_solver.py` | LLM solver validation |
| `papers/` | paper drafts |
| `AGENTS.md` | operating rules |
| `PROJECT_OVERVIEW_AND_INDEX_2026-04-22.md` | current map |
| `PROJECT_PIVOT_DECISION_2026-04-16.md` | pivot rationale |
| `TOPOMEM_ROUTING_MONITOR_RESULT_2026-04-16.md` | rejected routing evidence |
| `EXPERIMENT_LOG.md` | provenance log |

---

## Already Split Out

| Line | New location | Unified-SEL action |
|---|---|---|
| CEP-CC | `F:\cep-cc` | keep pointer at `archive/cep_cc/README.md`; remove active CEP-CC files after approval |

Validation after split:

```powershell
cd F:\cep-cc
python -m pytest tests\test_cep_cc_protocol.py -q
```

Observed result: `33 passed, 1 pytest cache warning`.

---

## Archive Batch A: Sidecar Research Lines

Move to `archive/sidecar/` or delete from active root after confirming `F:\cep-cc` and docs are sufficient.

| Candidate | Action | Reason |
|---|---|---|
| `experiments/cep_cc/` | archive/remove active copy | split to `F:\cep-cc` |
| `tests/test_cep_cc_protocol.py` | archive/remove active copy | split to `F:\cep-cc` |
| `CEP_CC_*.md` | archive/remove active copy | split to `F:\cep-cc/docs/results/` |
| `META_CONTROLLER_TO_CEP_CC_HANDOFF_2026-04-21.md` | archive/remove active copy | copied to `F:\cep-cc/docs/` |
| `experiments/meta_controller/` | archive | sidecar controller exploration |
| `META_CONTROLLER_*.md` | archive | sidecar controller docs |
| `structural_bayesian_field/` | archive | sidecar theory track |
| `STRUCTURAL_BAYESIAN_FIELD_NOTE_2026-04-15.md` | archive | sidecar theory doc |
| `weight_graph/` | archive or externalize | sidecar topology analysis; huge caches should not be active |

Recommended first action: move, do not delete.

---

## Archive Batch B: Old Mechanism Track

Move to `archive/old_mechanism_track/`.

| Candidate | Reason |
|---|---|
| `experiments/continual/` | old surprise/EWC mechanism line |
| `experiments/A1_*.py` | historical routing/mechanism diagnostics |
| `experiments/phase*.py` | old phase experiments |
| `experiments/surprise_*.py` | old surprise-driven experiments |
| `experiments/*freeze*.py` | old freeze/pool diagnostics |
| `experiments/*lambda*.py` | old scan diagnostics |
| `experiments/*snapshot*.py` | old snapshot routing diagnostics |
| `analysis/` | mostly old analysis scripts; keep only if actively referenced |
| `CAPABILITY_BENCHMARK_TRACK.md` | old track doc; keep archived |
| `CAPABILITY_MAINLINE_CONCLUSION_2026-04-11.md` | old conclusion doc; keep archived |
| `MAINLINE_EXECUTION_PLAN_2026-04-11.md` | old mainline plan |
| `PROJECT_MAINLINE_2026-04-09.md` | old mainline framing |
| `PROJECT_GAPS_AND_ROADMAP.md` | old roadmap |
| `PROJECT_NORTH_STAR_90_DAY_ROADMAP_2026-04-11.md` | old roadmap |

Reason: the original surprise-driven structural birth/death claim is archived and unverified.

---

## Delete Batch C: Generated Temporary State

Safe to delete after approval.

| Candidate | Reason |
|---|---|
| `.pytest_cache/` | generated test cache |
| `tmp_forgetting_chroma/` | generated Chroma temp |
| `tmp_forgetting_v2_chroma/` | generated Chroma temp |
| `tmp_topomem_fd/` | generated temp |
| `tmp_topomem_forgetting/` | generated temp |
| `topomem/tmp/` | generated temp |
| `topomem/tmp_exp_forgetting_v2/` | generated temp |
| `topomem/__test_add_tmp/` | generated test temp |
| `topomem/__test_import_tmp/` | generated test temp |

This batch should be deleted rather than archived.

---

## Externalize Batch D: Large Artifacts

Move outside the repo or redownload on demand.

| Candidate | Size / Reason |
|---|---|
| `double_helix/*.gguf` | multi-GB model files |
| `topomem/data/models/*.gguf` | model files |
| `weight_graph/cache_*.pkl` | multi-GB derived caches |
| `topomem/data/chromadb/` | generated vector DB |

Suggested external location:

```text
F:\unified-sel-artifacts\
```

Do not commit these files.

---

## Cleanup Order

1. Commit or stash current documentation changes.
2. Verify `F:\cep-cc` independently.
3. Archive Batch A by moving files/directories.
4. Run `python tests/smoke_test.py`.
5. Delete Batch C temporary state.
6. Externalize Batch D large artifacts.
7. Run focused Capability Router tests and smoke test again.
8. Update `PROJECT_OVERVIEW_AND_INDEX_2026-04-22.md`.

---

## Non-Negotiable Boundaries

- Do not delete `results/` during the first cleanup pass.
- Do not edit JSON result files manually.
- Do not remove `double_helix/` active paper-line scripts until the paper draft is stable.
- Do not remove TopoMem OBD files; only remove temp/vector DB/model artifacts.
- Do not re-promote TopoMem surprise as a per-task routing monitor.


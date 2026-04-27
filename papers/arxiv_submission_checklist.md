# arXiv Submission Checklist

## Paper Information

| Field | Current Value | arXiv Requirement | Status |
|-------|--------------|-------------------|--------|
| Title | Boundary-Local Amplification: Feedback Retry Concentrates Benefit in the Near-Boundary Zone | ≤ 300 chars | ✅ 96 chars |
| Abstract | 204 words | ≤ 1920 chars (~300 words) | ✅ |
| Authors | Anonymous | Real names required | ❌ TBD |
| Keywords | None | Optional but recommended | ⚠️ TBD |
| MSC Classes | None | Optional | ⚠️ TBD |

## Pre-Submission Steps

### 1. Replace Anonymous Author
```latex
% Before
\author{Anonymous}

% After (example)
\author{Your Name$^{1}$ \and Co-Author$^{2}$ \\
$^{1}$Your Institution \\ 
$^{2}$Co-Author Institution}
```

### 2. arXiv-Specific LaTeX Adjustments

arXiv uses TeX Live 2023 with pdflatex. Our paper uses standard packages, so no changes needed.

**But note**: arXiv does not support `biblatex` well. We use `natbib` + `thebibliography`, which is fully compatible.

### 3. File Upload Structure

```
boundary_local_amplification.tex    # Main file
# No separate .bib needed (references inline)
```

### 4. Abstract Quality Check

Current abstract (204 words) covers:
- ✅ Problem: Feedback retry benefit assumed uniform
- ✅ Finding: Boundary-local amplifier, inverted-U pattern
- ✅ Evidence: p = 0.0008, synthetic + real-LLM validation
- ✅ Practical: ABOVE-filtering saves 54.4% calls
- ✅ Limitation: Scale-dependent NEAR-zone emergence

**Suggested keywords**:
- LLM metacognition
- Feedback retry optimization
- Capability boundary detection
- Verification scheduling
- Resource-efficient inference

### 5. arXiv Category Selection

| Category | Description | Fit |
|----------|-------------|-----|
| cs.CL | Computation and Language (NLP) | ✅ Primary |
| cs.LG | Machine Learning | ✅ Secondary |
| cs.AI | Artificial Intelligence | ✅ Secondary |
| cs.SE | Software Engineering | ⚠️ (code repair focus) |

**Recommended**: `cs.CL` primary, `cs.LG` secondary

### 6. Post-Submission Actions

- [ ] Update CITATION.cff with arXiv ID
- [ ] Update README.md with arXiv badge
- [ ] Tweet/announce (optional)

## Workshop Backup Plan

If arXiv gets low engagement, target these workshops:

| Workshop | Deadline (2026) | Fit | Acceptance Rate |
|----------|----------------|-----|----------------|
| ACL Workshop on Efficient NLP | ~May | High | ~30% |
| NeurIPS Workshop on ML for Systems | ~Sep | Medium | ~25% |
| ICML Workshop on Efficient LLMs | ~May | High | ~30% |
| EMNLP Workshop on NLG Evaluation | ~Aug | Medium | ~35% |

## Current Blockers

1. **Author names** — need real names for arXiv
2. **ORCID** — optional but recommended
3. **Institution affiliation** — required

## Next Action

Decide:
- A) **先投 arXiv**（最快，无审稿，建立优先权）
- B) **先投 Workshop**（有审稿反馈，但周期长）
- C) **同时准备**（先 arXiv 占坑，再投 workshop）

推荐 **C**：先 arXiv 建立优先权，再投 workshop 获取审稿意见。

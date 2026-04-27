# Overleaf Upload Guide

## Quick Start

1. Go to [overleaf.com](https://www.overleaf.com) and create a new project
2. Upload `boundary_local_amplification_overleaf.zip`
3. Select **pdfLaTeX** as the compiler
4. Click "Recompile"

## File Structure

```
boundary_local_amplification.tex   # Main paper (self-contained)
```

This paper uses the `thebibliography` environment (inline references), so **no separate `.bib` file is needed**.

## Expected Output

- **Pages**: ~6-7 pages
- **Format**: 11pt, A4, two-column style (single-column article)
- **References**: 15 entries (inline, natbib style)

## Compiler Settings

| Setting | Value |
|---------|-------|
| Compiler | pdfLaTeX |
| TeX Live | 2023 or later |
| Main document | `boundary_local_amplification.tex` |

## Packages Used

All packages are standard and available in TeX Live:
- `amsmath`, `amssymb` — Math
- `graphicx` — Figures (if added)
- `booktabs` — Tables
- `hyperref` — Links
- `natbib` — Citations (used with `thebibliography`)
- `geometry` — Margins
- `algorithm`, `algpseudocode` — Algorithms
- `xcolor` — Colors
- `caption`, `subcaption` — Figure captions

## Troubleshooting

### Missing `natbib` style
If you see warnings about citation style, ensure `natbib` is loaded. The paper uses `\bibliographystyle{plainnat}`.

### Unicode characters
The paper uses UTF-8 encoding (`\usepackage[utf8]{inputenc}`). Ensure your Overleaf project uses UTF-8.

### Tables too wide
If tables overflow margins, the paper uses `booktabs` with standard column specs. On A4 with 2.5cm margins, all tables should fit.

## Post-Compilation Checklist

After successful compilation, verify:

- [ ] Abstract renders correctly
- [ ] All 6 tables are visible
- [ ] All 15 references appear in the bibliography
- [ ] No missing citation warnings (check raw logs)
- [ ] Section numbering is correct (1, 2, 3...)
- [ ] Appendix appears after references

## Anonymous Submission

The paper is set to `\author{Anonymous}` for double-blind review. Before camera-ready submission, replace with actual author names and affiliations.

## arXiv Submission (Future)

For arXiv:
1. Remove `Anonymous` author
2. Add ORCID links if available
3. Check abstract is under 1920 characters (current: ~1800)
4. Add MSC classes or ACM classes if required

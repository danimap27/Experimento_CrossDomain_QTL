# paperQAI — Reconstructed LaTeX of the IEEE QAI submission

LaTeX reconstruction of the paper actually submitted to IEEE QAI 2026
(paper 191), rebuilt from the submitted PDF `IEEE_QAI___Forgetting (6).pdf`
(7 pages, anonymised, and containing the final modifications by
F. Martínez-Álvarez, including the Experiment-1 values, the rewritten
learning-rate paragraph and the PPI2505 acknowledgment).

## Contents

- `main.tex` — full source: title, abstract, Sections I–V, equations (1)–(11),
  Tables I–VI, Fig. 1 (pipeline, redrawn in TikZ following the original
  layout), Fig. 2 (convergence, image extracted from the submitted PDF),
  acknowledgments, and references [1]–[28] as an inline bibliography.
- `figures/fig2_exp3_convergence.png` — the image embedded in the submitted
  PDF (extracted with `pdfimages`, 893×533).
- Build: `pdflatex main.tex` (twice). Produces 7 pages, the same page count
  as the submission. Word-level similarity against the PDF text: 0.93, with
  the remaining differences being float-ordering artefacts of text
  extraction (tables and figures spanning columns in different order).

## Two data notes before reusing any of this

1. **Experiment 1, Table II.** The submitted PDF reports QTL drops of
   42.60 / 44.10 / 47.20 (Ideal / Heron r2 / Legacy NISQ). The per-seed
   payloads stored in this repository (the noise-profile sweep under
   `../results/`) give QTL drops of 65.10 / 65.10 / 61.20, while the
   baselines match the PDF exactly (72.10 / 72.90 / 70.50). The provenance of
   the Table II QTL values is unresolved. Confirm which execution produced
   them before reusing these numbers in any derivative manuscript.
2. **Experiment 1 protocol text.** The submission states that the lowest
   layer is frozen during the sequential phase and that the learning-rate
   schedule is applied identically to the baseline and the QTL arms. The code
   in the parent repository applies neither: the freeze condition
   (`if '0' in name`) matches no parameter, and the arms use different rates
   (0.05 for the baseline against 0.01/0.005 for the QTL arm). See audit
   findings A2 and A8 in `../journal/PLAN.md`.

## Provenance

- Source PDF: the file distributed by F. Martínez-Álvarez (the version
  uploaded to CMT; anonymised for review).
- Reconstruction text: `pdftotext`; Fig. 2 via `pdfimages`; Fig. 1 redrawn in
  TikZ; bibliography transcribed verbatim from the PDF.

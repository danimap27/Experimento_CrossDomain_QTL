# Cover letter (working draft) — Quantum Machine Intelligence

> Nota interna (no incluir en el envío): (1) decidir con Paco la mención al envío previo de conferencia — el borrador usa una fórmula neutra; (2) título alternativo más sobrio: "Controlled characterisation of cross-domain synthetic pre-training for quantum continual learning"; (3) adaptar plantilla si finalmente se elige otro venue.

---

**To:** The Editors-in-Chief, *Quantum Machine Intelligence* (Springer)

**Subject:** Submission of "Optimisation regimes, not initialisations: a controlled study of cross-domain pre-training in quantum continual learning"

Dear Editors,

We are pleased to submit our manuscript, "Optimisation regimes, not initialisations: a controlled study of cross-domain pre-training in quantum continual learning", for consideration in *Quantum Machine Intelligence*.

Memory-free initialisation strategies have been proposed as an attractive route to mitigate catastrophic forgetting in quantum continual learning, since replay buffers are costly on near-term devices. This manuscript provides a controlled characterisation of one such strategy, cross-domain synthetic pre-training, and answers the question of what it actually delivers. Using a factorial design over 650 training runs that crosses the initialisation with the optimisation regime, we show that the apparent benefit reported by the original protocol is carried by the learning-rate regime, while the synthetic initialisation itself produces no significant change once rates are matched. A benchmark campaign over class-incremental sequences of up to five tasks, two noise profiles, qubit counts from four to eight and data budgets from five hundred to twelve thousand samples confirms the null reading, with paired tests, effect sizes and bootstrap intervals throughout. Mechanism probes then locate the role of the initialisation: it moves the circuit to a region with far stronger gradient signal without improving retention, dissociating gradient geometry from continual-learning performance. As part of the same protocol we quantify the behaviour of established remedies, and rehearsal emerges as the only evaluated component with a clear benefit on standard benchmarks.

We believe this study fits the scope of *Quantum Machine Intelligence* for three reasons. First, it addresses a question of practical interest for the community working on variational quantum classifiers under realistic noise budgets. Second, its contributions are methodological and mechanistic rather than incremental: a controlled design, an explicit statistical protocol, and an open pipeline with per-seed payloads, which we hope will serve as a reference for future claims about initialisation and transfer effects in quantum machine learning. Third, the negative core result is reported with the same care as a positive one, which we understand is aligned with the journal's emphasis on rigorous empirical studies of quantum learning systems.

A preliminary version of parts of this study was earlier submitted in a shorter form. This manuscript is substantially extended and includes entirely new experiments, analyses and conclusions.

All experiments, the full per-seed result payloads and the analysis scripts are openly available in the repository indicated in the manuscript, and the computing resources used (Centro de Informática Científica de Andalucía, Hércules cluster) are acknowledged. The manuscript is an original submission, is not under consideration elsewhere, and its authors declare no competing interests.

Thank you for your consideration.

Sincerely, on behalf of all authors,

Daniel Martín Pérez
Data Science and Big Data Lab, Pablo de Olavide University, Seville, Spain
danimp2002@gmail.com

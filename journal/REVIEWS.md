# IEEE QAI 2026 — Reviews of Paper 191

> Submission: "Cross-Domain Synthetic Pre-Training for Catastrophic Forgetting
> Mitigation in Quantum Continual Learning", IEEE QAI 2026, paper ID 191.
> Decision: **rejected** (16 Sep 2026). Both reviewers: Weak Reject.
> Transcribed from the CMT review export (2026-09-15).

## Reviewer #2 — Weak Reject

Relevance: Fair · Originality: Fair · Technical Quality: Poor · Presentation: Fair

The paper works on mitigating catastrophic forgetting in quantum continual
learning and offers a refreshingly simple, memory-free initialization strategy
that avoids the hardware overhead of replay buffers. That being said, the paper
has a few concerns that need to be addressed.

1. The novelty of the paper is overclaimed. The core premise of using
   cross-domain synthetic pre-training to improve sequential learning is a
   well-established classical transfer learning technique. The investigation of
   synthetic Gaussian data as a pre-training source is somewhat interesting,
   yet the paper provides no theoretical or empirical justification for why
   such data yields a favorable initialization. The authors must investigate
   the loss landscape, parameter distribution, or barren plateau behavior
   post-pre-training to obtain insights into the quantum-specific advantages.

2. The evaluation is too narrow to support the claims. The forgetting drop
   metric is not theoretically explained, and no statistical tests are
   performed despite large standard deviation values (~6.43, Table II). The
   paper also misses the comparison with baselines established in quantum
   continual learning, such as EWC-based regularization [R1] or simple
   rehearsal methods. The results are also not consistent: Experiment 1 never
   reports the absolute Task A accuracy after Task A training; only the drop in
   ΔA is reported, making it difficult to even verify the true initial accuracy
   for that experiment.

3. The approach is unlikely to scale to practical NISQ settings. The pipeline
   relies on a 4-qubit circuit; scaling to 8–12 qubits would exponentially
   increase the parameter space and noise sensitivity, while the benefits of
   synthetic pre-training are unlikely to hold in higher-dimensional, more
   complex feature spaces. The authors may want to explore the fault-tolerant
   resource utilization of their proposed approach to comment on scalability.
   Classical pre-training cost (MobileNetV2 + PCA) already dominates the
   pipeline, and for larger image datasets or longer task sequences, this
   pre-processing overhead would become prohibitive, undermining any quantum
   advantage. Also, an ablation across qubit count, circuit depth, or task
   count will make the practical impact of the contribution clearer.

References: [R1] C. Zhang, Z. Lu, L. Zhao, S. Xu, W. Li, et al.,
"Experimental demonstration of quantum continual learning with superconducting
qubits," npj Quantum Information, vol. 12, no. 28, 2026.

## Reviewer #4 — Weak Reject

Relevance: Excellent · Originality: Fair · Technical Quality: Fair · Presentation: Fair

This paper aims to mitigate catastrophic forgetting in quantum continual
learning through a cross-domain synthetic pretraining approach. It is evaluated
by first training the hybrid quantum-classical models on synthetic Gaussian
data before moving to sequential image recognition tasks. It compares three
parameterized quantum circuit ansätze (i.e., Strongly Entangling (SE), Basic
Entangler (BE), and Tree Tensor Network (TTN)) under ideal and simulated noise
profiles. In general, there are several concerns related to the proposed
research work, as discussed further below.

Weaknesses:

- The chain of arguments is not well established.
- The coverage of existing continual learning methods from classical networks
  is very limited.
- The continual learning scenario is not well defined.
- The evaluation datasets are too small.
- Overall, the research work is not mature for publication.

Main Comments:

- It claims that the quantum-classical models suffer from catastrophic
  forgetting, but the paper does not provide empirical evidence.
- The motivation for why a continual learning study is required here, is not
  convincing.
- It mentions that "The underlying hypothesis is that pre-training a
  parameterized quantum circuit on generic synthetic data provides a more
  favorable initialization for subsequent CL than random ones", but the
  rationale is not clear.
- The rationale regarding the decision of studying cross-domain synthetic
  pretraining is not clear.
- The coverage of existing continual learning methods from classical networks
  is very limited. Therefore, the study seems to be a preliminary work rather
  than a mature one.
- Circuit descriptions are incomplete and possibly inconsistent with standard
  ansatz definitions.
- The paper mixes the methodology with the experimental pipeline.
- The continual learning scenario is not well defined, which is critical for
  framing the entire study. The paper describes the scenario in very
  specifically-crafted settings, hence the generality of the solution seems to
  be limited.
- The task-incremental learning scenario considered in the evaluation is too
  simple and not widely used in the continual learning community.
- The evaluation datasets (i.e., MNIST and Fashion MNIST) are too small.
- The state-of-the-art comparison should be quantitative rather than
  descriptive.
- Claims about realistic quantum-hardware deployment based on wall-clock
  simulation time are not justified.

## Prior round (context)

The same study was rejected at IEEE QCE26 (paper 1184, July 2026) with one weak
accept and two weak rejects. Recurring criticisms across both rounds: novelty
framing, missing statistical tests, missing CL baselines, narrow evaluation,
and layer-freezing inconsistency. The journal version must close all of them.

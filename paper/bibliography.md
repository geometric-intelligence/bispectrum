# Bibliography: Best NeurIPS Papers on Software / Python Packages

*Compiled for: Bispectrum Python Package Paper — NeurIPS Submission*
*Date: March 2026*

______________________________________________________________________

## 1. Overview

This document surveys the most influential papers published at NeurIPS (formerly NIPS) that introduce software packages, Python libraries, or frameworks for machine learning. The goal is twofold: (a) provide citable references and related work for a software-oriented NeurIPS submission, and (b) extract structural and stylistic patterns that characterize successful software papers at this venue.

______________________________________________________________________

## 2. Landmark Software / Framework Papers at NeurIPS

### 2.1 PyTorch (NeurIPS 2019)

**Title:** PyTorch: An Imperative Style, High-Performance Deep Learning Library
**Authors:** Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, Soumith Chintala
**Venue:** NeurIPS 2019 (Advances in Neural Information Processing Systems 32)
**Citations:** ~100,000+ (2nd most internally cited NeurIPS paper with 720 NeurIPS-to-NeurIPS citations)

**Summary:** Introduces PyTorch as an imperative, Pythonic deep learning framework with dynamic computational graphs and GPU acceleration. The paper emphasizes usability, performance, and extensibility, positioning PyTorch as an alternative to static-graph frameworks (TensorFlow, Theano). By NeurIPS 2024, approximately 75% of papers at the conference used PyTorch.

**Key takeaways for your paper:**

- Clearly articulates *design philosophy* (imperative vs. declarative, define-by-run)
- Benchmarks against existing frameworks on both performance and usability
- Emphasizes community adoption and ecosystem (torchvision, torchaudio, etc.)
- Strong "why this matters" framing: lowering the barrier to research

______________________________________________________________________

### 2.2 Attention Is All You Need (NeurIPS 2017)

**Title:** Attention Is All You Need
**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
**Venue:** NeurIPS 2017 (NIPS 31)
**Citations:** ~150,000+ (most internally cited NeurIPS paper: 1,014 NeurIPS-to-NeurIPS citations)

**Summary:** Introduces the Transformer architecture, replacing recurrence and convolutions entirely with self-attention mechanisms. While primarily an architecture paper, it spawned the entire Hugging Face Transformers ecosystem and fundamentally changed how software packages for NLP and beyond are built.

**Relevance to your paper:** Demonstrates how a well-designed abstraction (the attention mechanism) becomes a reusable software building block adopted across an entire field.

______________________________________________________________________

### 2.3 Generative Adversarial Nets (NeurIPS 2014) — Test of Time Award 2024

**Title:** Generative Adversarial Nets
**Authors:** Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
**Venue:** NeurIPS 2014 (NIPS 27)
**Citations:** ~85,000+
**Award:** NeurIPS 2024 Test of Time Award

**Summary:** Introduces the GAN framework — a minimax game between a generator and discriminator network. Became one of the foundational paradigms for generative modeling, spawning dozens of specialized software libraries.

______________________________________________________________________

### 2.4 LightGBM (NeurIPS 2017)

**Title:** LightGBM: A Highly Efficient Gradient Boosting Decision Tree
**Authors:** Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu
**Venue:** NeurIPS 2017 (NIPS 30), pp. 3149–3157
**Citations:** ~20,000+

**Summary:** Introduces two novel techniques — Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) — to speed up gradient boosting. The accompanying Python/C++ library became one of the most widely used ML tools in industry and Kaggle competitions.

**Key takeaways for your paper:**

- Classic "software paper" structure: algorithm + implementation + extensive benchmarks
- Clear performance comparisons (speed and accuracy) against XGBoost and other baselines
- Released as a well-documented, pip-installable package

______________________________________________________________________

### 2.5 GPyTorch (NeurIPS 2018)

**Title:** GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration
**Authors:** Jacob R. Gardner, Geoff Pleiss, Kilian Q. Weinberger, David Bindel, Andrew Gordon Wilson
**Venue:** NeurIPS 2018
**Citations:** ~2,000+

**Summary:** Presents an efficient approach to GP inference based on Blackbox Matrix-Matrix multiplication (BBMM), reducing complexity from O(n³) to O(n²). The GPyTorch library, built on PyTorch, provides scalable GP inference with GPU acceleration.

**Key takeaways for your paper (most directly relevant):**

- **This is the closest template for your bispectrum paper** — a math-heavy Python package with GPU acceleration
- Structure: mathematical contribution (BBMM algorithm) + software design + benchmarks
- Emphasizes both algorithmic novelty and practical usability
- Built on top of PyTorch, leveraging existing ecosystem
- Clear API examples showing how users interact with the library

______________________________________________________________________

### 2.6 GPT-3 / Language Models are Few-Shot Learners (NeurIPS 2020) — Best Paper Award

**Title:** Language Models are Few-Shot Learners
**Authors:** Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, et al.
**Venue:** NeurIPS 2020
**Citations:** ~40,000+
**Award:** NeurIPS 2020 Best Paper Award

**Summary:** Demonstrates that scaling language models to 175B parameters enables strong few-shot learning without fine-tuning. While primarily a scaling/capability paper, it catalyzed the entire ecosystem of LLM software tools (OpenAI API, LangChain, etc.).

______________________________________________________________________

### 2.7 Benchopt (NeurIPS 2022)

**Title:** Benchopt: Reproducible, Efficient and Collaborative Optimization Benchmarks
**Authors:** Thomas Moreau, Mathurin Massias, Alexandre Gramfort, Pierre Ablin, Pierre-Antoine Bannier, Benjamin Charlier, Mathieu Dagréou, Tom Dupré la Tour, Ghislain Durif, Cassio F. Dantas, Quentin Klopfenstein, Johan Larsson, En Lai, Tanguy Lefort, Benoît Malézieux, Badr Moufad, Binh T. Nguyen, Alain Rakotomamonjy, Zaccharie Ramzi, Joseph Salmon, Samuel Vaiter
**Venue:** NeurIPS 2022 (Main Conference Track)

**Summary:** A collaborative framework to automate, publish, and reproduce optimization benchmarks in ML across programming languages (Python, R, Julia, C/C++) and hardware architectures. Demonstrates benchmarks on ℓ₂-regularized logistic regression, Lasso, and ResNet18 training.

**Key takeaways for your paper:**

- Excellent example of a "tool/framework" paper at NeurIPS main track
- Emphasizes reproducibility and collaboration as first-class design goals
- scikit-learn compatible API design
- Cross-language support as a differentiator

______________________________________________________________________

### 2.8 PettingZoo (NeurIPS 2021 — Datasets & Benchmarks Track)

**Title:** PettingZoo: A Standard API for Multi-Agent Reinforcement Learning
**Authors:** J. K. Terry, Benjamin Black, Nathaniel Grammel, Mario Jayber, Ananth Hari, Ryan Sullivan, Luis Santos, Clemens Dieffendahl, Caroline Horsch, Rodrigo Perez-Vicente, Niall Williams, Yashas Lokesh, Praveen Ravi
**Venue:** NeurIPS 2021 (Datasets & Benchmarks Track)

**Summary:** Introduces PettingZoo, a library of diverse multi-agent RL environments with a universal Python API. Proposes the Agent Environment Cycle (AEC) games model. Analogous to what OpenAI Gym did for single-agent RL.

**Key takeaways for your paper:**

- Shows how "API standardization" papers can succeed at NeurIPS
- Strong emphasis on *why* the design choices were made (not just *what*)
- Includes both theoretical justification (AEC model) and practical utility

______________________________________________________________________

### 2.9 Gymnasium (NeurIPS 2025 — Datasets & Benchmarks Track, Spotlight)

**Title:** Gymnasium: A Standard Interface for Reinforcement Learning Environments
**Authors:** Farama Foundation team
**Venue:** NeurIPS 2025 (Datasets & Benchmarks Track — Spotlight)

**Summary:** The maintained successor to OpenAI Gym. Provides a standard API for RL environments with over 18 million downloads since release. Describes the theoretical and practical design considerations behind the API.

**Relevance:** Demonstrates that even "API maintenance/upgrade" papers can receive spotlight recognition when the contribution is well-articulated and widely impactful.

______________________________________________________________________

### 2.10 GAUCHE (NeurIPS 2023)

**Title:** GAUCHE: A Library for Gaussian Processes in Chemistry
**Authors:** Ryan-Rhys Griffiths et al.
**Venue:** NeurIPS 2023 (Main Conference Track)

**Summary:** An open-source library providing GP kernels for structured chemical inputs (graphs, strings, bit vectors) for Bayesian optimization in chemistry. Built on GPyTorch and BoTorch.

**Key takeaways for your paper:**

- Domain-specific GP library accepted at NeurIPS main track
- Shows that building on existing frameworks (GPyTorch) and targeting a specific application domain is a viable publication strategy
- Modular design philosophy

______________________________________________________________________

## 3. Other Highly Influential Papers in the NeurIPS Ecosystem

These papers were not published at NeurIPS but are foundational to the ML software ecosystem and are frequently cited alongside NeurIPS software papers:

| Paper                                                                          | Venue        | Year | Citations | Notes                                         |
| ------------------------------------------------------------------------------ | ------------ | ---- | --------- | --------------------------------------------- |
| Scikit-learn: Machine Learning in Python (Pedregosa et al.)                    | JMLR         | 2011 | ~110,000+ | Most cited ML software paper ever             |
| TensorFlow: A System for Large-Scale ML (Abadi et al.)                         | OSDI         | 2016 | ~50,000+  | Static-graph predecessor to PyTorch           |
| Theano: A Python Framework (Theano Dev Team)                                   | arXiv        | 2016 | ~10,000+  | Pioneer of symbolic differentiation in Python |
| XGBoost: A Scalable Tree Boosting System (Chen & Guestrin)                     | KDD          | 2016 | ~40,000+  | Predecessor/competitor to LightGBM            |
| Optuna: A Next-generation Hyperparameter Optimization Framework (Akiba et al.) | KDD          | 2019 | ~5,000+   | Define-by-run HPO framework                   |
| Pyro: Deep Universal Probabilistic Programming (Bingham et al.)                | JMLR         | 2019 | ~3,000+   | PyTorch-based PPL                             |
| JAX: Composable Transformations of Python+NumPy programs (Bradbury et al.)     | GitHub/arXiv | 2018 | ~5,000+   | Functional transformations (vmap, pmap, jit)  |

______________________________________________________________________

## 4. NeurIPS Test of Time Awards (Software-Relevant)

These awards honor papers from ~10 years prior that have had lasting impact:

| Year | Paper                                                               | Original Year | Citations                                       |
| ---- | ------------------------------------------------------------------- | ------------- | ----------------------------------------------- |
| 2020 | HOGWILD!: Lock-Free SGD (Niu et al.)                                | 2011          | Foundational for parallel training software     |
| 2022 | AlexNet: ImageNet Classification with Deep CNNs (Krizhevsky et al.) | 2012          | Sparked the deep learning revolution            |
| 2023 | word2vec: Distributed Representations of Words (Mikolov et al.)     | 2013          | 40,000+ citations; foundational for NLP tooling |
| 2024 | Generative Adversarial Nets (Goodfellow et al.)                     | 2014          | 85,000+ citations                               |
| 2024 | Sequence to Sequence Learning (Sutskever et al.)                    | 2014          | Foundational for encoder-decoder architectures  |
| 2025 | Faster R-CNN (Ren, He, Girshick, Sun)                               | 2015          | 56,700+ citations                               |

______________________________________________________________________

## 5. Actual Reviewer Feedback from OpenReview — What Reviewers Say About Software Papers

This section contains real reviewer comments extracted from OpenReview/NeurIPS proceedings, organized by the key themes that emerge. These are direct insights into what reviewers look for and complain about.

### 5.1 PyTorch (NeurIPS 2019) — Reviews

**Scores:** Reviewer 1 raised to 9 after rebuttal, Reviewer 2 strong positive, Reviewer 3 positive.

**Reviewer 1 (Score: 9 after rebuttal):**

- Praised: Impact in the research community. "Given the impact of the library in the research community I strongly support the publication of this paper at NeurIPS."
- Raised the venue question: "One may argue whether a systems/software paper, presenting the implementation details of a library should be published at NeurIPS, or whether it would be a better fit for USENIX or SysML."
- Criticism: "My main criticism is that you claim that PyTorch offers performance comparable to the fastest current libraries for DL. But what are these libraries? [...] I feel to support this claim a broader comparison to other libraries would be necessary."
- Wanted broader benchmarks beyond MXNet and TensorFlow.

**Reviewer 2 (Strong positive):**

- Key insight: "I applaud the authors for focusing their submission around their key design principles and decisions, rather than presenting a detailed tutorial of the software's use. This distinguishes the submission from a 'white paper,' i.e., a paper which reports on how to use a piece of software while marketing its merits, but having less scientific value."
- Praised: "The paper reads as a kind of manifesto for what deep learning software should aspire to be."
- Appreciated tradeoff analysis: "The authors detail the consequences and tradeoffs of their design choices carefully, making clear that sometimes this may lead to slightly worse performance, which is acceptable if it makes the software easier to use."
- Noted: "There is no one big idea here [...] Nor is there any new or surprising result here, since PyTorch has been available (and widely used) for several years. However, the paper demonstrates the successful synthesis of several key ideas."

**Reviewer 3:**

- "Since Pytorch has been used and tested by the community for years, the validity of the paper is without a doubt."
- "It is very certain that this paper has high originality, quality, clarity, and significance, though it might come a bit late."

**Key lesson for your paper:** Even PyTorch faced the "is this the right venue?" question. The paper succeeded because it framed itself as a *design principles manifesto*, not a tutorial. Reviewers explicitly praised this framing.

______________________________________________________________________

### 5.2 GPyTorch (NeurIPS 2018) — Reviews

**All 3 reviewers positive. Accepted.**

**Reviewer 1:**

- "The work is precisely explained, with appropriate references for related and relevant work, and is of great impact."
- "The key contribution of the paper is a framework that computes GP inference via a blackbox routine that performs matrix-matrix multiplications efficiently with GPU acceleration."
- Praised: Clear separation from state of the art with explicit quotes showing how the method differs.
- Minor issues only (typos).

**Reviewer 2:**

- Praised the dual contribution: algorithmic (modified PCG) + software (GPyTorch).
- Concern: "The paper focused on regression. It is true that it can be easily extended to other problems like classification, but in that case the algorithm needs specific inference algorithms [...] and it could be argued that it is not that black-box anymore."
- "Overall, it is a strong well-written paper with nice theoretical and practical contributions."

**Reviewer 3:**

- "The ideas presented in the paper are not ground-breakingly new, but are a clever combination and implementation of existing ones. That is absolutely fine and sufficient for acceptance."
- "Novel is the theoretical analysis of the preconditioner."
- Criticism: "The presentation of the results regarding speed-up appear arbitrary — twenty conjugate gradient steps are not a lot. I made the experience that in bad cases, conjugate gradients may need more than a thousand steps. [...] The quality of this submission could be improved by adding plots how the approximation quality and the residual evolve over time."
- "Facilitating the usage of GPUs for GP inference is an important contribution."

**Key lesson for your paper:** GPyTorch succeeded by pairing a clear algorithmic improvement (BBMM reducing O(n³) to O(n²)) with practical GPU software. Reviewers accepted that the ideas were not individually groundbreaking — they valued the clever synthesis. But they pushed for more thorough convergence/quality plots.

______________________________________________________________________

### 5.3 GAUCHE (NeurIPS 2023) — Reviews

**Scores: 8 (Strong Accept), 7 (Accept), 6 (Weak Accept), 5 (Borderline Accept), 3 (Reject)**

This is the most instructive case — a domain-specific library with no new algorithms that still got accepted despite a Reject score.

**Reviewer giving 8 (Strong Accept):**

- Praised comprehensive exploration of GP representations for chemistry.
- Liked diverse benchmarks (regression, UQ, Bayesian optimization).
- Concern: "Comparison against SOTA neural networks is notably lacking (e.g., ChemProp, ChemBERTa)."
- Limitations noted: "GPR models can be computationally intensive for larger datasets."

**Reviewer giving 3 (Reject) — THE CRITICAL REVIEW:**

- "All of the implemented kernels are known in the literature."
- **"NeurIPS is probably not the right venue for publishing software libraries."**
- "The supported methods are not well documented within the article."
- "The future of the library is also not mentioned: what is the governance model and what are the next steps?"
- "The code provided shows nothing under 'gauche/kernels/graph_kernels'. [...] I cannot count 20 kernels in the code."
- Wanted: Code-documentation consistency, clear roadmap, governance model.

**Reviewer giving 7 (Accept):**

- Soundness: 3/4, Presentation: 4/4, Contribution: 3/4.
- "The proposed library seems to fill a gap in the open-source GP stack. The code cleanliness is rather high and the code is well-tested."
- Concern about code-paper mismatch: "The article is not well-documented when it comes to listing all kernels that are currently supported."

**Reviewer giving 5 (Borderline Accept):**

- Contribution rated only 2/4 (fair).
- "The major weakness is the contribution. The training task uses GPyTorch [...] The main contribution of the paper is to provide some important kernels in chemistry. It seems somewhat incremental."

**Reviewer giving 6 (Weak Accept):**

- "The library's objective is clearly defined and holds potential."
- "I find the objective of the paper somewhat unclear."
- Suggested: Better API documentation, clearer distinction between library paper vs. benchmark paper.

**Authors' rebuttal strategy (successful):**

- Cited NeurIPS 2023 Call for Papers explicitly calling for "libraries" in the Infrastructure section.
- Referenced other library papers published at NeurIPS as precedent.
- Added extensive additional experiments (DNN comparisons, ablation studies, new kernels).
- Detailed the engineering contribution (SIGP wrapper enabling GraKel integration with PyTorch autodiff).

**Key lessons for your paper:**

1. **The "is this the right venue?" attack will come.** Prepare a rebuttal citing the NeurIPS CFP's explicit mention of software/infrastructure.
2. **Code must match paper claims exactly.** Reviewers will inspect your repository and call out discrepancies.
3. **Document a roadmap and governance model** for the library.
4. **Demonstrate engineering novelty**, not just wrapping existing code. GAUCHE survived by showing the non-trivial SIGP wrapper.
5. **Compare against deep learning baselines**, even if your method is classical/statistical.

______________________________________________________________________

### 5.4 LightGBM (NeurIPS 2017) — Reviews

**All 3 reviewers positive. Accepted.**

**Reviewer 1:**

- "The approaches are interesting and smart."
- Criticism: "The experiments lack standard deviation indications, for the performances are often very close from one method to another."
- Wanted noise sensitivity analysis, especially for GOSS.
- "The paper is not easy to read, because the presentation, explanation, experimental study and analysis are spread all over the paper. It would be better to concentrate fully on one improvement, then on the other."

**Reviewer 2:**

- "This paper is well-motivated, and the proposed algorithm is interesting and effective."
- Wanted: "details on the datasets and why these datasets are selected."
- Wanted: generalization analysis, not just approximation error bounds.

**Reviewer 3:**

- "The proposed study here is well motivated and of important practical value."
- Also wanted generalization error analysis beyond approximation error.

**Key lesson for your paper:** Even for a highly successful paper, reviewers wanted error bars, dataset justification, and cleaner paper organization. Theoretical bounds should address generalization, not just approximation.

______________________________________________________________________

### 5.5 Benchopt (NeurIPS 2022) — Reviews

**Scores: 7, 7, 4 (Borderline Reject). Accepted.**

**Reviewer giving 7 (Accept):**

- "The authors identify an important problem in machine learning, the benchmarking and comparisons of optimizers."
- Concern: "The discussion of the results and the specific experiments are not dramatically different compared to similar optimizer benchmarking papers."
- Wanted: "What makes using Benchopt better than simply writing a custom script?"

**Reviewer giving 7 (Accept):**

- Praised: "The modular design of the benchmark guarantees the versatility."
- Praised: "Interesting findings such as the unexpected performance of baseline optimization methods like L-BFGS."
- Wanted: more problem types beyond the three demonstrated.

**Reviewer giving 4 (Borderline Reject):**

- "I commend the effort to improve transparency and reproducibility in OptML research."
- Key criticism: "I think it is a bad choice to report comparisons in terms of wall-clock-time. This can be extremely dependent on specific combinations of framework, implementation and hardware."
- Wanted: comparisons in number of iterates + reported time per iteration, to separate algorithmic insights from implementation details.
- Felt the paper was "more like a technical report or software documentation than a research paper."

**Key lesson for your paper:** Benchmarking methodology matters as much as results. Separate algorithmic cost from wall-clock time if possible. Address the "is this a research paper or documentation?" concern head-on.

______________________________________________________________________

## 6. Structural Patterns of Successful NeurIPS Software Papers

Based on analysis of the actual papers and reviewer feedback above:

### Paper Structure Comparison

**PyTorch (12 pages, NeurIPS 2019):**

1. Introduction
2. Design Principles (Be Pythonic, Put Researchers First, Provide Pragmatic Performance, Worse is Better)
3. Usability Centric Design (Deep Learning Specific, Separation of Control and Data Flow)
4. Technical Implementation (Internals: autograd, JIT compiler, C++ frontend)
5. Key Subsystems (Automatic Differentiation, Data Loading, Serialization, Multiprocessing)
   - 5.3: GPU Memory Management (custom caching allocator)
6. Evaluation
   - 6.1: Asynchronous Dataflow vs. Eager Mode benchmarks
   - 6.2: Memory Management evaluation
   - 6.3: Benchmarks on common models (AlexNet, VGG-19, ResNet-50, etc.) vs. TensorFlow/MXNet
   - 6.4: Adoption metrics (papers, GitHub contributors)
7. Conclusion and Future Work

**GPyTorch (9 pages + supplementary, NeurIPS 2018):**

1. Introduction
2. Numerical Methods Background (CG, Lanczos, SLQ)
3. GP Inference Background (marginal log-likelihood, predictive distributions)
4. Blackbox Matrix-Matrix (BBMM) Inference — the core algorithmic contribution
   - 4.1: Modified Batched Conjugate Gradients
   - 4.2: Preconditioning via Pivoted Cholesky
   - 4.3: Convergence Analysis
5. The GPyTorch Software Platform
6. Experiments
   - 6.1: Exact GPs (speed + accuracy vs. baselines)
   - 6.2: Scalable GP Approximations (KISS-GP, SGPR)
   - 6.3: Deep Kernel Learning
7. Discussion and Conclusion

**Key structural insight:** GPyTorch spends 2 full sections on math background before introducing the algorithmic contribution. The software description is only 1 section. Experiments are comprehensive (3 subsections covering different GP regimes).

### What Makes Software Papers Stand Out at NeurIPS

- **Algorithmic novelty paired with practical utility**: Pure "wrapper" libraries face strong pushback (see GAUCHE Reviewer 3). There must be a technical contribution (GPyTorch's BBMM, LightGBM's GOSS/EFB).
- **Frame as design principles, not a tutorial**: PyTorch Reviewer 2 explicitly praised this distinction over a "white paper."
- **Ecosystem integration**: Successful packages build on PyTorch, NumPy, or scikit-learn rather than reinventing the wheel.
- **Reproducibility**: Code availability, pip-installable packages, clear documentation. Reviewers will check your repo!
- **Benchmark rigor**: Error bars (LightGBM criticism), convergence plots (GPyTorch criticism), wall-clock vs. algorithmic cost separation (Benchopt criticism).
- **Community adoption evidence**: Download stats, GitHub stars, usage in other papers.
- **Clear API examples**: Short code snippets showing how the library is used in practice.
- **Address the venue question preemptively**: Multiple papers faced "is NeurIPS the right venue?" Cite the CFP.

### NeurIPS Submission Practicalities (2025 Guidelines)

- 9 content pages + unlimited references and checklist
- LaTeX only (neurips_2025.sty)
- Double-blind review
- Mandatory NeurIPS Paper Checklist (reproducibility, ethics, broader impact)
- Code submission encouraged but not required (strongly recommended for software papers)
- Datasets & Benchmarks Track is an alternative venue that welcomes tool contributions
- The 2023+ CFP explicitly mentions "libraries" under Infrastructure contributions

______________________________________________________________________

## 7. Common Reviewer Criticisms for Software Papers — Checklist

Based on all the reviews analyzed above, here is a checklist of attacks to anticipate and preempt:

| Criticism                                      | Frequency   | How to Preempt                                                                                       |
| ---------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------- |
| "Is NeurIPS the right venue for this?"         | Very common | Cite CFP's infrastructure/library mention; reference precedent papers                                |
| "The algorithms are known / not novel"         | Very common | Clearly articulate what IS novel (the combination, the GPU implementation, the theoretical analysis) |
| "This is just a wrapper around X"              | Common      | Show non-trivial engineering (custom allocators, new preconditioners, integration challenges)        |
| "Missing comparison to deep learning / SOTA"   | Common      | Include DNN baselines even if your method is classical                                               |
| "Code doesn't match paper claims"              | Common      | Audit your repo before submission; every feature mentioned must be verifiable                        |
| "No error bars / standard deviations"          | Common      | Always report mean ± std over multiple runs/splits                                                   |
| "Wall-clock benchmarks are hardware-dependent" | Common      | Report both wall-clock AND algorithmic complexity; test on multiple hardware                         |
| "Paper reads like documentation"               | Occasional  | Focus on design principles and WHY, not HOW-TO                                                       |
| "No roadmap / future governance"               | Occasional  | Include a brief roadmap section; mention maintenance plans                                           |
| "Limited experimental scope"                   | Common      | Cover multiple datasets, problem sizes, and settings                                                 |
| "Convergence/quality plots missing"            | Occasional  | Show how approximation quality evolves, not just final numbers                                       |
| "No generalization analysis"                   | Occasional  | Provide theory beyond approximation error if applicable                                              |

______________________________________________________________________

## 8. Recommendations for Your Bispectrum Paper

Based on this deep analysis of actual papers and reviewer feedback:

### Structure (follow GPyTorch template)

1. **Introduction** — Frame the bispectrum computation problem, why existing tools are insufficient, and your 3-4 key contributions.
2. **Background** — Mathematical foundations of the bispectrum (1-2 pages). Reviewers appreciate solid grounding.
3. **Algorithmic Contribution** — What is novel about your approach? New algorithms, GPU parallelization strategy, complexity reduction? This is the heart of the paper.
4. **Software Design** — Brief (~1 page) section on API design, ecosystem integration (PyTorch/JAX/NumPy). Include a code snippet.
5. **Experiments** — At least 3 subsections: (a) speed benchmarks with error bars, (b) accuracy/correctness verification, (c) scaling behavior. Compare CPU vs GPU. Compare against existing implementations if any.
6. **Conclusion & Future Work** — Include a brief roadmap.

### Preempt the Top Reviewer Attacks

1. **"Not novel enough"** → Clearly state what is algorithmically new (not just "we made a Python package"). Even a clever combination of existing techniques suffices (per GPyTorch Reviewer 3).

2. **"Wrong venue"** → In the introduction, position your work within NeurIPS's scope by connecting bispectrum computation to ML applications (signal processing for neural networks, higher-order statistics in deep learning, etc.). Cite the CFP.

3. **"Just a wrapper"** → Demonstrate non-trivial engineering decisions. GPU memory management, batched computation, numerical stability considerations.

4. **"Missing baselines"** → Benchmark against naive NumPy/SciPy implementations, any existing bispectrum packages, and if possible, show a comparison with a deep-learning-based alternative.

5. **"Code doesn't match"** → Before submission, verify every claim in the paper is demonstrable in the codebase. Include a reproducibility section.

6. **"No error bars"** → Report mean ± std for all benchmarks. Use multiple random seeds and dataset splits.

### Framing Strategy

Follow PyTorch Reviewer 2's insight: write a *design principles paper*, not a *user manual*. Explain WHY your bispectrum library is designed the way it is, what tradeoffs you made, and how those tradeoffs serve the target user community. This separates a NeurIPS paper from documentation.

______________________________________________________________________

## 9. Graphein (NeurIPS 2022 — Accepted) — Reviews

**Paper:** Graphein - a Python Library for Geometric Deep Learning and Network Analysis on Biomolecular Structures and Interaction Networks
**Authors:** Arian Rokkum Jamasb, Ramon Viñas Torné, Eric J Ma, Yuanqi Du, Charles Harris, Kexin Huang, Dominic Hall, Pietro Lio, Tom Leon Blundell
**Venue:** NeurIPS 2022 (Datasets & Benchmarks Track — Accepted)
**Scores: 6, 8, 7, 6**

**Reviewer giving 6 (Weak Accept):**

- Summary: "A python library on modeling protein-protein interaction for target discovery and drug design."
- Strength: "Open access code; beneficial for both target discovery and drug design communities."
- Weakness: "It could be better if this toolbox could provide common methods and make benchmarks on these standard databases."
- Questions: Performance on large-scale databases? Plans for AI baselines?

**Reviewer giving 8 (Strong Accept):**

- "The tool is very well documented and the code seems to be professional-grade."
- "I've managed to run ALL the examples successfully in a couple of days."
- Only weakness: "Lack of a manual for the advanced API."
- This reviewer actually tested the software — a strong signal that reviewers do try to run your code.

**Reviewer giving 7 (Accept):**

- Originality: "Existing tools exhibit limited utilities for geometric deep learning. The proposed tool covers a broader selection."
- Weakness: "More empirical results based on the library need to be collected and shown."
- Wanted: runtime/memory profiling, comparison with TorchDrug, scalability experiments.

**Reviewer giving 6 (Weak Accept):**

- Praised: "Flexible preprocessing pipeline. Excellent and clearly written documentation and notebooks."
- Key criticism: "The paper is mostly a summary of what the library does rather than a scientific contribution."
- "There is limited novelty and no benchmark results to put the datasets on the map."
- "I would want to see the authors actually use the toolkit for something interesting."
- Wanted the paper to demonstrate a use case, not just describe what the library can do.

**Key lessons for your paper:**

1. **One reviewer will actually run your code.** Make sure setup is flawless, examples work, and documentation is good.
2. **"The paper is just a summary of what the library does"** is a common attack. Include empirical results and a use case that demonstrates something interesting/surprising discovered using your library.
3. **Runtime/memory profiling and scalability experiments** are expected for library papers, even in the Datasets & Benchmarks track.
4. **Compare against existing tools** in the same space (TorchDrug, etc.) — don't just describe your tool in isolation.

______________________________________________________________________

## 10. Rejected Library Papers — Autopsy of What Went Wrong

### 10.1 PGLearn (ICLR 2025 — Rejected)

**Paper:** PGLearn - An Open-Source Learning Toolkit for Optimal Power Flow
**Authors:** Michael Klamkin, Mathieu Tanneau, Pascal Van Hentenryck
**Venue:** Submitted to ICLR 2025 — **Rejected**
**Scores: 8, 3, 3**
**OpenReview:** https://openreview.net/forum?id=cecIf0CKnH

**Reviewer giving 8 (Accept):**

- Soundness: 4/4 (excellent), Contribution: 4/4 (excellent).
- "A typical ML for OPF study goes through complex steps of generation, solving, post-processing... This library addresses all stages under a single, standardized platform."
- Domain expert who understood the need.

**Reviewer giving 3 (Reject):**

- Soundness: 1/4 (poor), Contribution: 1/4 (poor).
- Devastating critique: Existing datasets (Google DeepMind OPFData) are larger (13,659 buses vs. PGLearn's 9,241 buses).
- "The authors argue that numerous datasets used in benchmarking [...] have minor issues, and need to be replaced by their benchmark" — reviewer disagreed.
- "I could not find novelty in the proposed work compared to PowerModels, OPFData, etc."
- Attacked the data generation methodology as not capturing realistic temporal variations (diurnal patterns, ENTSO-E data).
- "Technically, the paper does not propose any new scientific contribution."

**Reviewer giving 3 (Reject):**

- Soundness: 2/4 (fair), Contribution: 1/4 (poor).
- "The proposed method is still limited to user-defined synthetic settings."
- "Scalability to other grid models/data distributions, other power system tasks, or settings other than supervised learning is unclear."
- "The novelty is questionable since existing datasets and benchmarks already cover similar scenarios."
- Wanted: real-world data integration, broader task coverage, unsupervised/RL settings.

**Why PGLearn was rejected — the pattern:**

1. **Failed to differentiate from existing tools.** Two reviewers pointed to existing alternatives (OPFData, PowerModels) that already did what PGLearn claimed. The paper didn't convincingly explain why the world needs *yet another* OPF toolkit.
2. **Contribution rated 1/4 (poor) by two reviewers.** When reviewers see the contribution as "just re-implementing what exists," scores collapse.
3. **No novel scientific insight.** The library was useful engineering, but didn't discover anything new or propose a new algorithm.
4. **Narrow scope.** Only supervised learning; only synthetic data. Reviewers wanted broader applicability.
5. **Polarized scores (8 vs. 3, 3).** The domain expert loved it, but ML generalists saw no contribution — a classic failure mode for domain-specific toolkit papers.

______________________________________________________________________

### 10.2 CTBench (ICLR 2025 — Rejected)

**Paper:** CTBench: A Library and Benchmark for Certified Training
**Authors:** (from ICLR 2025 submission)
**Venue:** Submitted to ICLR 2025 — **Rejected**
**Scores: 5, 6, 3, 5**
**OpenReview:** https://openreview.net/forum?id=2bn7gayfz9

**Reviewer giving 5 (Marginally Below):**

- Soundness: 4/4, Presentation: 4/4, Contribution: 2/4 (fair).
- Praised: "Standardizing training schedules, certification methods, and hyperparameter tuning."
- Key weakness: "Although the proposed framework and insights are valuable, they may be better suited for a systems or benchmarking venue (e.g., MLSys, NeurIPS D&B track) rather than a top ML venue like ICLR."
- "The contribution doesn't include a novel algorithm or method — it primarily re-evaluates existing approaches."

**Reviewer giving 6 (Marginally Above):**

- Soundness: 3/4, Presentation: 4/4, Contribution: 2/4 (fair).
- "The field needed this kind of paper."
- But: "The key claim is that all recent methods don't really offer improvements over tuned baselines. This is an important finding but the paper can also be seen as just an extensive hyperparameter search."
- "I'm not sure the benchmark will be widely adopted. [...] different research groups may already have their own internal benchmarking."
- Wanted: more architectures, more datasets, standardized attack/certification methods.

**Reviewer giving 3 (Reject):**

- Soundness: 2/4 (fair), Contribution: 1/4 (poor).
- "The library is restricted to relatively small-scale CIFAR-level experiments."
- "The paper primarily re-implements existing methods with better hyperparameters. This raises the question of whether this is a research contribution or an engineering effort."
- Questioned technical soundness: claims about batch normalization fixes and stochastic weight averaging needed better justification.
- "The analysis sections (loss fragmentation, OOD generalization) lack depth — each could be its own paper if done properly."

**Reviewer giving 5 (Marginally Below):**

- Soundness: 4/4, Presentation: 4/4, Contribution: 2/4 (fair).
- "The amount of experiments and computation required in this paper is beyond impressive."
- But: "The contribution is limited to a re-evaluation of existing methods."
- "The main finding that newer methods don't improve over tuned baselines is interesting but could be due to insufficient benchmarking scope."

**Why CTBench was rejected — the pattern:**

1. **Contribution scored 2/4 or 1/4 by ALL four reviewers.** The fatal weakness: re-implementing existing methods with better hyperparameters is not seen as a research contribution.
2. **"Wrong venue" explicitly raised.** Reviewer 1 suggested MLSys or NeurIPS D&B instead of ICLR.
3. **No new algorithm.** The benchmark was well-executed engineering, but every reviewer noted it proposed nothing methodologically novel.
4. **Limited scale.** Only CIFAR-level experiments; reviewers wanted ImageNet or larger.
5. **Shallow analysis sections.** The paper tried to include analysis of loss fragmentation, OOD generalization, etc., but reviewers felt each was underdeveloped rather than being a strength.
6. **Adoption skepticism.** Reviewer 2 questioned whether the community would actually use the benchmark.

______________________________________________________________________

## 11. Accepted vs. Rejected — The Decisive Differences

| Factor                                  | Accepted (GPyTorch, Benchopt, GAUCHE, Graphein)                                 | Rejected (PGLearn, CTBench)                                |
| --------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Algorithmic novelty**                 | At least one novel element (BBMM, SIGP wrapper, collaborative benchmark design) | Re-implementation of existing methods with better settings |
| **Contribution scores**                 | At least 3/4 from most reviewers                                                | Consistently 1-2/4 from most reviewers                     |
| **Differentiation from existing tools** | Clear gap identified and filled                                                 | Reviewers pointed to existing alternatives                 |
| **Scale of experiments**                | Multiple regimes, problem sizes, GPU/CPU                                        | Narrow scope (one domain, one scale)                       |
| **Something surprising discovered**     | Yes — unexpected findings using the library                                     | Mostly confirmatory results                                |
| **Framing**                             | Design principles, algorithmic contribution                                     | Software description, hyperparameter search                |
| **Venue fit**                           | Justified with CFP citations                                                    | Reviewers suggested alternative venues                     |
| **Adoption evidence**                   | Download stats, community usage, GitHub activity                                | No evidence of adoption                                    |

### The Bottom Line for Your Bispectrum Paper

The single most important factor distinguishing accepted from rejected software papers is **whether the paper contributes a novel scientific insight**, not just a useful tool. Every rejected paper was acknowledged as useful engineering — but that's not enough. Your bispectrum paper must contain at least one of:

- A new algorithm or computational technique (like GPyTorch's BBMM)
- A surprising empirical finding discovered using the library
- A theoretical analysis that provides new understanding
- A significant complexity reduction that enables previously impossible computations

Without one of these, even excellent engineering and documentation will not cross the acceptance threshold.

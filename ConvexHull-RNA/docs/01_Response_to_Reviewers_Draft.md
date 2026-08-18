# Response to Reviewers

> Note: CD-HIT cluster-aware validation is still running. The final response should replace the CD-HIT placeholder with the final cluster-level split performance once `cdhit_id0.8.fa.clstr` and `cdhit_cluster_split_metrics.csv` are available.

## Overview of major revisions

We thank the reviewers for their constructive and detailed comments. In response, we substantially revised the manuscript and added several new control analyses to clarify the scope, interpretation, and reproducibility of ConvexHull-RNA. The main changes are:

1. We reframed high-dimensional convex-hull disjointness as a geometric diagnostic rather than stand-alone proof of biological separability.
2. We added length-control analyses, including length-only models, log(length)-residualized ACNV features, and ablation of the first eight basic count/position features.
3. We added random-feature, label-permutation, dimensionality, and sample-size controls.
4. We added corrected convex-hull controls for the mammalian and multispecies benchmark datasets, as well as supplementary analyses for RNACentral top20 and cRNA/ncRNA classification.
5. We added k-mer/BPE/fusion comparisons and revised the manuscript to describe BPE as an alternative and partly redundant representation rather than a guaranteed complementary improvement.
6. We repositioned coding versus non-coding RNA classification as a first-stage screening module for the online predictor, not as the central contribution of the paper.
7. We added cRNA/ncRNA convex-hull controls and PCA/UMAP visualizations. The UMAP visualization was generated from a balanced subset of 10,000 ENA sequences, including 5,000 cRNA and 5,000 ncRNA samples.
8. We reorganized the code and documentation into a GitHub-ready reproducibility package with example commands and reviewer-control scripts.
9. We initiated CD-HIT-est redundancy control at 80% sequence identity and will add cluster-aware split performance when the clustering finishes.

---

## Response to comment: high-dimensional convex-hull disjointness may be an artifact

We thank the reviewer for raising this important point. We agree that complete convex-hull disjointness in high-dimensional spaces can arise from high-dimensional geometry and finite-sample sparsity, and therefore should not be interpreted alone as biological separability.

To address this concern, we added random Gaussian feature controls, label-permutation controls, dimensionality controls, and repeated subsampling analyses. These analyses showed that observed labels were already strongly organized in low-dimensional ACNV spaces, especially at 24 dimensions. For example, in the corrected mammalian and multispecies benchmark analyses, the observed 24-dimensional disjoint-hull rates were approximately 0.876 and 0.943, respectively, whereas both random-feature and permuted-label controls showed zero disjointness at 24 dimensions. This supports the presence of low-dimensional class organization in ACNV space.

At higher dimensions, however, random and permuted controls also became increasingly disjoint. We therefore revised the manuscript to interpret high-dimensional complete hull disjointness as a geometric diagnostic rather than stand-alone proof of biological separability. The revised manuscript now emphasizes the convergence of classifier performance, length controls, random/permutation controls, dimensionality controls, and redundancy-aware validation.

---

## Response to comment: length confounding

We agree that transcript length can contribute to ncRNA family discrimination and that length-related signals should be controlled explicitly. We therefore added multiple length-control analyses.

First, we trained a length-only model on the RNACentral label_set10 task. This model achieved approximately 0.59 accuracy, confirming that length contains moderate information. However, full ACNV features achieved approximately 0.90 accuracy, indicating that length alone is insufficient to explain the model performance.

Second, we residualized ACNV features by log(length). The residualized ACNV model retained approximately 0.875-0.887 accuracy across seeds. Third, we removed the first eight basic count/position features, after which performance remained approximately 0.893-0.897. Finally, after both removing the first eight features and residualizing by log(length), performance remained approximately 0.876-0.885. These results show that ACNV performance is not driven solely by transcript length or by the most basic count/position features.

---

## Response to comment: k-mer and BPE representations

We thank the reviewer for asking whether fixed k-mer and BPE-based representations are independent alternatives or should be fused. We added a direct comparison using the sampled RNACentral top20 task. We compared 5-mer ACNV, BPE-based ACNV, and a direct concatenation of k-mer+BPE features.

The 5-mer ACNV representation achieved approximately 0.830 accuracy, BPE-based ACNV achieved approximately 0.823 accuracy, and the concatenated k-mer+BPE representation achieved approximately 0.830 accuracy. Thus, direct concatenation did not improve over 5-mer ACNV alone. We therefore revised the manuscript to describe BPE as an alternative representation that is partly redundant with fixed k-mer ACNV under the current classifier setting, rather than claiming that simple feature fusion necessarily improves performance.

---

## Response to comment: coding versus non-coding RNA classification needs more context

We agree that coding versus non-coding RNA classification is a distinct task with its own literature and specialized tools. In the revised manuscript, we reposition this task as a secondary demonstration of ACNV generality and as the first-stage screening module of the online server. The central contribution of the manuscript remains ncRNA family classification.

To address the reviewer’s request, we added a dedicated convex-hull analysis for the cRNA/ncRNA task using the deduplicated ENA binary dataset. Observed cRNA/ncRNA labels showed complete convex-hull disjointness already at 24 and 88 ACNV dimensions, whereas random-feature and permuted-label controls showed no disjointness at those dimensions. This indicates that the coarse coding/non-coding distinction is strongly represented in low-dimensional ACNV space.

We also added PCA and UMAP visualizations of the cRNA/ncRNA ACNV space. The UMAP visualization was generated from a balanced subset of 10,000 ENA sequences, including 5,000 cRNA and 5,000 ncRNA samples. The revised manuscript clarifies that this module is not intended to replace specialized coding-potential predictors, but rather provides a lightweight sequence-only screening step integrated with the ncRNA family prediction framework and the online server.

---

## Response to comment: annotation refinement and possible circularity

We agree that computational reassignment of database labels should be interpreted cautiously. We revised the manuscript to avoid overclaiming. Specifically, we replaced stronger language such as “annotation refinement” or “reassignment” with “computational prioritization for manual curation.” The revised text emphasizes that ACNV-based predictions can identify candidate records for further inspection, but cannot replace Rfam annotation, structural evidence, genomic context, expert curation, or experimental validation.

---

## Response to comment: redundancy and highly similar sequences

We agree that exact duplicate removal is not sufficient to rule out homology-driven performance inflation. We therefore initiated CD-HIT-est clustering at 80% sequence identity for the RNACentral label_set10 dataset. Once clustering finishes, we will construct a cluster-aware train/test split and report the corresponding performance in the revised supplementary material.

**Placeholder for final revision:**  
CD-HIT-est clustering at 80% identity produced [N] clusters. Cluster-aware training/testing achieved [accuracy], [macro-F1], and [MCC]. These results will be added to Supplementary Table Sx.

---

## Response to comment: GitHub and reproducibility

We reorganized the code into a GitHub-ready reproducibility package. The repository now includes installation instructions, input format examples, ACNV feature-construction commands, training commands, prediction commands, reviewer-control scripts, online-server instructions, and a model-card placeholder. Large datasets and trained model weights are not committed directly to GitHub; instead, the repository contains placeholders and instructions for downloading them from an external archive or release.

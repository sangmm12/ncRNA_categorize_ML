# Revised Results Draft

## Length alone does not explain ACNV performance

To examine whether ncRNA family classification was driven primarily by transcript length, we performed a series of length-control experiments on the RNACentral label_set10 benchmark. A length-only classifier achieved moderate performance, with accuracy around 0.59 across random seeds, confirming that transcript length contains family-level information. However, the full ACNV model achieved approximately 0.90 accuracy, substantially outperforming the length-only baseline.

We further residualized ACNV features by log(length). The residualized ACNV model retained approximately 0.875-0.887 accuracy across seeds. Removing the first eight basic count/position features also produced only a modest decrease in performance, with accuracy remaining approximately 0.893-0.897. Combining both controls, i.e., removing the first eight features and residualizing by log(length), still retained approximately 0.876-0.885 accuracy. These results indicate that ACNV captures sequence-organization signals beyond transcript length and the most basic count/position descriptors.

## Random-feature and label-permutation controls support the classifier signal

We next tested whether high-dimensional feature spaces alone could explain the observed classifier performance. Across 24, 88, 344, and 1368 dimensions, true ACNV features consistently outperformed random and permuted controls. In contrast, label-permutation models and random Gaussian features produced near-chance performance, approximately 0.10 accuracy for the 10-class task. Sample-size sensitivity analysis showed that performance increased with the number of training examples per class, supporting the interpretation that the classifier learns structured sequence-derived signals rather than random high-dimensional noise.

## Low-dimensional ACNV spaces show meaningful convex-hull organization

We reran convex-hull controls for the mammalian and multispecies benchmark datasets using the corrected feature-column index. At 24 dimensions, observed labels showed strong disjoint-hull organization: the mammalian benchmark had an average disjoint rate of approximately 0.876, and the multispecies benchmark had an average disjoint rate of approximately 0.943. In contrast, label-permutation and random-feature controls showed zero disjointness at 24 dimensions.

At higher dimensions, complete or near-complete convex-hull disjointness was also observed in random-feature or permuted-label controls. We therefore revised our interpretation: high-dimensional complete hull disjointness is not used as stand-alone evidence of biological separability. Instead, convex-hull analysis is treated as a geometric diagnostic, and the low-dimensional results are emphasized because they show class organization before random high-dimensional effects dominate.

The RNACentral top20 analysis showed a similar pattern. Observed labels had high 24-dimensional disjointness, whereas random and permuted controls had zero disjointness at 24 dimensions. At 88 dimensions and above, random and permuted controls also became fully disjoint, reinforcing the need for dimensionality-matched controls.

## k-mer and BPE are alternative but partly redundant representations

We compared 5-mer ACNV, BPE-based ACNV, and direct k-mer+BPE concatenation on the sampled RNACentral top20 task. The 5-mer ACNV model achieved approximately 0.830 accuracy, BPE-based ACNV achieved approximately 0.823 accuracy, and direct concatenation achieved approximately 0.830 accuracy. Thus, simple feature concatenation did not improve over 5-mer ACNV alone. We therefore interpret BPE-based ACNV as an alternative representation that is partly redundant with fixed k-mer ACNV under the current classifier setting.

## Coding versus non-coding classification as a first-stage screening module

The coding versus non-coding RNA task was included as a secondary module and as the first-stage screening component of the online predictor. To clarify its role, we added a dedicated convex-hull analysis and PCA/UMAP visualizations on the deduplicated ENA binary dataset.

Observed cRNA/ncRNA labels showed complete convex-hull disjointness at 24 and 88 ACNV dimensions, whereas random-feature and label-permutation controls showed no disjointness at those dimensions. This indicates that the coarse coding/non-coding distinction is strongly represented in low-dimensional ACNV space. At 344 and 1368 dimensions, random and permuted controls also became disjoint, again supporting cautious interpretation of high-dimensional hull results.

The UMAP visualization was generated from a balanced subset of 10,000 ENA sequences, including 5,000 cRNA and 5,000 ncRNA samples. The two classes formed clearly separated regions in the UMAP projection. These results support the use of the binary module as an integrated first-stage screening step, while the revised manuscript explicitly states that this module is not intended to replace specialized coding-potential tools.

## CD-HIT cluster-aware validation

CD-HIT-est clustering at 80% sequence identity is currently running for the RNACentral label_set10 dataset. Once clustering finishes, we will construct cluster-aware train/test splits and report the resulting performance metrics.

**Placeholder:** Cluster-aware validation results will be inserted here.

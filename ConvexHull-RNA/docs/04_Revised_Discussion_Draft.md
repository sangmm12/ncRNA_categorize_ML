# Revised Discussion Draft

## Interpretation of convex-hull separability

The revised analyses show that ACNV features produce meaningful low-dimensional organization of ncRNA classes. In the mammalian, multispecies, RNACentral top20, and cRNA/ncRNA analyses, observed labels showed strong disjoint-hull organization at low dimensions, especially at 24 dimensions, whereas random-feature and permuted-label controls did not. This supports the idea that ACNV captures biologically relevant sequence-organization signals.

However, the control analyses also show that complete convex-hull disjointness in high-dimensional spaces can occur even for random or permuted data. We therefore no longer interpret high-dimensional complete hull separation as stand-alone evidence of biological separability. Instead, convex-hull analysis is used as a geometric diagnostic that must be interpreted together with classifier performance, length controls, random-feature controls, label-permutation controls, dimensionality controls, and redundancy-aware validation.

## Length and other confounding factors

Transcript length contributes to ncRNA family classification, as shown by the moderate performance of the length-only model. Nevertheless, ACNV models retain high performance after log(length) residualization and after removing the first eight basic count/position features. This suggests that ACNV captures sequence-organization patterns beyond length alone. Future work should continue to evaluate additional confounders, including species composition, family imbalance, genomic context, and database-specific annotation biases.

## k-mer and BPE representations

The BPE-based ACNV representation provides an alternative tokenization strategy, but direct concatenation of k-mer and BPE features did not improve performance over 5-mer ACNV alone in the sampled RNACentral top20 experiment. This suggests that the two representations may capture partly overlapping information under the current classifier setting. Future work could explore more structured fusion approaches, such as attention-based fusion, representation learning, or task-specific tokenization.

## Role of coding versus non-coding classification

The coding versus non-coding module is not the central contribution of ConvexHull-RNA. Rather, it serves as a practical first-stage screening step in the online prediction workflow. The cRNA/ncRNA convex-hull and UMAP analyses show that ACNV captures strong coarse separation between coding and non-coding sequences. Nevertheless, specialized coding-potential predictors such as CPAT-, CPC-, or RNAsamba-like tools address a mature and distinct problem. ConvexHull-RNA should therefore be viewed as an integrated sequence-only screening and ncRNA family-prediction framework, not as a replacement for dedicated coding-potential tools.

## Annotation prioritization rather than automatic reassignment

The revised manuscript avoids claiming that ACNV predictions can automatically refine or replace database annotations. Instead, high-confidence disagreements between the model and the database label should be interpreted as computational prioritization for manual curation. Such cases may be useful for identifying candidates for further review, but final reassignment requires independent evidence such as Rfam annotation, secondary structure, genomic context, comparative genomics, expert curation, or experimental validation.

## Reproducibility and web deployment

We reorganized the code into a GitHub-ready structure with example commands, reviewer-control scripts, online prediction code, and documentation. The online server reports both the coding probability and the ncRNA subtype prediction, and users can download the prediction CSV. Large datasets and trained model weights should be distributed through an external archive or release rather than committed directly to GitHub.

## Limitations and future work

Several limitations remain. First, CD-HIT cluster-aware validation is still being completed and will be added to the final revision. Second, direct comparison against specialized coding-potential tools remains a useful future extension for the binary screening module. Third, the annotation-prioritization analysis requires independent biological validation. Finally, although ACNV provides interpretable sequence-derived features, future models may benefit from integrating structure, genomic context, evolutionary conservation, and external RNA family annotations.

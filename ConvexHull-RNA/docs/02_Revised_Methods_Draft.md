# Revised Methods Draft

## ACNV feature representation

Each RNA sequence was converted into an accumulated convex-hull nucleotide vector (ACNV) representation. Briefly, for a given sequence and tokenization scheme, nucleotide or token occurrence patterns were summarized using count, positional, and higher-order accumulated distribution descriptors. For the fixed k-mer representation used in the main models, 5-mer ACNV features generated a 1,368-dimensional feature vector (NV1368). Unless otherwise stated, the first metadata columns were excluded from the feature matrix and only the numerical ACNV columns were used for model training and geometric analysis.

For RNACentral datasets with columns `GeneName, Label, Species, Length, Sequence, V1, ..., V1368`, ACNV features begin at column index 5. For mammalian and multispecies benchmark files with columns `Sequence, Label, Source, V1, ..., V1368`, ACNV features begin at column index 3. For the ENA binary cRNA/ncRNA dataset with columns `GeneName, Label, Length, Sequence, V1, ..., V1368`, ACNV features begin at column index 4.

## ncRNA family classification

For ncRNA family classification, ACNV features were used to train supervised classifiers on RNACentral family labels. The primary RNACentral label_set10 benchmark used 10 ncRNA families with balanced sampling across families. A larger top20 RNACentral benchmark was also used for representation comparisons. Model performance was evaluated using accuracy, macro-F1, weighted-F1, and Matthews correlation coefficient (MCC). Multiple random seeds were used where appropriate.

## Coding versus non-coding RNA screening

The coding versus non-coding RNA classifier was included as a secondary module and as the first-stage screening component of the online server. This task was not treated as the central contribution of the manuscript. The binary model was trained using the deduplicated ENA cRNA/ncRNA dataset. The online server reports the probability that an input sequence is coding RNA and also provides an ncRNA subtype prediction using the family classifier.

## Length-control analysis

To test whether ACNV performance was driven by transcript length, we performed four length-control analyses on the RNACentral label_set10 benchmark.

First, a length-only baseline was trained using only the `Length` column. Second, the full ACNV feature matrix was trained under the same split. Third, ACNV features were residualized by log(length) using linear regression fitted on the training set and applied to the test set. Fourth, the first eight basic count/position features were removed. We also combined the feature-ablation and log(length)-residualization controls. All models were evaluated using the same metrics as the main family-classification task.

## Random-feature and label-permutation controls

To determine whether classifier performance could be explained by high dimensionality alone, we added random-feature and label-permutation controls. For each ACNV dimensionality subset (24, 88, 344, and 1368 dimensions), models were trained using the true labels, permuted labels, and random Gaussian features with matched dimensions. We also performed sample-size sensitivity experiments by varying the number of sequences per class.

## Convex-hull separability analysis

Pairwise convex-hull intersection was evaluated using a linear-programming feasibility formulation. For two classes with point sets A and B, we tested whether there exist convex coefficients over A and B such that the two convex combinations are equal. This avoids explicit vertex enumeration and is computationally feasible for sampled subsets.

We performed convex-hull controls on the mammalian and multispecies benchmark datasets, RNACentral top20, and the ENA cRNA/ncRNA dataset. For each dataset, we evaluated observed labels, label-permutation controls, and random Gaussian feature controls across ACNV dimensionalities. Because complete convex-hull disjointness can arise in high-dimensional spaces even for random data, the revised manuscript interprets high-dimensional hull disjointness as a geometric diagnostic rather than stand-alone biological evidence. Low-dimensional results, especially 24-dimensional ACNV controls, were emphasized because random and permuted controls remained non-disjoint in that setting.

## k-mer, BPE, and feature-fusion comparison

To evaluate whether fixed k-mer ACNV and BPE-based ACNV provide complementary information, we compared 5-mer ACNV, BPE-based ACNV, and direct concatenation of k-mer+BPE features on the sampled RNACentral top20 task. The same classifier and evaluation protocol were used for all three representations.

## cRNA/ncRNA visualization

To visualize the binary coding versus non-coding task, a balanced subset of 10,000 ENA sequences was sampled, including 5,000 cRNA and 5,000 ncRNA sequences. ACNV features were standardized and projected into two dimensions using PCA and UMAP. These visualizations were used to illustrate the coarse separation of the two classes and to complement the convex-hull analysis.

## CD-HIT redundancy-control analysis

CD-HIT-est clustering at 80% sequence identity was initiated on the RNACentral label_set10 dataset. The resulting clusters will be used to construct cluster-aware train/test splits. Cluster-aware performance will be reported once CD-HIT clustering is complete.

**Placeholder:** The final manuscript will include the number of clusters, cluster-level train/test sizes, and cluster-aware performance metrics.

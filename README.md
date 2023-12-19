# Mid_bootcamp_project

# General information:
The midbootcamp project: Can we predict cancer using gene expression profile?

# To approach this objective:
1. select gene expression datasets for different types of cancers
2. Data checking, cleaning, transform if needed
3. Identify genes that are differentially expressed (called DEGs) in cancer samples vs normal samples. Using two samples t-test at p_sig = 0.05
4. Check and excluding genes that are highly correlated among the identified DEGs subset with a threshold for exluding at 0.95
5. Split and train model on working datasets:
    * using transformed data (using quantile transformation) vs non-transformed data 
6. Validation:
   * on the whole (train + test) dataset
   * using a new dataset of the same cancer type
   * calculation all validation metrics: precision, accuracy, recel, F1, cohen_kappa_score
     
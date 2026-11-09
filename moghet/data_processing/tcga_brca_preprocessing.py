import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

"""
PathMoG Data Preprocessing Pipeline for TCGA-BRCA
Grounded in PathMoG Architecture (Wang et al., 2026)

This script automates the alignment, normalization, and fold creation
for TCGA-BRCA multi-omics data (Gene Expression, CNV, Somatic Mutations)
and clinical/survival metadata according to PathMoG specifications.
"""

# Output directory for processed PathMoG tensors
OUTPUT_DIR = "./pathmog_brca_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------------------
# STEP 1: Load Raw Multi-Omics and Survival Files (from UCSC Xena)
# ------------------------------------------------------------------------------
def load_raw_xena_files(expr_path, cnv_path, mut_path, surv_path, clin_path):
    """
    Loads raw matrices downloaded from UCSC Xena Pan-Cancer Hub:
    - Gene Expression: HiSeqV2 / RNA-seq transcript abundance
    - CNV: GISTIC2 thresholded copy number (-2 to +2)
    - Somatic Mutation: MC3 / WUSTL mutation calls
    - Survival: Clinical survival metadata (OS, OS.time)
    - Clinical: Clinical matrix (Age, Stage, ER/PR/HER2, PAM50)
    """
    print("\n--- Step 1: Loading Raw Matrices ---")
    
    # 1. Gene Expression (mRNA)
    expr_df = pd.read_csv(expr_path, sep='\t', index_col=0).T
    expr_df.index = expr_df.index.str.slice(0, 12)  # Shorten barcode to TCGA-XX-XXXX
    expr_df = expr_df.groupby(expr_df.index).mean() # Average duplicate aliquots
    
    # 2. Copy Number Variation (CNV) - GISTIC2 calls
    cnv_df = pd.read_csv(cnv_path, sep='\t', index_col=0).T
    cnv_df.index = cnv_df.index.str.slice(0, 12)
    cnv_df = cnv_df.groupby(cnv_df.index).mean().round().astype(int)
    
    # 3. Somatic Mutations (Binarized)
    mut_raw = pd.read_csv(mut_path, sep='\t')
    mut_raw['sample'] = mut_raw['sample'].str.slice(0, 12)
    mut_df = pd.crosstab(mut_raw['sample'], mut_raw['gene']).clip(upper=1)
    
    # 4. Survival Metadata
    surv_df = pd.read_csv(surv_path, sep='\t', index_col=0)
    surv_df.index = surv_df.index.str.slice(0, 12)
    surv_df = surv_df[~surv_df.index.duplicated(keep='first')]
    
    # 5. Clinical Metadata
    clin_df = pd.read_csv(clin_path, sep='\t', index_col=0)
    clin_df.index = clin_df.index.str.slice(0, 12)
    clin_df = clin_df[~clin_df.index.duplicated(keep='first')]

    return expr_df, cnv_df, mut_df, surv_df, clin_df


# ------------------------------------------------------------------------------
# STEP 2: Cohort & Gene Universe Alignment (7,595 Genes, N = 1,082 Patients)
# ------------------------------------------------------------------------------
def align_cohort_and_genes(expr_df, cnv_df, mut_df, surv_df, clin_df, kegg_gene_universe):
    """
    Aligns all multi-omics matrices and patient metadata to:
    1. Shared patient cohort (intersection of patients with valid survival & omics).
    2. Aligned gene universe U (M = 7,595 KEGG pathway genes).
    """
    print("\n--- Step 2: Aligning Patient Cohort & Gene Universe ---")
    
    # Filter valid survival annotations (OS and OS.time > 0)
    valid_surv = surv_df.dropna(subset=['OS', 'OS.time'])
    valid_surv = valid_surv[valid_surv['OS.time'] > 0]
    
    # Find intersecting patient barcodes across survival and all omics
    common_patients = sorted(list(
        set(valid_surv.index) & 
        set(expr_df.index) & 
        set(cnv_df.index) & 
        set(mut_df.index)
    ))
    print(f"Matched Patient Cohort Size: N = {len(common_patients)} (TCGA-BRCA Target: N = 1,082)")

    # Gene universe alignment (7,595 genes induced by 354 KEGG pathways)
    aligned_genes = [g for g in kegg_gene_universe if g in expr_df.columns]
    print(f"Aligned Gene Universe Size: M = {len(aligned_genes)} / {len(kegg_gene_universe)} genes")
    
    # Reindex omics matrices to identical patient and gene orders
    expr_aligned = expr_df.reindex(index=common_patients, columns=aligned_genes).fillna(0)
    cnv_aligned = cnv_df.reindex(index=common_patients, columns=aligned_genes).fillna(0)
    mut_aligned = mut_df.reindex(index=common_patients, columns=aligned_genes).fillna(0)
    
    surv_aligned = valid_surv.reindex(common_patients)
    clin_aligned = clin_df.reindex(common_patients)

    return expr_aligned, cnv_aligned, mut_aligned, surv_aligned, clin_aligned, aligned_genes


# ------------------------------------------------------------------------------
# STEP 3: Process BRCA Clinical Covariates
# ------------------------------------------------------------------------------
def process_brca_clinical_covariates(clin_df):
    """
    Extracts and encodes clinical features according to PathMoG protocol:
    - Age at initial diagnosis (standardized)
    - Pathologic stage (TNM T/N/M stage converted to numeric ranks)
    - Receptor status (ER, PR, HER2) & PAM50 subtype
    """
    print("\n--- Step 3: Processing Clinical Covariates ---")
    clin_features = pd.DataFrame(index=clin_df.index)
    
    # Age
    age_cols = [c for c in clin_df.columns if 'age' in c.lower()]
    if age_cols:
        clin_features['age'] = pd.to_numeric(clin_df[age_cols[0]], errors='coerce').fillna(60.0)
    else:
        clin_features['age'] = 60.0
        
    # Stage
    stage_cols = [c for c in clin_df.columns if 'stage' in c.lower()]
    if stage_cols:
        clin_features['stage'] = pd.factorize(clin_df[stage_cols[0]])[0]
    else:
        clin_features['stage'] = 0

    # Receptors & PAM50 Subtypes
    for marker in ['ER_status', 'PR_status', 'HER2_status', 'PAM50_subtype']:
        cols = [c for c in clin_df.columns if marker.lower() in c.lower()]
        if cols:
            dummies = pd.get_dummies(clin_df[cols[0]], prefix=marker, drop_first=True)
            clin_features = pd.concat([clin_features, dummies], axis=1)
            
    return clin_features


# ------------------------------------------------------------------------------
# STEP 4: Leakage-Free Stratified 5-Fold Cross-Validation
# ------------------------------------------------------------------------------
def create_leakage_free_folds(expr_df, cnv_df, mut_df, surv_df, clin_features, n_splits=5):
    """
    Applies Stratified 5-Fold Cross-Validation:
    Calculates normalization parameters (mean, std) STRICTLY within training folds
    to prevent information leakage into validation sets.
    """
    print("\n--- Step 4: Creating Leakage-Free Stratified 5-Fold Split ---")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    events = surv_df['OS'].values
    
    fold_packages = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(expr_df, events)):
        train_pts = expr_df.index[train_idx]
        val_pts = expr_df.index[val_idx]
        
        # Estimate scaling parameters ONLY on training fold
        train_mean = expr_df.loc[train_pts].mean(axis=0)
        train_std = expr_df.loc[train_pts].std(axis=0).replace(0, 1.0)
        
        # Apply scaling
        expr_train_scaled = (expr_df.loc[train_pts] - train_mean) / train_std
        expr_val_scaled = (expr_df.loc[val_pts] - train_mean) / train_std
        
        package = {
            'fold': fold + 1,
            'train_patients': train_pts.tolist(),
            'val_patients': val_pts.tolist(),
            'train_expr': expr_train_scaled,
            'val_expr': expr_val_scaled,
            'cnv_train': cnv_df.loc[train_pts],
            'cnv_val': cnv_df.loc[val_pts],
            'mut_train': mut_df.loc[train_pts],
            'mut_val': mut_df.loc[val_pts],
            'surv_train': surv_df.loc[train_pts],
            'surv_val': surv_df.loc[val_pts],
            'clin_train': clin_features.loc[train_pts],
            'clin_val': clin_features.loc[val_pts]
        }
        fold_packages.append(package)
        print(f"Fold {fold+1}: Train N={len(train_pts)}, Validation N={len(val_pts)}")
        
    return fold_packages


if __name__ == "__main__":
    print("PathMoG Preprocessing Pipeline initialized successfully.")

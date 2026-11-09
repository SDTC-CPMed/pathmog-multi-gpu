import os
import sys
import argparse
import urllib.request
import gzip
from pathlib import Path
from urllib.parse import quote

"""
Universal UCSC Xena Multi-Omics & Clinical Data Downloader for PathMoG
====================================================================
This script automates fetching the 5 essential TCGA datasets required by PathMoG
for any specified TCGA cancer cohort (e.g., BRCA, PRAD, COAD, KIRC, LUAD, etc.).

Downloaded Datasets per Cohort:
  1. Gene Expression (RNAseq HiSeqV2, log2(x+1) transformed)
  2. Somatic Mutations (MC3 gene-level non-silent)
  3. Copy Number Variations (GISTIC2 thresholded calls: -2 to +2)
  4. Curated Survival Data (Overall Survival duration and censoring event)
  5. Clinical Phenotype Matrix (Stage, Age, Receptors, Subtypes/Gleason)

Downloads use the AWS S3 Xena mirror (region-friendly vs tcga.xenahubs.net):
  https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/<path with %2F>
"""

# AWS S3 mirror for UCSC Xena TCGA hub (avoids 403s on xenahubs.net in some regions)
XENA_HUB_BASE = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download"

# The 10 TCGA cancer cohorts evaluated in the PathMoG paper
PATHMOG_10_COHORTS = [
    "BRCA", "COAD", "GBM", "KIRC", "LGG",
    "LIHC", "LUAD", "UCEC", "HNSC", "OV"
]


def s3_download_url(object_key: str) -> str:
    """
    Build an S3 download URL with path separators encoded as %2F,
    matching Xena object keys (e.g. survival%2FBRCA_survival.txt).
    """
    # Keep dots/underscores; only encode path separators and unsafe chars
    encoded = quote(object_key, safe="")
    return f"{XENA_HUB_BASE}/{encoded}"


def build_cohort_manifest(cohort: str) -> dict:
    """
    Constructs the dataset manifest with download URLs and target filenames
    for a specified TCGA cohort.
    
    Args:
        cohort (str): TCGA cohort abbreviation (e.g. 'BRCA', 'PRAD', 'LUAD').
        
    Returns:
        dict: Manifest mapping dataset key to filename, description, and URL.
    """
    c = cohort.upper().strip()
    return {
        "expression": {
            "filename": f"TCGA.{c}.HiSeqV2.gz",
            "url": s3_download_url(f"TCGA.{c}.sampleMap/HiSeqV2.gz"),
            "description": "RNAseq Gene Expression (log2(x+1) RSEM counts)",
            "compressed": True,
        },
        "mutation": {
            "filename": f"{c}_mc3_gene_level.txt.gz",
            "url": s3_download_url(f"mc3_gene_level/{c}_mc3_gene_level.txt.gz"),
            "description": "Somatic Mutations (MC3 gene-level non-silent)",
            "compressed": True,
        },
        "cnv": {
            "filename": f"TCGA.{c}.Gistic2_thresholded.gz",
            "url": s3_download_url(
                f"TCGA.{c}.sampleMap/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz"
            ),
            "description": "Thresholded Copy Number Variations (GISTIC2 5-level)",
            "compressed": True,
        },
        # Survival / clinical are stored uncompressed on the S3 mirror
        "survival": {
            "filename": f"{c}_survival.txt",
            "url": s3_download_url(f"survival/{c}_survival.txt"),
            "description": "Curated Survival Metadata (OS duration and status)",
            "compressed": False,
        },
        "clinical": {
            "filename": f"TCGA.{c}.clinicalMatrix",
            "url": s3_download_url(f"TCGA.{c}.sampleMap/{c}_clinicalMatrix"),
            "description": "Phenotype and Clinical Metadata",
            "compressed": False,
        },
    }


def download_and_extract_cohort(cohort: str, output_root: str = "./tcga_data", decompress: bool = True):
    """
    Downloads and extracts all 5 PathMoG datasets for a given cohort.
    
    Args:
        cohort (str): TCGA cohort code.
        output_root (str): Output parent directory.
        decompress (bool): Whether to auto-extract .gz files into raw TSV format.
    """
    c = cohort.upper().strip()
    cohort_dir = Path(output_root) / c
    cohort_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = build_cohort_manifest(c)
    
    print("\n" + "=" * 65)
    print(f"  PathMoG Universal Downloader: TCGA-{c}")
    print(f"  Mirror: {XENA_HUB_BASE}")
    print(f"  Target Folder: {cohort_dir.resolve()}")
    print("=" * 65)
    
    for key, info in manifest.items():
        download_path = cohort_dir / info["filename"]
        url = info["url"]
        desc = info["description"]
        is_compressed = info.get("compressed", download_path.suffix == ".gz")
        
        print(f"\n[+] [{key.upper()}] {desc}")
        print(f"    URL: {url}")
        
        if is_compressed:
            final_path = cohort_dir / download_path.stem
        else:
            final_path = download_path
        
        # Already have usable plain file
        if final_path.exists() and (not is_compressed or final_path.stat().st_size > 0):
            print(f"    -> Ready file exists ({final_path.name}), skipping download.")
            continue
            
        if download_path.exists():
            print(f"    -> File exists ({download_path.name}), skipping download.")
        else:
            print(f"    -> Downloading from UCSC Xena S3 mirror...")
            try:
                urllib.request.urlretrieve(url, download_path)
                mb_size = download_path.stat().st_size / (1024 * 1024)
                print(f"    [SUCCESS] Saved {download_path.name} ({mb_size:.2f} MB)")
            except Exception as err:
                print(f"    [ERROR] Failed to fetch {url}: {err}")
                continue
                
        # Decompress only gzipped downloads
        if decompress and is_compressed and download_path.exists():
            print(f"    -> Decompressing to {final_path.name}...")
            try:
                with gzip.open(download_path, 'rb') as gz_in:
                    with open(final_path, 'wb') as f_out:
                        f_out.write(gz_in.read())
                print(f"    [SUCCESS] Extracted {final_path.name}")
            except Exception as err:
                print(f"    [ERROR] Extraction failed for {download_path.name}: {err}")


def main():
    parser = argparse.ArgumentParser(
        description="Universal UCSC Xena Multi-Omics & Clinical Data Downloader for PathMoG"
    )
    parser.add_argument(
        "--cohort", 
        type=str, 
        default="BRCA", 
        help="TCGA cohort abbreviation (e.g., BRCA, PRAD, COAD, KIRC, LUAD, LGG, GBM, LIHC, UCEC, HNSC, OV)"
    )
    parser.add_argument(
        "--outdir", 
        type=str, 
        default="./tcga_data", 
        help="Root directory where downloaded cohorts will be saved"
    )
    parser.add_argument(
        "--all-pathmog", 
        action="store_true", 
        help="Download all 10 TCGA cancer cohorts evaluated in the PathMoG paper"
    )
    parser.add_argument(
        "--no-extract", 
        action="store_true", 
        help="Do not automatically decompress downloaded .gz files"
    )
    
    args = parser.parse_args()
    
    if args.all_pathmog:
        print(f"Starting batch download for all 10 PathMoG cohorts: {', '.join(PATHMOG_10_COHORTS)}")
        for ch in PATHMOG_10_COHORTS:
            download_and_extract_cohort(cohort=ch, output_root=args.outdir, decompress=not args.no_extract)
    else:
        download_and_extract_cohort(cohort=args.cohort, output_root=args.outdir, decompress=not args.no_extract)

if __name__ == "__main__":
    main()

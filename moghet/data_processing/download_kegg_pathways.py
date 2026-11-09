#!/usr/bin/env python3
"""
Download and parse human KEGG pathways (KGML) for PathMoG / UroFusion-X training.
Extracts gene-to-pathway mappings, HUGO gene symbols, and typed directed regulatory edges
(activation, inhibition, phosphorylation, etc.) required for PyG HeteroData construction.
"""

import os
import time
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import requests
import pandas as pd
from tqdm import tqdm

# Configuration
KEGG_REST_URL = "https://rest.kegg.jp"
ORGANISM = "hsa"  # Homo sapiens
OUTPUT_DIR = Path("./data/kegg_pathways")
KGML_CACHE_DIR = OUTPUT_DIR / "kgml_raw"
RATE_LIMIT_DELAY = 0.35  # KEGG API allows ~3 requests per second


def fetch_with_retry(url: str, retries: int = 3, delay: float = 1.0) -> str:
    """Fetch text content from KEGG REST API with basic retry logic."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 404:
                return ""
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"Failed to fetch {url}: {e}")
        time.sleep(delay * (attempt + 1))
    return ""


def get_human_pathway_list() -> dict:
    """Retrieve the catalogue of all human pathways from KEGG."""
    print("Fetching human pathway catalogue from KEGG...")
    text = fetch_with_retry(f"{KEGG_REST_URL}/list/pathway/{ORGANISM}")
    pathways = {}
    for line in text.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        # Format: path:hsa04110 \t Cell cycle - Homo sapiens (human)
        pathway_id = parts[0].replace("path:", "").strip()
        pathway_name = parts[1].replace(" - Homo sapiens (human)", "").strip()
        pathways[pathway_id] = pathway_name
    print(f"Found {len(pathways)} human pathways.")
    return pathways


def get_kegg_gene_to_symbol_map() -> dict:
    """
    Build a mapping from KEGG Gene ID (e.g., 'hsa:7157') to HUGO Gene Symbol (e.g., 'TP53')
    so pathway nodes align with UCSC Xena HUGO probeMap matrices.
    """
    print("Fetching KEGG gene ID to HUGO symbol mapping...")
    text = fetch_with_retry(f"{KEGG_REST_URL}/list/{ORGANISM}")
    gene_map = {}
    for line in text.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        kegg_id = parts[0].strip()  # e.g., hsa:7157
        # Format in KEGG: hsa:7157\tCDS\t17p13.1\tTP53, BCC7, BMFS5, LFS1, P53, TRP53; tumor protein p53
        # Or: hsa:7157\tTP53, P53; tumor protein p53
        desc_field = parts[-1]
        if ";" in desc_field:
            symbols_part = desc_field.split(";")[0]
            primary_symbol = symbols_part.split(",")[0].strip()
            if primary_symbol:
                gene_map[kegg_id] = primary_symbol
    print(f"Mapped {len(gene_map)} human genes to HUGO symbols.")
    return gene_map


def normalize_relation_type(rel_elem: ET.Element) -> str:
    """
    Normalize KGML <relation> and <subtype> tags into clean edge relation labels
    for PathMoG's Heterogeneous Graph Transformer (HGT).
    """
    rel_type = rel_elem.attrib.get("type", "interaction")
    subtypes = [sub.attrib.get("name", "") for sub in rel_elem.findall("subtype")]
    subtypes = [s for s in subtypes if s]

    if subtypes:
        # Join multiple subtypes (e.g., 'activation_phosphorylation') or take primary
        norm_label = "_".join(sorted(set(s.replace(" ", "_") for s in subtypes)))
        return norm_label.lower()
    return rel_type.lower()


def parse_kgml(kgml_text: str, pathway_id: str, gene_map: dict):
    """
    Parse a KGML XML string to extract:
      1. Pathway gene set (HUGO symbols)
      2. Typed directed edges (source_gene, target_gene, relation_type)
    """
    root = ET.fromstring(kgml_text)
    entry_to_genes = {}
    pathway_genes = set()

    # 1. Extract gene entries
    for entry in root.findall("entry"):
        if entry.attrib.get("type") == "gene":
            entry_id = entry.attrib["id"]
            kegg_names = entry.attrib.get("name", "").split()
            symbols = [gene_map[k] for k in kegg_names if k in gene_map]
            if symbols:
                entry_to_genes[entry_id] = symbols
                pathway_genes.update(symbols)

    # 2. Extract directed relations between gene entries
    edges = []
    for rel in root.findall("relation"):
        e1 = rel.attrib.get("entry1")
        e2 = rel.attrib.get("entry2")
        if e1 in entry_to_genes and e2 in entry_to_genes:
            rel_label = normalize_relation_type(rel)
            # Expand paralog/complex groups in KGML entries into pairwise gene edges
            for src_gene in entry_to_genes[e1]:
                for dst_gene in entry_to_genes[e2]:
                    if src_gene != dst_gene:
                        edges.append((pathway_id, src_gene, dst_gene, rel_label))

    return sorted(pathway_genes), edges


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    KGML_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    pathways = get_human_pathway_list()
    gene_map = get_kegg_gene_to_symbol_map()

    pathway_modules = {}
    all_edges = []
    global_gene_universe = set()

    print("Downloading and parsing KGML files...")
    for pid, pname in tqdm(pathways.items(), desc="Processing Pathways"):
        kgml_path = KGML_CACHE_DIR / f"{pid}.xml"

        # Load from local cache if already downloaded
        if kgml_path.exists():
            kgml_text = kgml_path.read_text(encoding="utf-8")
        else:
            kgml_text = fetch_with_retry(f"{KEGG_REST_URL}/get/{pid}/kgml")
            if kgml_text:
                kgml_path.write_text(kgml_text, encoding="utf-8")
            time.sleep(RATE_LIMIT_DELAY)

        if not kgml_text or "<pathway" not in kgml_text:
            continue

        genes, edges = parse_kgml(kgml_text, pid, gene_map)

        # PathMoG Rule: Pathways retaining fewer than 2 mapped genes are not instantiated
        if len(genes) < 2:
            continue

        pathway_modules[pid] = {
            "pathway_id": pid,
            "pathway_name": pname,
            "num_genes": len(genes),
            "num_edges": len(edges),
            "genes": genes,
        }
        all_edges.extend(edges)
        global_gene_universe.update(genes)

    # Deduplicate edges
    edges_df = pd.DataFrame(
        all_edges, columns=["pathway_id", "source", "target", "relation"]
    ).drop_duplicates()

    # Save outputs
    modules_file = OUTPUT_DIR / "kegg_pathway_modules.json"
    with open(modules_file, "w", encoding="utf-8") as f:
        json.dump(pathway_modules, f, indent=2)

    edges_file = OUTPUT_DIR / "kegg_pathway_edges.csv"
    edges_df.to_csv(edges_file, index=False)

    universe_file = OUTPUT_DIR / "aligned_gene_universe.txt"
    sorted_universe = sorted(global_gene_universe)
    universe_file.write_text("\n".join(sorted_universe) + "\n", encoding="utf-8")

    print("\n" + "=" * 55)
    print("KEGG Pathway Download & Preprocessing Complete!")
    print(f"Instantiated Pathway Modules (>=2 genes): {len(pathway_modules)}")
    print(f"Total Aligned Genes in Universe (U)     : {len(sorted_universe)}")
    print(f"Total Directed Typed Edges              : {len(edges_df)}")
    print(f"Unique Relation Types (R_k)             : {edges_df['relation'].nunique()}")
    print("-" * 55)
    print(f"Saved pathway dictionary to : {modules_file}")
    print(f"Saved edge list to          : {edges_file}")
    print(f"Saved gene universe to      : {universe_file}")
    print("=" * 55)


if __name__ == "__main__":
    main()
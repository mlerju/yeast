import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path
from tqdm import tqdm
import orfs
from constants import all_codons
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more output
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler("pipeline.log")  # Log file
    ]
)

logger = logging.getLogger(__name__)

# ------------------------ #
#       FILE PATHS         #
# ------------------------ #
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
FASTA_PATH = DATA_DIR / "sacCer3.fasta"
PREDICTED_ORFS_PATH = OUTPUT_DIR / "predicted_orfs.csv"
FEATURES_BASIC_PATH = OUTPUT_DIR / "orf_features_basic.csv"
FEATURES_CODON_PATH = OUTPUT_DIR / "orf_features_with_codon_usage.csv"
FEATURES_KMER_PATH = OUTPUT_DIR / "orf_features_with_kmers.csv"
ORF_FASTA_PATH = "predicted_orfs.fasta"

# ------------------------ #
#        FUNCTIONS         #
# ------------------------ #

def load_genome(fasta_path: Path) -> dict:
    """Load genome sequences from FASTA into a dictionary."""
    return {record.id: record.seq for record in SeqIO.parse(fasta_path, "fasta")}

def predict_orfs(genome: dict, min_len=300) -> list:
    """Scan genome and extract ORFs from each chromosome."""
    all_orfs = []
    for chrom, seq in tqdm(genome.items(), desc="Scanning for ORFs"):
        chrom_orfs = orfs.find_orfs(seq, min_len=min_len)
        for orf in chrom_orfs:
            orf["chromosome"] = chrom
        all_orfs.extend(chrom_orfs)
    return all_orfs

def save_orf_fasta(orfs_list: list, genome: dict, out_path: str):
    """Save ORF sequence to a FASTA file."""
    records = []
    for orf in tqdm(orfs_list, desc="Saving ORF FASTA"):
        chrom_seq = genome[orf['chromosome']]
        seq = orfs.extract_orf_seq(SeqRecord(chrom_seq, id=orf['chromosome']), orf)
        header = f"{orf['chromosome']}_start{orf['start']}_end{orf['end']}_strand{orf['strand']}"
        records.append(SeqRecord(seq, id=header, description=""))
    SeqIO.write(records, out_path, "fasta")

def extract_basic_features(df: pd.DataFrame, genome: dict) -> pd.DataFrame:
    """Add basic sequence features (GC content, strand, frame, sequence)."""
    features = df.apply(lambda row: orfs.get_orf_features(row, genome), axis=1)
    return pd.concat([df, features], axis=1)

def add_codon_usage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add codon usage frequencies and stop codon %."""
    codon_df = df['sequence'].apply(orfs.codon_features).apply(pd.Series)
    return pd.concat([df, codon_df], axis=1)

def add_kmer_features(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Add normalized k-mer frequency features."""
    kmer_df = df['sequence'].apply(lambda s: orfs.get_kmer_counts(s, k)).apply(pd.Series)
    return pd.concat([df, kmer_df], axis=1)

def plot_orf_length_distribution(df: pd.DataFrame):
    """Plot a histogram of ORF lengths."""
    sns.histplot(df['length'], bins=100)
    plt.title("ORF Length Distribution")
    plt.xlabel("Length (bp)")
    plt.ylabel("Frequency")
    plt.show()

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    logger.info("Loading genome...")
    genome = load_genome(FASTA_PATH)

    logger.info("Predicting ORFs...")
    all_orfs = predict_orfs(genome)
    df = pd.DataFrame(all_orfs)
    df.to_csv(PREDICTED_ORFS_PATH, index=False)
    logger.info(f"Saved {len(df)} ORFs to {PREDICTED_ORFS_PATH}")

    logger.info("Saving ORF sequences to FASTA...")
    save_orf_fasta(all_orfs, genome, ORF_FASTA_PATH)

    logger.info("Extracting basic features...")
    df = extract_basic_features(df, genome)
    df.to_csv(FEATURES_BASIC_PATH, index=False)
    logger.info(f"Basic features saved to {FEATURES_BASIC_PATH}")

    logger.info("Adding codon usage features...")
    df = add_codon_usage_features(df)
    df.to_csv(FEATURES_CODON_PATH, index=False)
    logger.info(f"Codon usage features saved to {FEATURES_CODON_PATH}")
    logger.info(df[[c for c in df.columns if c.startswith("codon_")]].mean().sort_values(ascending=False).head(10))

    logger.info("Adding 3-mer frequency features...")
    df = add_kmer_features(df, k=3)
    df.to_csv(FEATURES_KMER_PATH, index=False)
    logger.info(f"3-mer features saved to {FEATURES_KMER_PATH}")

    logger.info("Plotting ORF length distribution...")
    plot_orf_length_distribution(df)

    logger.info("Done!")

if __name__ == "__main__":
    main()
import logging
import os

import gffutils
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
INPUT_CSV = "../outputs/orf_features_basic.csv"
GFF_FILE = "../data/saccharomyces_cerevisiae.gff3"
DB_PATH = "../outputs/yeast_genes.db"
OUTPUT_CSV = "../outputs/orf_labeled.csv"


def create_or_load_gff_db(gff_file, db_path):
    """
    Creates or loads a gffutils database from a GFF3 file.
    """
    if os.path.exists(db_path):
        logger.info("Loading existing GFF database...")
        return gffutils.FeatureDB(db_path)
    else:
        logger.info("Creating GFF database...")
        return gffutils.create_db(
            gff_file,
            dbfn=db_path,
            force=True,
            keep_order=True,
            merge_strategy='merge',
            disable_infer_genes=True,
            disable_infer_transcripts=True
        )


def label_orf(row, db):
    """
    Assigns a label to the ORF based on overlap with known CDS regions in the genome.
    """
    chrom = row['chromosome'].replace("chr", "")
    start = int(row['start']) + 1  # GFF is 1-based
    end = int(row['end'])
    strand = "+" if row['strand'] == 1 else "-"

    try:
        overlapping = list(db.region(region=(chrom, start, end), featuretype='CDS'))
        for feature in overlapping:
            if feature.strand == strand:
                return "known_gene"
        return "novel_orf"
    except Exception as e:
        logger.warning(f"Error checking region {chrom}:{start}-{end}: {e}")
        return "error"


def main():
    """
    Main labeling function that reads predicted ORFs, loads GFF DB,
    applies label based on overlap, and saves the labeled CSV.
    """
    logger.info("Loading predicted ORFs...")
    df = pd.read_csv(INPUT_CSV)

    logger.info("Loading gene annotation DB...")
    db = create_or_load_gff_db(GFF_FILE, DB_PATH)

    logger.info("Applying labels to ORFs...")
    df['label'] = df.apply(lambda row: label_orf(row, db), axis=1)

    # Reorder columns with 'label' to the front
    cols = ['label'] + [col for col in df.columns if col!='label']
    df = df[cols]

    logger.info("Saving labeled ORFs...")
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"✅ Labeled ORFs saved to {OUTPUT_CSV}")
    logger.info(df['label'].value_counts())


if __name__ == "__main__":
    main()

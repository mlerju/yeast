import pandas as pd
import os
import gffutils

# Load predicted ORFs
df = pd.read_csv("orf_features_with_kmers.csv")

# Load a GFF database
db_path = "yeast_genes.db"
gff_file = "saccharomyces_cerevisiae.gff3"

if os.path.exists(db_path):
    db = gffutils.FeatureDB(db_path)
else:
    print("Creating a GFF database...")
    db = gffutils.create_db(
        gff_file,
        dbfn=db_path,
        force=True,         # Overwrites any existing DB
        keep_order=True,
        merge_strategy='merge',         # Fixes duplicated ID issues
        disable_infer_genes=True,
        disable_infer_transcripts=True
    )

# Function to check if an ORF overlaps a known gene
def is_known_orf(chrom, start, end, strand):
    try:
        genes = list(db.region(region=(chrom, start, end), featuretype='gene'))
        for gene in genes:
            gene_strand = 1 if gene.strand == "+" else -1
            if gene_strand == strand:
                return True
        return False
    except Exception as e:
        return False

# Label each ORF
labels = []
for idx, row in df.iterrows():
    chrom = row['chromosome']
    start = int(row['start'])
    end = int(row['end'])
    strand = int(row['strand'])
    label = "known_gene" if is_known_orf(chrom, start, end, strand) else "novel_orf"
    labels.append(label)

df['label'] = labels

# Save output
df.to_csv("orf_labeled.csv", index=False)
print("Saved labeled ORFs to orf_labeled.csv")


# Load ORFs
df = pd.read_csv("predicted_orfs.csv")

# Load gffutils database
db = gffutils.FeatureDB("yeast_genes.db")

def label_orf(row):
    chrom = row['chromosome'].replace("chr","")
    start = int(row['start']) + 1
    end = int(row['end'])
    strand = "+" if row['strand'] == 1 else "-"

    # Check for overlap with known CDS features
    try:
        overlapping = list(db.region(region=(chrom, start, end),featuretype='CDS'))
        for feature in overlapping:
            if feature.strand == strand:
                return "known_gene"
        return "novel_orf"
    except Exception as e:
        print(f"Error checking region {chrom}:{start}-{end}: {e}")
        return "error"

# Apply labeling
df["label"] = df.apply(label_orf, axis=1)
df.to_csv("labeled_orfs.csv", index=False)
print("Labeled ORFs saved to labeled_orfs.csv")
print(df["label"].value_counts())
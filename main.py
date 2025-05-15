import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from collections import Counter
from itertools import product
from collections import defaultdict
import orfs


all_orfs = []

# Load S.cerevisiae genome
records = []
for record in SeqIO.parse("data/sacCer3.fasta", "fasta"):
    print(f"Processing {record.id}...")
    sequence = record.seq
    orfs = orfs.find_orfs(sequence, min_len=300)

    for orf in orfs:
        orf['chromosome'] = record.id
    all_orfs.extend(orfs)

for record in SeqIO.parse("data/sacCer3.fasta", "fasta"):
    chrom_seq = record.seq
    for orf in all_orfs:
        if orf['chromosome'] == record.id:
            orf_seq = orfs.extract_orf_seq(record, orf)
            header = f"{orf['chromosome']}_start{orf['start']}_end{orf['end']}_strand{orf['strand']}"
            records.append(SeqRecord(orf_seq, id=header, description=""))
SeqIO.write(records, "predicted_orfs.fasta", "fasta")

# Save ORFs to CSV
df = pd.DataFrame(all_orfs)
df.to_csv("outputs/predicted_orfs.csv", index=False)

print(f"Found {len(df)} ORFs across all chromosomes.")

sns.histplot(df['length'], bins=100)
plt.title("ORF Length Distribution")
plt.xlabel("Length (bp)")
plt.ylabel("Frequency")
plt.show()

# Load the ORF table
df = pd.read_csv("outputs/predicted_orfs.csv")

# Store the genome in a dictionary for a fast lookup
genome = {rec.id:rec.seq for rec in SeqIO.parse("data/sacCer3.fasta", "fasta")}

feature_df = df.apply(lambda row: orfs.get_orf_features(row, genome), axis=1)
df = pd.concat([df, feature_df], axis=1)

df.to_csv("outputs/orf_features_basic.csv", index=False)
print("Features saved to outputs/orf_features_basic.csv")

#-----CODON USAGE-----
# Standard codon table
standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
all_codons = [a + b + c for a in "ATGC" for b in "ATGC" for c in "ATGC"]
stop_codons = set(standard_table.stop_codons)

codon_df = df['sequence'].apply(orfs.codon_features).apply(pd.Series)
df = pd.concat([df, codon_df], axis=1)

df.to_csv("outputs/orf_features_with_codon_usage.csv", index=False)
print("Codon usage features added and saved.")

print(df[[c for c in df.columns if c.startswith("codon_")]].mean().sort_values(ascending=False).head(10))

# 3-mer features
kmer_df = df['sequence'].apply(lambda s: orfs.get_kmer_counts(s, k=3)).apply(pd.Series)
df = pd.concat([df, kmer_df], axis=1)
df.to_csv("outputs/orf_features_with_kmers.csv", index=False)
print("Added 3-mer frequencies.")

df = pd.read_csv("outputs/orf_features_with_kmers.csv")


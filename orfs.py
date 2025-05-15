import pandas as pd
from collections import Counter
from constants import all_codons, stop_codons, get_all_kmers

def find_orfs(seq, min_len=300, max_orf_len=10000):
    start_codon = "ATG"
    stop_codons = {"TAA", "TAG", "TGA"}
    orfs = []

    # Scan each reading frame (fwd strand)
    for strand, nuc_seq in [(+1, seq), (-1, seq.reverse_complement())]:
        for frame in range(3):
            i = frame
            while i < len(nuc_seq) - 3:
                codon = str(nuc_seq[i:i+3])
                if codon == start_codon:
                    found_stop = False
                    for j in range(i+3, len(nuc_seq) - 3, 3):
                        stop = str(nuc_seq[j:j+3])
                        if stop in stop_codons:
                            orf_len = j + 3 - i
                            if orf_len >= min_len:
                                orfs.append({
                                    'start': i if strand == 1 else len(seq) - j - 3,
                                    'end': j + 3 if strand == 1 else len(seq) - i,
                                    'length': orf_len,
                                    'strand': strand,
                                    'frame': frame
                                })
                                found_stop = True
                                i = j + 3
                            break
                    if not found_stop:
                        i += 3
                else:
                    i += 3
    return orfs

def extract_orf_seq(record, orf):
    seq = record.seq
    if orf['strand'] == 1:
        return seq[orf['start']:orf['end']]
    else:
        return seq[orf['start']:orf['end']].reverse_complement()

def get_orf_features(row, genome):
    chrom = row['chromosome']
    start = int(row['start'])
    end = int(row['end'])
    strand = int(row['strand'])
    seq = genome[chrom][start:end]

    if strand == -1:
        seq = seq.reverse_complement()

    gc = 100 * float(seq.count("G") + seq.count("C")) / len(seq)

    return pd.Series({
        'gc_content': gc,
        'strand': strand,
        'frame': row['frame'],
        'sequence': str(seq)
    })

def codon_features(seq):
    seq = seq.upper()
    codons = [seq[i:i+3] for i in range(0, len(seq) - 2, 3)]
    codon_counts = Counter(codons)

    total_codons = sum(codon_counts.values())
    if total_codons == 0:
        return {f"codon_{c}": 0 for c in all_codons} | {"stop_percent": 0}
    features = {
        f"codon_{codon}": codon_counts.get(codon, 0) / total_codons
        for codon in all_codons
    }

    stop_count = sum(codon_counts.get(stop, 0) for stop in stop_codons)
    features["stop_percent"] = stop_count / total_codons

    return features

def get_kmer_counts(seq, k=3):
    seq = seq.upper()
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    total = len(kmers)
    counts = Counter(kmers)

    # Normalize and fill missing
    all_kmers = get_all_kmers(k)
    freqs = {f"{k}mer_{kmer}": counts.get(kmer, 0) / total if total > 0 else 0
             for kmer in all_kmers}
    return freqs
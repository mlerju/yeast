import pandas as pd
from collections import Counter
import logging
from constants import all_codons, stop_codons, get_all_kmers

# Setup logging
logger = logging.getLogger(__name__)

def find_orfs(seq, min_len=300, max_orf_len=10000):
    """
    Identifies open reading frames (ORFs) in the input DNA sequence.
    Searches all six frames (3 forward, 3 reverse) for ORFs that start with 'ATG'
    and end with a standard stop codon, longer than the specified minimum length.
    """
    start_codon = "ATG"
    stop_codons = {"TAA", "TAG", "TGA"}
    orfs = []

    logger.debug("Scanning for ORFs...")
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
    logger.debug(f"Found {len(orfs)} ORFs in sequence")
    return orfs

def extract_orf_seq(record, orf):
    """
    Extracts the nucleotide sequence corresponding to an ORF from a SeqRecord.
    Handles reverse complementing if ORF is on the reverse strand.
    """
    seq = record.seq
    extracted = seq[orf['start']:orf['end']]
    return extracted if orf['strand'] == 1 else extracted.reverse_complement()

def get_orf_features(row, genome):
    """
    Computes basic features for a given ORF row including GC content,
    strand, frame, and full nucleotide sequence.
    """
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
    """
    Computes codon usage frequencies and proportion of stop codons in a given sequence.
    Returns a dictionary of normalized codon counts and stop codon percentage.
    """
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
    """
    Computes k-mer frequency features from a sequence.
    Returns a dictionary with normalized frequencies of each possible k-mer.
    """
    seq = seq.upper()
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    total = len(kmers)
    counts = Counter(kmers)
    all_kmers = get_all_kmers(k)
    freqs = {f"{k}mer_{kmer}": counts.get(kmer, 0) / total if total > 0 else 0
             for kmer in all_kmers}
    return freqs
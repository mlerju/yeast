from collections import defaultdict
from itertools import product

from Bio.Data import CodonTable

# Codon data
standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
all_codons = [a + b + c for a in "ATGC" for b in "ATGC" for c in "ATGC"]
stop_codons = set(standard_table.stop_codons)
start_codons = set(standard_table.start_codons)
codon_to_aa = standard_table.forward_table

# Reverse map
aa_to_codons = defaultdict(list)
for codon, aa in codon_to_aa.items():
    aa_to_codons[aa].append(codon)

def get_all_kmers(k):
    return [''.join(p) for p in product('ATGC', repeat=k)]

def is_valid_codon(codon):
    return codon in all_codons and len(codon) == 3

def translate_codon(codon):
    return codon_to_aa.get(codon, '*')
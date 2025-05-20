from Bio.Data import CodonTable
from itertools import product

standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
all_codons = [a + b + c for a in "ATGC" for b in "ATGC" for c in "ATGC"]
stop_codons = set(standard_table.stop_codons)

def get_all_kmers(k):
    return [''.join(p) for p in product('ATGC', repeat=k)]
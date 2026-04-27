"""
MRjobs.py
-----------
The script computes the chi-square statistic for every (term, category)
pair appearing in the Amazon Review Dataset, keeps the top-75 most discriminative
terms per category, and finally produces output.txt.

Tokenisation, case-folding and stop-word filtering happen in ``utils.py``.
"""

from __future__ import annotations

import heapq
import json
from typing import Iterable, Iterator, Tuple

from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol
from mrjob.step import MRStep

from utils import load_stopwords, preprocess_text

# Tags for the step-1 output.
TAG_TERM_CAT = "T"   # ("T", term, category)  -> doc-frequency of term in category
TAG_CAT      = "C"   # ("C", category)        -> #docs in category
TAG_TOTAL    = "N"   # ("N",)                 -> total #docs

class MRChiSquare(MRJob):
    """
    mrjob.MRJob subclass.
    The input is a text file where every line is a JSON-encoded review.
    The output is the final output.txt produced by the last reducer.
    """
    OUTPUT_PROTOCOL = RawValueProtocol
    FILES = ["utils.py", "../data/stopwords.txt"]

    # Steps                                          
    def steps(self) -> list[MRStep]:
        """Return the two-step pipeline described in the module docstring."""
        return [
            MRStep(
                mapper_init=self.mapper_init,
                mapper=self.mapper_count,
                combiner=self.combiner_count,
                reducer=self.reducer_count,
            ),
            MRStep(
                mapper=self.mapper_chi,
                reducer=self.reducer_chi,
            ),
        ]

    # Per-task setup: load stopwords once into a Python set.             
    def mapper_init(self) -> None:
        """Load the stopword list."""
        self._stop = load_stopwords("../data/stopwords.txt")

    # ================================================================== 

    # STEP 1 -- Mapper                                                   
    def mapper_count(self, _: None, line: str) -> Iterator[Tuple[Tuple[str, ...], int]]:
        """
        Parse one review and emit:

            (("T", term, category), 1)  -- once per unique term in the doc
            (("C", category),       1)  -- once per document
            (("N",),                1)  -- once per document  (=> total N)

        Parameters
        ----------
        _ : None
            ``None`` because we read line-by-line.
        line : str
            One JSON-encoded review.
        """
        # JSON parsing - silently skip malformed lines
        try:
            rec = json.loads(line)
            text = rec["reviewText"]
            cat = rec["category"]
        except (ValueError, KeyError, TypeError):
            return

        unique = set(preprocess_text(text, self._stop))

        yield (TAG_TOTAL,), 1
        yield (TAG_CAT, cat), 1
        for term in unique:
            yield (TAG_TERM_CAT, term, cat), 1


    # STEP 1 -- Combiner                                                 
    def combiner_count(self, key: Tuple[str, ...], values: Iterable[int]) -> Iterator[Tuple[Tuple[str, ...], int]]:
        """
        Local, in-mapper aggregation.  Halves the shuffle
        volume because the same (term, category) pair appears many times
        in a single input split.
        """
        yield key, sum(values)

    # STEP 1 -- Reducer                                                  
    def reducer_count(self, key: Tuple[str, ...], values: Iterable[int]) -> Iterator[Tuple[Tuple[str, ...], int]]:
        """Final aggregation -- identical to the combiner but global."""
        yield key, sum(values)

    # ================================================================== 

    # STEP 2 -- Mapper                                                   
    def mapper_chi(self, key: Tuple[str, ...], value: int) -> Iterator[Tuple[str, Tuple]]:
        """Re-emit step-1 output under a constant key for the final reducer."""
        yield "_", (key, value)

    # STEP 2 -- Reducer                                                  
    def reducer_chi(self, _: str, records: Iterable[Tuple]) -> Iterator[Tuple[None, str]]:
        """
        Compute chi^2, top 75 per category, and the merged dictionary.

        The reducer streams through the (key, count) pairs once and
        builds three in-memory structures:

            n_total                                    -- total #docs
            cat_total[category]                        -- #docs in cat
            term_cat[(term, cat)]                      -- #docs in cat containing term
        """
        n_total = 0
        cat_total: dict[str, int] = {}
        term_cat: dict[Tuple[str, str], int] = {}

        # ---- ingest all step-1 outputs --------------------------------
        for key, count in records:
            tag = key[0]
            if tag == TAG_TERM_CAT:
                term_cat[(key[1], key[2])] = count
            elif tag == TAG_CAT:
                cat_total[key[1]] = count
            elif tag == TAG_TOTAL:
                n_total += count

        if n_total == 0:
            return  # empty input, nothing to do

        # ---- per-term doc-frequency (sum over categories) -------------
        term_total: dict[str, int] = {}
        for (term, cat), c in term_cat.items():
            term_total[term] = term_total.get(term, 0) + c

        # ---- chi-square + per-category 75 heap ---------------------
        per_cat_heap: dict[str, list] = {c: [] for c in cat_total}

        for (term, cat), A in term_cat.items():
            cat_n = cat_total[cat]                 # A + C  (#docs in cat)
            term_n = term_total[term]              # A + B  (#docs with term)
            B = term_n - A                         # term, NOT in cat
            C = cat_n - A                          # cat,  NOT term
            D = n_total - A - B - C                # neither

            denom = (A + B) * (C + D) * (A + C) * (B + D)
            num = n_total * (A * D - B * C) ** 2
            chi2 = num / denom

            heap = per_cat_heap[cat]
            if len(heap) < 75:
                heapq.heappush(heap, (chi2, term))
            elif chi2 > heap[0][0]:
                heapq.heapreplace(heap, (chi2, term))

        # ---- emit per-category lines, alphabetic by category ---------
        merged_vocab: set[str] = set()
        for cat in sorted(per_cat_heap):
            # Sort the heap descending by chi^2 (ties broken by term).
            top = sorted(per_cat_heap[cat], key=lambda x: (-x[0], x[1]))
            if not top:
                continue
            parts = [cat]
            for chi2, term in top:
                merged_vocab.add(term)
                parts.append(f"{term}:{chi2:.4f}")
            yield None, " ".join(parts)

        # ---- emit merged dictionary line ------------------------------
        yield None, " ".join(sorted(merged_vocab))


if __name__ == "__main__":
    MRChiSquare.run()
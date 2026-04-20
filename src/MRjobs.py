from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONProtocol
import json
import sys
import os

# Important: Tell Python where to find utils.py
sys.path.append(os.path.dirname(__file__))
import utils 

class ChiSquarePipeline(MRJob):

    INTERNAL_PROTOCOL = JSONProtocol

    def steps(self):
        return [
            MRStep(mapper_init=self.mapper_init_count,
                   mapper=self.mapper_count,
                   combiner=self.reducer_count,
                   reducer=self.reducer_count),
            MRStep(reducer_init=self.reducer_init_calc,
                   reducer=self.reducer_calc,
                   reducer_final=self.reducer_final_calc)
        ]

    def mapper_init_count(self):
        # Load stopwords using the utility function
        self.stop_words = utils.load_stopwords("data/stopwords.txt")

    def mapper_count(self, _, line):
        try:
            data = json.loads(line)
            cat = data.get("category", "Unknown")
            review_text = data.get("reviewText", "")

            # Use the utility function for preprocessing
            tokens = utils.preprocess_text(review_text, self.stop_words)
            unique_tokens = set(tokens)

            yield ("__TOTAL__", "__N__"), 1
            yield ("__CAT__", cat), 1
            for t in unique_tokens:
                yield (t, cat), 1
                yield (t, "__ALL__"), 1
        except Exception:
            pass

    def reducer_count(self, key, values):
        yield key, sum(values)

    def reducer_init_calc(self):
        self.N = 0
        self.cat_counts = {}
        self.term_totals = {}
        self.term_cat_pairs = []

    def reducer_calc(self, key, values):
        val = sum(values)
        if key[0] == "__TOTAL__": 
            self.N = val
        elif key[0] == "__CAT__": 
            self.cat_counts[key[1]] = val
        elif key[1] == "__ALL__": 
            self.term_totals[key[0]] = val
        else: 
            self.term_cat_pairs.append((key, val))

    def reducer_final_calc(self):
        for (term, cat), A in self.term_cat_pairs:
            A_plus_B = self.term_totals.get(term, 0)
            A_plus_C = self.cat_counts.get(cat, 0)
            
            B, C = A_plus_B - A, A_plus_C - A
            D = self.N - (A + B + C)
            
            den = (A + B) * (A + C) * (B + D) * (C + D)
            chi2 = (self.N * ((A * D - B * C) ** 2) / den) if den != 0 else 0.0
            
            yield (cat, term), chi2

if __name__ == "__main__":
    ChiSquarePipeline.run()
#!/usr/bin/env python

# bleu_scorer.py
# David Chiang <chiang@isi.edu>

# Copyright (c) 2004-2006 University of Maryland. All rights
# reserved. Do not redistribute without permission from the
# author. Not for commercial use.

# Modified by: 
# Hao Fang <hfang@uw.edu>
# Tsung-Yi Lin <tl483@cornell.edu>
#
# Modified to work with Python 3 (in 2018) by:
# Frosina Stojanovska

"""Provides:
cook_refs(refs, n=4): Transform a list of reference sentences as strings into a form usable by cook_test().
cook_test(test, refs, n=4): Transform a test sentence as a string (together with the cooked reference sentences)
                            into a form usable by score_cooked().
"""

import copy
import math
from collections import defaultdict


def precook(s, n=4):
    """ Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n+1):
        for i in range(len(words)-k+1):
            n_gram = tuple(words[i:i+k])
            counts[n_gram] += 1
    return len(words), counts


def cook_refs(refs, eff=None, n=4):  # lhuang: oracle will call with "average"
    """ Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    """
    ref_len = []
    max_counts = {}
    for ref in refs:
        rl, counts = precook(ref, n)
        ref_len.append(rl)
        for (n_gram, count) in counts.items():
            max_counts[n_gram] = max(max_counts.get(n_gram, 0), count)

    # Calculate effective reference sentence length.
    if eff == "shortest":
        ref_len = min(ref_len)
    elif eff == "average":
        ref_len = float(sum(ref_len))/len(ref_len)

    # lhuang: N.B.: leave ref_len computaiton to the very end!!
    
    # lhuang: N.B.: in case of "closest", keep a list of ref_lens!! (bad design)

    return ref_len, max_counts


def cook_test(test, ref_len, ref_max_counts, eff=None, n=4):
    """Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    """

    test_len, counts = precook(test, n)

    result = {}

    # Calculate effective reference sentence length.
    
    if eff == "closest":
        result["reflen"] = min((abs(l-test_len), l) for l in ref_len)[1]
    else:
        # i.e., "average" or "shortest" or None
        result["reflen"] = ref_len

    result["testlen"] = test_len

    result["guess"] = [max(0, test_len-k+1) for k in range(1, n+1)]

    result['correct'] = [0]*n
    for (n_gram, count) in counts.items():
        result["correct"][len(n_gram)-1] += min(ref_max_counts.get(n_gram, 0), count)

    return result


class BleuScorer(object):
    """
    Bleu scorer.
    """

    __slots__ = "n", "crefs", "ctest", "_score", "_ratio", "_testlen", "_reflen", "special_reflen"
    # special_reflen is used in oracle (proportional effective ref len for a node).

    def copy(self):
        """
        copy the refs.
        """
        new = BleuScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        new._score = None
        return new

    def __init__(self, test=None, refs=None, n=4, special_reflen=None):
        """ singular instace """

        self.n = n
        self.crefs = []
        self.ctest = []
        self.cook_append(test, refs)
        self.special_reflen = special_reflen
        self._score = None
        self._testlen = 0
        self._reflen = 0

    def cook_append(self, test, refs):
        """called by constructor and __iadd__ to avoid creating new instances."""
        
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                cooked_test = cook_test(test, self.crefs[-1][0], self.crefs[-1][1])
                # N.B.: -1
                self.ctest.append(cooked_test)
            else:
                # lens of crefs and ctest have to match
                self.ctest.append(None)

        # need to recompute
        self._score = None

    def ratio(self, option=None):
        self.compute_score(option=option)
        return self._ratio

    def score_ratio(self, option=None):
        """ return (bleu, len_ratio) pair """
        return self.fscore(option=option), self.ratio(option=option)

    def score_ratio_str(self, option=None):
        return "%.4f (%.2f)" % self.score_ratio(option)

    def reflen(self, option=None):
        self.compute_score(option=option)
        return self._reflen

    def testlen(self, option=None):
        self.compute_score(option=option)
        return self._testlen        

    def retest(self, new_test):
        if type(new_test) is str:
            new_test = [new_test]
        assert len(new_test) == len(self.crefs), new_test
        self.ctest = []
        for t, rs in zip(new_test, self.crefs):
            self.ctest.append(cook_test(t, rs[0], rs[1]))
        self._score = None

        return self

    def rescore(self, new_test):
        """ replace test(s) with new test(s), and returns the new score. """
        
        return self.retest(new_test).compute_score()

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        """ add an instance (e.g., from another sentence). """

        if type(other) is tuple:
            # avoid creating new BleuScorer instances
            self.cook_append(other[0], other[1])
        else:
            assert self.compatible(other), "incompatible BLEUs."
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
            # need to recompute
            self._score = None

        return self        

    def compatible(self, other):
        return isinstance(other, BleuScorer) and self.n == other.n

    def single_reflen(self, option="average"):
        return self._single_reflen(self.crefs[0][0], option)

    @staticmethod
    def _single_reflen(ref_lens, option=None, test_len=None):
        
        if option == "shortest":
            ref_len = min(ref_lens)
        elif option == "average":
            ref_len = float(sum(ref_lens)) / len(ref_lens)
        elif option == "closest":
            ref_len = min((abs(l - test_len), l) for l in ref_lens)[1]
        else:
            assert False, "unsupported reflen option %s" % option

        return ref_len

    def recompute_score(self, option=None, verbose=0):
        self._score = None
        return self.compute_score(option, verbose)
        
    def compute_score(self, option=None, verbose=0):
        n = self.n
        small = 1e-9
        # so that if guess is 0 still return 0
        tiny = 1e-15
        bleu_list = [[] for _ in range(n)]

        if self._score is not None:
            return self._score

        if option is None:
            option = "average" if len(self.crefs) == 1 else "closest"

        self._testlen = 0
        self._reflen = 0
        total_comps = {'testlen': 0, 'reflen': 0, 'guess': [0]*n, 'correct': [0]*n}

        # for each sentence
        for comps in self.ctest:            
            test_len = comps['testlen']
            self._testlen += test_len

            if self.special_reflen is None:
                # need computation
                ref_len = self._single_reflen(comps['reflen'], option, test_len)
            else:
                ref_len = self.special_reflen

            self._reflen += ref_len
                
            for key in ['guess', 'correct']:
                for k in range(n):
                    total_comps[key][k] += comps[key][k]

            # append per image bleu score
            bleu = 1.
            for k in range(n):
                bleu *= (float(comps['correct'][k]) + tiny) \
                        / (float(comps['guess'][k]) + small)
                bleu_list[k].append(bleu ** (1./(k+1)))
            # N.B.: avoid zero division
            ratio = (test_len + tiny) / (ref_len + small)
            if ratio < 1:
                for k in range(n):
                    bleu_list[k][-1] *= math.exp(1 - 1/ratio)

            if verbose > 1:
                print(comps, ref_len)

        total_comps['reflen'] = self._reflen
        total_comps['testlen'] = self._testlen

        bleus = []
        bleu = 1.
        for k in range(n):
            bleu *= float(total_comps['correct'][k] + tiny) \
                    / (total_comps['guess'][k] + small)
            bleus.append(bleu ** (1./(k+1)))
        # N.B.: avoid zero division
        ratio = (self._testlen + tiny) / (self._reflen + small)
        if ratio < 1:
            for k in range(n):
                bleus[k] *= math.exp(1 - 1/ratio)

        if verbose > 0:
            print(total_comps)
            print("ratio:", ratio)

        self._score = bleus
        return self._score, bleu_list

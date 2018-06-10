#!/usr/bin/env python
# 
# File Name : ptbtokenizer.py
#
# Description : Do the PTB Tokenization and remove punctuations.
#
# Creation Date : 29-12-2014
# Last Modified : Thu Mar 19 09:53:35 2015
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>
#
# Modified to work with Python 3 (in 2018) by:
# Frosina Stojanovska

import os
import subprocess
import tempfile

# path to the stanford corenlp jar
STANFORD_CORENLP_3_4_1_JAR = 'stanford-corenlp-3.9.1.jar'

# punctuations to be removed from the sentences
PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                ".", "?", "!", ",", ":", "-", "--", "...", ";"]


class PTBTokenizer:
    """Python wrapper of Stanford PTBTokenizer"""

    @staticmethod
    def tokenize(captions_for_image):
        cmd = ['java', '-cp', STANFORD_CORENLP_3_4_1_JAR,
               'edu.stanford.nlp.process.PTBTokenizer',
               '-preserveLines', '-lowerCase']

        # ======================================================
        # prepare data for PTB Tokenizer
        # ======================================================
        final_tokenized_captions_for_image = {}
        image_id = [k for k, v in captions_for_image.items() for _ in range(len(v))]
        sentences = '\n'.join([c.replace('\n', ' ') for k, v in captions_for_image.items() for c in v])

        # ======================================================
        # save sentences to temporary file
        # ======================================================
        path_to_jar_dir_name = os.path.dirname(os.path.abspath(__file__))
        tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dir_name)
        tmp_file.write(sentences.encode())
        tmp_file.close()

        # ======================================================
        # tokenize sentence
        # ======================================================
        cmd.append(os.path.basename(tmp_file.name))
        p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dir_name,
                                       stdout=subprocess.PIPE)
        token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
        lines = token_lines.decode().split('\n')
        # remove temp file
        os.remove(tmp_file.name)

        # ======================================================
        # create dictionary for tokenized captions
        # ======================================================
        for k, line in zip(image_id, lines):
            if k not in final_tokenized_captions_for_image:
                final_tokenized_captions_for_image[k] = []
            tokenized_caption = ' '.join([w for w in line.rstrip().split(' ')
                                          if w not in PUNCTUATIONS])
            final_tokenized_captions_for_image[k].append(tokenized_caption)

        return final_tokenized_captions_for_image

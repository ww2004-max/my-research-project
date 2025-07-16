#!/usr/bin/env python
"""obtain_ted_ml_corpus.py: Get a Multilingual Parallel Corpus with separate files per language."""
__author__ = "Raj Dabre"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "prajdabre@gmail.com"
__status__ = "Development"

import sys, os
from collections import defaultdict


#langs = set(["zh-cn", "zh-tw", "ja", "ko"])
#langs = set(["de", "fr","it", "nl", "ar", "he", "ru", "es", "pt-br"])

## This script separates a N lingual corpus into individual files per language. 
## The format of each line in the input file is: 
## <line_id>:<language_code>:sentence
## This script was developed for processing the corpus available at https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus/tree/master/Multilingual_Parallel_Corpus

def extract_corpus(input_file):
	a=open(input_file)
	langs = set(["de", "fr","it", "nl", "ar", "he", "ru", "es", "pt-br", "zh-cn", "zh-tw", "ja", "ko"]) # Change this according the the language
	corpus = defaultdict(lambda: defaultdict())
	skipped_count = 0
	for line in a:
		components = line.strip().split(":")
		if len(components) < 3:
			skipped_count += 1
			continue
		lang=components[1]
		if lang not in langs:
			skipped_count += 1
			continue
		lineid=components[0]
		content=":".join(components[2:])
		corpus[lineid][lang] = content

	print "Languages are: ", langs
	print "Skipped ", skipped_count, " lines which are incorrectly formatted."
	lang_files = {}
	for lang in langs:
		lang_files[lang] = open(sys.argv[1]+"."+lang, 'w')

	skipped_count = 0

	for lineid in corpus:
		skip = False
		for lang in lang_files:
			if not corpus[lineid].has_key(lang):
				skip = True
				break
		if skip:
			skipped_count += 1
			continue
		for lang in lang_files:
			if corpus[lineid].has_key(lang):
				lang_files[lang].write(corpus[lineid][lang]+"\n")
			else:
				lang_files[lang].write("Dummy line to ensure equal line count.\n")

	print "Skipped ", skipped_count, " lines which are not available in all languages."
	for lang in langs:
		lang_files[lang].flush()
		lang_files[lang].close()

	a.close()


def commandline():
    
    import argparse
    parser = argparse.ArgumentParser(description= "Get N lingual corpus", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_file", help = "The input file")

    args = parser.parse_args()

    extract_corpus(args.input_file)

if __name__ == '__main__':
    commandline() 

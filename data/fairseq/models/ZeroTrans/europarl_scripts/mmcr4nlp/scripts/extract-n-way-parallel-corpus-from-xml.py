# -*- coding: utf-8 -*-

#!/usr/bin/env python
"""extract-n-way-parallel-corpus-from-xml.py: Get a Multilingual Parallel Bible corpus from the XML files."""
__author__ = "Raj Dabre"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "prajdabre@gmail.com"
__status__ = "Development"

import sys, io
import xml.etree.ElementTree as ET

def extract_corpus(extension, files):
	L_files = []

	out_files = []
	print "Number of languages: ", len(files)
	for i in range(len(files)):
		L_files.append(ET.parse(files[i]))

	L_files_roots = []

	for i in L_files:
		L_files_roots.append(i.getroot())

	for i in range(len(files)):
		out_files.append(io.open(extension+".corpus."+files[i].split(".")[0],"w",encoding='utf8'))

	sent_id_to_sent_set = {}

	for pos in range(0,len(L_files_roots)):
		file_root = L_files_roots[pos]
		for seg in file_root.iter('seg'):
			if not sent_id_to_sent_set.has_key(seg.attrib['id'].strip()):
				sent_id_to_sent_set[seg.attrib['id'].strip()] = ['']* len(L_files_roots)
			if seg.text is not None:
				sent_id_to_sent_set[seg.attrib['id'].strip()][pos] = seg.text.strip()
				
	for i in sent_id_to_sent_set.keys():
		if not '' in sent_id_to_sent_set[i]:
			for pos in range(0,len(L_files_roots)):
				out_files[pos].write(sent_id_to_sent_set[i][pos]+unicode("\n"))
				out_files[pos].flush()

	print "Total lines extracted: ", len(sent_id_to_sent_set)
	  

	for i in out_files:
		i.close()


def commandline():
    
    import argparse
    parser = argparse.ArgumentParser(description= "Get N lingual corpus from the XML files for the Bible corpus", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("extension", help = "The extension that will be prepended to the final file containing the corpus.")
    parser.add_argument("input_files", nargs='+', help = "The input XML files.")

    args = parser.parse_args()

    extract_corpus(args.extension, args.input_files)

if __name__ == '__main__':
    commandline() 
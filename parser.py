# -*- coding: utf-8 -*-
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from cStringIO import StringIO
import nltk.data
import re
import pyPdf


def pdf_to_text(pdfname, all):
    with open(pdfname, 'rb') as fp:
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        device = TextConverter(rsrcmgr, retstr, codec='utf-8', laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        page = set()
        if not all:
            pdf = pyPdf.PdfFileReader(fp)
            num_of_pages = pdf.getNumPages()
            page = set(range(num_of_pages // 2))
        for page in PDFPage.get_pages(fp, page):
            interpreter.process_page(page)

        text = retstr.getvalue()
        device.close()
        retstr.close()
    return text


def text_to_sentences(text):
    # nltk.download()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(text.decode('utf-8').strip())

    # 2. Loop over each sentence
    sentences = []
    for sentence in raw_sentences:
        # If a sentence is empty, skip it
        # if (re.match(ur'.*\b\s*Р\s*Е\s*Ш\s*И\s*Л\s*.*?', sentence, flags=re.UNICODE) or
        #         re.match(ur'.*?отказать.*?', sentence, flags=re.UNICODE)):
        #     break
        if len(sentence) > 0:
            # Otherwise, append a list of words to sentences
            sentences.append(sentence_to_wordlist(sentence))

    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def sentence_to_wordlist(sentence):
    sentence = re.sub(ur'[\W\d]+', u' ', sentence, flags=re.UNICODE)
    words = [word for word in sentence.lower().split() if len(word) > 2]
    return words

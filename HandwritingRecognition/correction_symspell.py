from symspellpy import SymSpell
symsp = SymSpell()
symsp.load_dictionary('corpus.txt',\
                      term_index=0, \
                      count_index=1, \
                      separator=' ')
txt='\" with gadly fean prepaned an ant fon .'
terms = symsp.lookup_compound(txt,2)
for k in terms:
    print(k.term)
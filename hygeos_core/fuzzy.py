import difflib


def _split_string(s: str):
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = s.strip()
    return s.split(" ")

def _fuzzy_score(search_terms, string: str, word_threshold = 0.71):
    """
    return the number of search term found in reference string within word_treshold variability
    """
    
    iscore = 0
    score  = 0
    
    # word_match = 0
    splitted_string = _split_string(string)
    nrefterms = len(splitted_string)
    for term in search_terms: #for each term
        for word in splitted_string: # for each word in the line
            res = difflib.SequenceMatcher(None, term, word).ratio() # check match
            if res >= word_threshold: 
                score += res # weight by word match -> allow to discriminate perfect matches with low accuracy matches
                iscore += 1
                # word_match += res
                break # search term has been found.

    # penalize string that have terms which have not been matched
    unmatched_term_penalty =  (1-word_threshold) * (1 - (iscore / nrefterms))
    
    return score - unmatched_term_penalty
    

def search(keywords: list[str]|str, list_of_string, threshold = 0.71, nmax = None) :
    
    assert nmax is None or nmax > 0
    if type(keywords) == str:
        keywords = [keywords]
    keywords = [k.lower().strip() for k in keywords]
    
    results = []
    
    minimum_score_ratio = threshold
    nterms = len(keywords)
    
    for string in list_of_string: #for each name
        score = _fuzzy_score(keywords, string.lower().strip())
        ratio = score / nterms
        if ratio >= minimum_score_ratio:
            results.append((string, ratio))
                    
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    if nmax is not None and nmax >= (len(sorted_results)):
        nmax = None
    
    return sorted_results[:nmax]



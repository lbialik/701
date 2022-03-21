import json

def load_sentences(data):
    '''Takes a JSON object as input and returns a dictionary e.g. {'1': 'ORC': 'sentence'}'''
    sentences = {}
    for sentence_number in data:
        sentences[sentence_number] = {}
        s = data[sentence_number]

        orc = s["subject"] + s["clausal noun"] + s["clausal verb"] + s["verb phrase"] + "."
        orc_adv = s["subject"] + s["clausal noun"] + s["clausal verb"] + s["clausal adverb"] + s["verb phrase"] + "."
        orc_clausal_verb = s["subject"] + s["clausal noun"] + s["clausal phrasal verb"] + s["verb phrase"] + "."
        orc_clausal_verb_adv = s["subject"] + s["clausal noun"] + s["clausal phrasal verb"] + s["clausal adverb"] + s["verb phrase"] + "."
        src = s["subject"] + s["clausal verb"] + s["clausal noun"] + s["verb phrase"] + "."
        src_adv = s["subject"] + s["clausal verb"] + s["clausal noun"] + s["clausal adverb"] + s["verb phrase"] + "."

        sentences[sentence_number]['ORC'] = orc
        sentences[sentence_number]['ORC adv'] = orc_adv
        sentences[sentence_number]['ORC clausal verb'] = orc_clausal_verb
        sentences[sentence_number]['ORC clausal verb adverb'] = orc_clausal_verb_adv
        sentences[sentence_number]['SRC'] = src
        sentences[sentence_number]['SRC adverb'] = src_adv

    return sentences

import spacy
import random
import copy
import tensorflow.compat.v1 as tf

class ConceptGenerator:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def check_availability(self, sentence):
        def check_availability_sentence(x):
            x = x.numpy().decode('utf-8')
            doc = self.nlp(str(x))
            V_concepts = []
            N_concepts = []
            original_tokens = []
            for token in doc:
                original_tokens.append(token.text_with_ws)
                if token.pos_.startswith('V') and token.is_alpha and not token.is_stop:
                    V_concepts.append(token.text_with_ws)
            for noun_chunk in doc.noun_chunks:
                root_noun = noun_chunk[-1]
                if root_noun.pos_ == "NOUN":
                    N_concepts.append(root_noun.text_with_ws)
            if len(N_concepts) >= 2 or len(V_concepts) >= 2:
                if len(set(N_concepts)) == 1 or len(set(V_concepts)) == 1:
                    return False
                else:
                    return True
            else:
                return False

        if type(sentence) == dict:
            result = tf.py_function(check_availability_sentence, [sentence['text']], [tf.bool])[0]
        else:
            result = tf.py_function(check_availability_sentence, [sentence], [tf.bool])[0]

        return result

    def cor_generate(self, prompt):
        doc = self.nlp(str(prompt))
        V_concepts = []
        N_concepts = []
        original_tokens = []
        for token in doc:
            original_tokens.append(token.text_with_ws)
            if token.pos_.startswith('V') and token.is_alpha and not token.is_stop:
                V_concepts.append(token.text_with_ws)
        for noun_chunk in doc.noun_chunks:
            root_noun = noun_chunk[-1]
            if root_noun.pos_ == "NOUN":
                N_concepts.append(root_noun.text_with_ws)

        if len(N_concepts) >= 2:
            previous = copy.deepcopy(N_concepts)
            while previous == N_concepts:
                random.shuffle(N_concepts)
        if len(V_concepts) >= 2:
            previous = copy.deepcopy(V_concepts)
            while previous == V_concepts:
                random.shuffle(V_concepts)

        shuffled_tokens = []
        N_concepts_index = 0
        V_concepts_index = 0
        for tok in original_tokens:
            if tok in V_concepts and V_concepts_index < len(V_concepts):
                shuffled_tokens.append(V_concepts[V_concepts_index].strip())
                V_concepts_index += 1
            elif tok in N_concepts and N_concepts_index < len(N_concepts):
                shuffled_tokens.append(N_concepts[N_concepts_index].strip())
                N_concepts_index += 1
            else:
                shuffled_tokens.append(tok.strip())

        assert len(shuffled_tokens) == len(original_tokens)

        result = ' '.join(shuffled_tokens)
        return result

    def c2s_generate(self, prompt):
        doc = self.nlp(str(prompt))

        matched_concepts = []
        for token in doc:
            if (token.pos_.startswith('V') or token.pos_.startswith('PROP')) and token.is_alpha and not token.is_stop:
                matched_concepts.append(token.lemma_)
        for noun_chunk in doc.noun_chunks:
            root_noun = noun_chunk[-1]
            if root_noun.pos_ == "NOUN":
                matched_concepts.append(root_noun.lemma_)

        result = " ".join([token for token in matched_concepts])
        return result

    def generate(self, prompt):

        negative_sampling = random.uniform(0,1) < 0.5
        if negative_sampling:
            return self.cor_generate(prompt)
        else:
            return self.c2s_generate(prompt)



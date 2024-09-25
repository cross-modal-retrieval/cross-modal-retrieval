"""This module defines the TextEmbedding interface for converting audio captions into
embeddings.
"""
import zipfile
import functools
from abc import abstractmethod
from pathlib import Path
from typing import Set, List, Tuple, Union, Callable
from collections import defaultdict

import numpy as np
import torch
import gensim
import requests
import spacy
import transformers
import pkg_resources
from typeguard import typechecked
from symspellpy import SymSpell, Verbosity
from zsvision.zs_utils import BlockTimer

from CLIP.clip import clip 

class TextEmbedding:

    def __init__(self, 
                 model, 
                 dim: int, 
                 tokenizer: Union[Callable, None],
                 remove_stopwords:bool):
        self.model = model
        self.dim = dim
        self.device = None
        self.tokenizer = tokenizer
        self.remove_stopwords = remove_stopwords

    @abstractmethod
    def text2vec(self, text: str) -> np.ndarray:
        """Convert a string of text into an embedding.

        Args:
            text: the content to be embedded

        Returns:
            (d x n) array, where d is the dimensionality of the embedding and `n` is the
                number of words that were successfully parsed from the text string.

        NOTE: For some text embedding models (such as word2vec), not all words are
        converted to vectors (e.g. certain kinds of stop words) - these are dropped from
        the output.
        """
        raise NotImplementedError

    @typechecked
    def set_device(self, device: torch.device):
        self.model = self.model.to(device)
        self.device = device


@functools.lru_cache(maxsize=64, typed=False)
def load_w2v_model_from_cache(
        w2v_weights: Path,
) -> gensim.models.keyedvectors.Word2VecKeyedVectors:
    with BlockTimer("Loading w2v from disk"):
        model = gensim.models.KeyedVectors.load_word2vec_format(
            fname=w2v_weights,
            binary=True,
        )
    return model


@typechecked
def fetch_model(url: str, weights_path: Path):
    weights_path.parent.mkdir(exist_ok=True, parents=True)
    with BlockTimer(f"Fetching weights {url} -> {weights_path}"):
        resp = requests.get(url, verify=False)
        with open(weights_path, "wb") as f:
            f.write(resp.content)

class W2V_Lookup:
    def __init__(self, w2v):
        self.w2v = w2v
        self.vocab = set(w2v.vocab.keys())

    def __call__(self,key):
        return self.w2v.get_vector(key)


class W2VEmbedding(TextEmbedding):
    """This model embeds text using the google-released implementation of the word2vec
    model introduced in:

        Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013).
        Distributed representations of words and phrases and their compositionality.
        In Advances in neural information processing systems (pp. 3111-3119).

    For words that are present in the w2v vocabulary, a 300-dimensional embedding is
    produced via a lookup table.
    """
    @typechecked
    def __init__(
            self,
            dim: int,
            weights_path: Path,
            remove_stopwords: bool,
            num_samples_for_unknown: int = 50000,
    ):

        w2v = load_w2v_model_from_cache(weights_path)
        model = W2V_Lookup(w2v=w2v)
        tokenizer = Tokenizer(vocab=model.vocab)
        with BlockTimer("generating unknown vector"):
            vecs = np.zeros((min(num_samples_for_unknown, len(model.vocab)),dim))
            for ii, key in enumerate(sorted(model.vocab)):
                if ii >= num_samples_for_unknown:
                    break
                vecs[ii] = model(key)
        self.unknown_vector = np.mean(vecs, 0)

        super().__init__(model=model, 
                         dim=dim, 
                         tokenizer=tokenizer,
                         remove_stopwords=remove_stopwords,)

    @typechecked
    def text2vec(self, text: str) -> Tuple[np.ndarray, List[str]]:
        # convert the text string to tokens that can be processed by w2v.
        if self.remove_stopwords:
            processed_string = gensim.parsing.preprocessing.remove_stopwords(text)
        else:
            processed_string = text

        tokens, failed = self.tokenizer(processed_string)
        embeddings = []
        for token in tokens:
            embeddings.append(self.model(token))
        embeddings = np.array(embeddings)
        msg = (f"Failed to embed any tokens! (text:{text}, processed_string: "
               f"{processed_string}, failed: {failed})")
        # For empty sequences, we use zeros with the dimensionality of the features on
        # the second dimension (this is the format expected by the CE codebase)
        if embeddings.size == 0:
            print(f"Warning: {msg}, falling back to unknown vector")
            embeddings = np.array([self.unknown_vector])
        return embeddings, failed
    

    @typechecked
    def set_device(self, device: torch.device):
        msg = f"w2v only supports CPU-based execution found {device.type}"
        assert device.type == "cpu", msg

class Tokenizer:
    """For word-level embeddings, we convert words that are absent from the embedding lookup table to a canonical tokens ( and then re-check the table). This is to ensure that we get reasonable embeddings for as many words as possible.
    """

    @typechecked
    def __init__(self, vocab: Set[str]):
        with BlockTimer("preparing tokenizer dictionaries"):
            # we only use spacy for lemmatising, so we don't need NER or the parser.
            # NOTE: all pronouns are mapped to -PRON-, because it's not clear with
            # their lemma should be (we try to handle these via the spellchecker)
            self.nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])

            # Symspell is, in theory, a fast spell checker:
            sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            dictionary_path = pkg_resources.resource_filename(
                    "symspellpy","frequency_dictionary_en_82_765.txt")
            # term_index is the column of the term and count_index us the
            # column of the term frequency
            sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
            self.sym_spell = sym_spell
            self.vocab = vocab

            # For a small number of cases, the tokenizer fauks
            self.custom = {
                    "roundtable":["round","table"],
            }

    def __call__(self, text:str) -> List[str]:
        doc = self.nlp(text)
        tokens, failed = [], []
        for token in doc:
            token, lemma = str(token), token.lemma_
            if token is self.vocab:
                tokens.append(token)
            elif lemma in self.vocab:
                tokens.append(lemma)
            elif lemma in self.custom:
                for subtoken in self.custom[lemma]:
                    if subtoken in self.vocab:
                        tokens.append(subtoken)
                    else:
                        failed.append(subtoken)
            else:
                suggestions = self.sym_spell.lookup(
                        phrase = token,
                        verbosity = Verbosity.CLOSEST,
                        max_edit_distance=2,
                )
                success = False
                for suggestion in suggestions:
                    if suggestion.term in self.vocab:
                        success = True
                        tokens.append(suggestion.term)
                        break
                if not success:
                    failed.append(str(token))
        return tokens, failed

class HuggingFaceWrapper(TextEmbedding):
    """This class wraps the mebdedding of text provided by HuggingFace pretrained models
    : The models can be found here:
    https://huggingface.co/transformers/pretrained_models.html
    """

    def __init__(self, dim:int, embedding_name: str):
        tokenizers = {
            "openai-gpt": transformers.OpenAIGPTTokenizer,
            "bert-base-uncased": transformers.BertTokenizer,
            "ctrl": transformers.CTRLTokenizer,
            "transfo-xl-wt103": transformers.TransfoXLTokenizer,
            "electra": transformers.ElectraTokenizer,
        }
        models = {
            "openai-gpt":transformers.OpenAIGPTModel,
            "bert-base-uncased": transformers.BertModel,
            "ctrl":transformers.CTRLModel,
            "transfo-xl-wt103": transformers.TransfoXLModel,
            "electra": transformers.ElectraModel,
        }

        add_special_tokens = defaultdict(lambda:True)
        add_decoder_input_ids = defaultdict(lambda:False)

        for name in ["gpt2","gpt2-medium","gpt2-large","gpt2-xl","gpt2-xl-finetune"]:
            tokenizers[name] = transformers.GPT2Tokenizer
            models[name] = transformers.GPT2Model

        for name in ["t5-small","t5-base","t5-large","t5-3b","t5-11b"]:
            tokenizers[name] = transformers.T5Tokenizer
            models[name] = transformers.T5Model
            add_special_tokens[name] = False
            add_decoder_input_ids[name] = True

        for name in ["albert-base-v2","albert-large-v2","albert-xlarge-v2"]:
            tokenizers[name] = transformers.AlbertTokenizer
            models[name] = transformers.AlbertModel

        for name in ["roberta-base","roberta-large"]:
            tokenizers[name] = transformers.RobertaTokenizer
            models[name] = transformers.RobertaModel

        for name in ["xlnet-base-cased","xlnet-large-cased"]:
            tokenizers[name] = transformers.XLNetTokenizer
            models[name] = transformers.XLNetModel
            add_special_tokens[name] = False

        # handle inconsistent naming scheme for electra
        transformer_keys = {"electra":"google/electra-small-discriminator"}
        transformer_key = transformer_keys.get(embedding_name, embedding_name)
        tokenizer = tokenizers[embedding_name].from_pretrained(transformer_key)
        model = models[embedding_name].from_pretrained(transformer_key)

        self.add_special_tokens = add_special_tokens[embedding_name]
        self.add_decoder_input_ids = add_decoder_input_ids[embedding_name]
        super().__init__(
                model = model,
                dim = dim,
                tokenizer = tokenizer,
                remove_stopwords = False,
        )

    @typechecked
    def text2vec(self, text: str) -> Tuple[np.ndarray, List[str]]:
        tokens = self.tokenizer.encode(
                text,
                add_special_tokens=self.add_special_tokens,
#                add_space_before_punct_symbol=True,
        )
        input_idx = torch.LongTensor(tokens).to(self.model.device)
        kwargs = {"input_ids": input_idx.unsqueeze(0)}
        if self.add_decoder_input_ids:
            kwargs["decoder_input_ids"] = input_idx.unsqueeze(0)
        with torch.no_grad():
            hidden_states = self.model(**kwargs)
            embeddings = hidden_states[0].cpu().numpy()
        return embeddings.squeeze(0), [] 

class CLIP:
    def __init__(self, 
                model_name = "ViT-B/32", 
                dim = 512): 
        model, preprocess = clip.load(model_name)
        self.dim = dim
        self.model = model
        self.device = None

    @abstractmethod
    def text2vec(self, text: str) -> np.ndarray:
        text = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_text(text)
            embeddings = embeddings.cpu().numpy()
            return embeddings, []

    @typechecked
    def set_device(self, device: torch.device):
        self.model = self.model.to(device)
        self.device = device




if __name__ == "__main__":
    import argparse
    import pandas as pd
    import pickle
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, 
                        default=300, help='text feature dim')
    parser.add_argument('--weights', default='model/GoogleNews-vectors-negative300.bin',
                        help='path to model weights')
    parser.add_argument('--dataset', default='AudioCaps',
                        help='dataset name to process')
    parser.add_argument('--split',default='train',
                        help='train, val or test')
    parser.add_argument('--model', default='w2v',
            help='text encoding model')
    args = parser.parse_args()
   
    if args.model == 'w2v':    
        txt_enc = W2VEmbedding(dim=args.dim,
                          weights_path=Path(args.weights),
                          remove_stopwords=False)
    elif args.model == 'bert':
        txt_enc = HuggingFaceWrapper(dim = args.dim,
                                 embedding_name='bert-base-uncased')
    elif args.model == 'clip':
        txt_enc = CLIP()
        txt_enc.set_device(torch.device("cuda"))


    if args.dataset == "AudioCaps":
        df = pd.read_csv("data/%s/Index/%s.csv"%(args.dataset,args.split))
        dic= {}
        for youtube_id in df['youtube_id'].unique():
            dic[youtube_id] = []
        for filename, raw_caption in tqdm(zip(df['youtube_id'],df['caption']), total=len(df)):
            embedding,_ = txt_enc.text2vec(raw_caption)
            dic[filename].append(embedding)

    if args.dataset == "CLOTHO" or args.dataset == "CLOTHO_V2":
        df = pd.read_json("data/%s/Index/%s.json"%(args.dataset, args.split))
        dic = {}
        for i in tqdm(range(len(df['audios'])), total=len(df['audios'])):
            dic[df['audios'][i]['audio_id']] = []
            for j in range(5):
                raw_caption = df['audios'][i]['captions'][j]['caption']
                embedding,_ = txt_enc.text2vec(raw_caption)
                dic[df['audios'][i]['audio_id']].append(embedding)
    with open('data/%s/TextEmbeddings/%s_%s.pkl'%(args.dataset, args.model,args.split),'wb') as handle:
        pickle.dump(dic,handle)

    

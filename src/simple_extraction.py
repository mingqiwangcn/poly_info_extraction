from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

import record_extractor
import utils as record_extractor_utils
from utils import LoadNormalizationDataset
from base_classes import GROUPED_SPAN_COLUMNS

import torch
import argparse
import logging
import time
import json

class SimpleExtraction:
    def __init__(self):
        self.timer = {'abstract_preprocessing': [], 'ner': [], 'relation_extraction': []}
        if torch.cuda.is_available():
            print('GPU device found')
            self.device = 0
        else:
            self.device = -1
        self.polymer_filter = True
        self.verbose = True
        self.debug = True
        if not self.debug:
            self.logger = logging.getlogger(__name__)
        else:
            self.logger = None
        model_file = '/home/cc/code/ner_poly_models' # Location of BERT encoder model file to load

        # Load NormalizationDataset used to normalize polymer names
        #normalization_dataloader = LoadNormalizationDataset()
        #self.train_data = normalization_dataloader.process_normalization_files()
        self.train_data = {} # No NormalizationDataset right now

        tokenizer = AutoTokenizer.from_pretrained(model_file, model_max_length=512)
        model = AutoModelForTokenClassification.from_pretrained(model_file)
        # Load model and tokenizer
        self.ner_pipeline = pipeline(task="ner", model=model, tokenizer=tokenizer, grouped_entities=True, device=self.device)

    def read_abstracts(self, mode):
        data_file = f'../data/PolymerAbstracts/{mode}.json'
        with open(data_file) as f:
            for line in f:
                item = json.loads(line)
                words = item['words']
                abstract = ' '.join(words)
                yield abstract

    def start(self, mode):
        for abstract in self.read_abstracts(mode):
            ner_output = self.ner_pipeline(abstract)
            record_extraction_input = record_extractor_utils.ner_feed(ner_output, abstract)
            relation_extractor = record_extractor.RelationExtraction(text=abstract, spans=record_extraction_input, normalization_dataset=self.train_data, polymer_filter=self.polymer_filter, logger=self.logger, verbose=self.verbose)
            try:
                begin = time.time()
                output, _ = relation_extractor.process_document()
                if output:
                    self.timer['relation_extraction'].append(time.time()-begin)
            except Exception as e:
                if not self.debug:
                    self.logger.warning(f'Exception {e} occurred for doi {doi} while parsing the input\n')
                    self.logger.exception(e)
                else:
                    print(f'Exception {e} occurred for doi {doi} while parsing the input\n')
                    print(traceback.format_exc())
                    self.relation_extractor = relation_extractor
            import pdb; pdb.set_trace()
            # Log some metrics when applying model at scale
            if output:
                output['abstract'] = abstract
                if self.verbose:
                    output['material_mentions'] = relation_extractor.material_entity_processor.material_mentions.return_list_dict()
                    output['grouped_spans'] = [named_tuple_to_dict(span) for span in relation_extractor.material_entity_processor.grouped_spans]
                # Insert output to collection
                if not self.debug:
                    self.collection_output.insert_one(output)
                else:
                    print(output)
                    self.relation_extractor = relation_extractor

def named_tuple_to_dict(named_tuple):
    current_dict = {}
    for col in GROUPED_SPAN_COLUMNS:
        current_dict[col] = getattr(named_tuple, col)
    return current_dict

if __name__ == '__main__':
    extractor = SimpleExtraction()
    extractor.start('train')
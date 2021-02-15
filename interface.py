from model.schema_interaction_model import SchemaInteractionATISModel
from data_util.tokenizers import nl_tokenize
from model import torch_utils

import data_util.atis_data as atis_data
from data_util.interaction import Schema

import torch
import json
from dataclasses import dataclass

from debug import register_pdb_hook
register_pdb_hook()

MODEL_LOC = 'logs_test_pretrained/save_31_sparc_editsql'
# HACK:
@dataclass
class Configs:
    no_gpus = False
    # TODO
    data_directory = 'processed_data_sparc_removefrom_test'
    database_schema_filename = 'data/sparc_data_removefrom/tables.json'
    raw_validation_filename = 'data/sparc_data_removefrom/dev.pkl'
    raw_train_filename = 'data/sparc_data_removefrom/train.pkl'
    embedding_filename = 'glove/glove.840B.300d.txt'
    processed_train_filename = 'train.pkl'
    processed_dev_filename = 'dev.pkl'
    processed_validation_filename = 'validation.pkl'
    processed_test_filename = 'test.pkl'
    input_vocabulary_filename = 'input_vocabulary.pkl'
    output_vocabulary_filename = 'output_vocabulary.pkl'
    input_key = "utterance"
    anonymize = False
    anonymization_scoring = False
    # def. model specific
    use_bert = True
    bert_type_abb = 'uS'
    fine_tune_bert = True
    debug = False
    train = False
    output_embedding_size = 300
    input_embedding_size = 300
    encoder_state_size = 300
    decoder_state_size = 300
    encoder_num_layers = 1
    decoder_num_layers = 2
    snippet_num_layers = 1
    use_snippets = False
    discourse_level_lstm = True
    state_positional_embeddings = True
    positional_embedding_size = 50
    use_previous_query = True
    use_query_attention = True
    use_copy_switch = True
    use_schema_encoder = True
    use_schema_encoder_2 = True
    use_schema_attention = True
    use_encoder_attention = True
    use_schema_self_attention = True
    maximum_utterances = 5
    use_utterance_attention = False
    previous_decoder_snippet_encoding = False
    bert_input_version = 'v1'

class OnlineModel(SchemaInteractionATISModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        self.predictions = []
        self.input_hidden_states = []
        self.input_sequences = []
        self.final_utterance_states_c = []
        self.final_utterance_states_h = []
        self.previous_query_states = []
        self.previous_queries = []

        self.discourse_state = None
        if self.params.discourse_level_lstm:
            self.discourse_state, self.discourse_lstm_states = self._initialize_discourse_states()
        self.schema_states = []

    def predict(self, input_sequence, input_schema, max_generation_length=1000):

        if self.previous_queries:
            previous_query = self.previous_queries[-1]
        else:
            previous_query = []

        if input_schema and not self.params.use_bert:
            self.schema_states = self.encode_schema_bow_simple(input_schema)


        if not self.params.use_bert:
            if self.params.discourse_level_lstm:
                utterance_token_embedder = lambda token: torch.cat([self.input_embedder(token),
                                                                    self.discourse_state], dim=0)
            else:
                utterance_token_embedder = self.input_embedder
            final_utterance_state, utterance_states = self.utterance_encoder(
                input_sequence,
                utterance_token_embedder)
        else:
            bert_enc = self.get_bert_encoding(input_sequence, input_schema,
                                              self.discourse_state, dropout=False)
            final_utterance_state, utterance_states, self.schema_states = bert_enc

        self.input_hidden_states.extend(utterance_states)
        self.input_sequences.append(input_sequence)

        num_utterances_to_keep = min(self.params.maximum_utterances, len(self.input_sequences))

        if self.params.discourse_level_lstm:
            fwd = torch_utils.forward_one_multilayer(self.discourse_lstms,
                                                     final_utterance_state[1][0],
                                                     self.discourse_lstm_states)
            _, self.discourse_state, self.discourse_lstm_states = fwd

        if self.params.use_utterance_attention:
            ut_att = self.get_utterance_attention(self.final_utterance_states_c,
                                                  self.final_utterance_states_h,
                                                  self.final_utterance_state, num_utterances_to_keep)
            self.final_utterance_states_c, \
            self.final_utterance_states_h, \
            self.final_utterance_state = ut_att

        if self.params.state_positional_embeddings:
            utterance_states, flat_sequence = \
                self._add_positional_embeddings(self.input_hidden_states, self.input_sequences)
        else:
            flat_sequence = []
            for utt in self.input_sequences[-num_utterances_to_keep:]:
                flat_sequence.extend(utt)

        # TODO: removed snippets as I don't know what it is

        if self.params.use_previous_query and len(previous_query) > 0:
            pq = self.get_previous_queries(self.previous_queries,
                                           self.previous_query_states,
                                           previous_query,
                                           input_schema)
            self.previous_queries, self.previous_query_states = pq

        results = self.predict_turn(final_utterance_state,
                                    utterance_states,
                                    self.schema_states,
                                    max_generation_length,
                                    input_sequence=flat_sequence,
                                    previous_queries=self.previous_queries,
                                    previous_query_states=self.previous_query_states,
                                    input_schema=input_schema,
                                    snippets=None)

        return results

def read_database_schema(database_schema_filename):
    with open(database_schema_filename, "r") as f:
        database_schema = json.load(f)

    database_schema_dict = {}
    column_names_surface_form = []
    column_names_embedder_input = []
    for table_schema in database_schema:
        db_id = table_schema['db_id']
        database_schema_dict[db_id] = table_schema

        column_names = table_schema['column_names']
        column_names_original = table_schema['column_names_original']
        table_names = table_schema['table_names']
        table_names_original = table_schema['table_names_original']

        for i, (table_id, column_name) in enumerate(column_names_original):
            if table_id >= 0:
                table_name = table_names_original[table_id]
                column_name_surface_form = '{}.{}'.format(table_name,column_name)
            else:
                column_name_surface_form = column_name
            column_names_surface_form.append(column_name_surface_form.lower())

        # also add table_name.*
        for table_name in table_names_original:
            column_names_surface_form.append('{}.*'.format(table_name.lower()))

        for i, (table_id, column_name) in enumerate(column_names):
            if table_id >= 0:
                table_name = table_names[table_id]
                column_name_embedder_input = table_name + ' . ' + column_name
            else:
                column_name_embedder_input = column_name
            column_names_embedder_input.append(column_name_embedder_input.split())

        for table_name in table_names:
            column_name_embedder_input = table_name + ' . *'
            column_names_embedder_input.append(column_name_embedder_input.split())

    database_schema = database_schema_dict

    return database_schema, column_names_surface_form, column_names_embedder_input


class DBInterface:

    def __init__(self, db_info_path, max_gen_len=1000):
        self.history = [] # (utterance, sql query)

        # Construct the model
        # TODO: see if data object is necessary
        # TODO: what if vocab needs to be updated based on db_info?
        conf = Configs()
        data = atis_data.ATISDataset(conf)

        self.model = OnlineModel(
            conf,
            data.input_vocabulary,
            data.output_vocabulary,
            data.output_vocabulary_schema,
            data.anonymizer if conf.anonymize and conf.anonymization_scoring else None
        )
        self.model.load(MODEL_LOC)

        self.max_gen_len = max_gen_len
        database_schema, _, _ = read_database_schema(db_info_path)
        self.input_schema = Schema(next(iter(database_schema.values())))
        print('DB Info is loaded')
        print('DB Meta data is blah')

    def predict_ans(self, utterance: str):
        input_sequence = nl_tokenize(utterance)
        pred = self.model.predict(input_sequence, self.input_schema, self.max_gen_len)
        sequence, loss, token_accuracy, _, decoder_results = pred

        if sequence[-1] != '_EOS':
            return 'End of String Pred not found. Can you try another query?'

        return " ".join(sequence[:-1])

    def run_interaction(self):
        while True:
            print('--'*20)
            utterance = input('Q: ')
            if utterance == 'reset':
                # clear interaction
                self.history.clear()
                print('A: Interaction restarted.')
            else:
                ans = self.predict_ans(utterance)
                print(f'A: {ans}')

if __name__ == '__main__':

    app = DBInterface('test_db/tables_tr.json')
    app.run_interaction()
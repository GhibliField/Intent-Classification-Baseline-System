class Config(object):
    def __init__(self):
        self.exp_name="Intent Recognition"
        self.PROPER_NOUN_PARENT='utils/proper_noun'
        self.PREDICATES='utils/predicates.tsv'
        self.PROPER_NOUN_ALL='utils/proper_noun_all.tsv'
        self.PRETTY_SURE='utils/pretty_sure.tsv'
        self.RULES='utils/rules.tsv'
        self.srl_path='models/ltp_data_v3.4.0/pisrl.model'
        self.parser_path='models/ltp_data_v3.4.0/parser.model'
        self.cws_path='models/ltp_data_v3.4.0/cws.model'
        self.pos_path='models/ltp_data_v3.4.0/pos.model'
        self.models_path='models'

    
configs = Config()
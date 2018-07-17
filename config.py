class Config(object):
    def __init__(self):
        self.exp_name="Intent Recognition"

        self.PREDICATES='/home/d0main/NLP_Competition/Intent-Classification_SMP2018-ECDT/utils/predicates.tsv'
        self.PROPER_NOUN_ALL='/home/d0main/NLP_Competition/Intent-Classification_SMP2018-ECDT/utils/proper_noun_all.tsv'
        self.PRETTY_SURE='/home/d0main/NLP_Competition/Intent-Classification_SMP2018-ECDT/utils/pretty_sure.tsv'
        self.RULES='/home/d0main/NLP_Competition/Intent-Classification_SMP2018-ECDT/utils/rules.tsv'
        self.srl_path='/home/d0main/NLP_Competition/Intent-Classification_SMP2018-ECDT/models/ltp_data_v3.4.0/pisrl.model'
        self.parser_path='/home/d0main/NLP_Competition/Intent-Classification_SMP2018-ECDT/models/ltp_data_v3.4.0/parser.model'
        self.cws_path='/home/d0main/NLP_Competition/Intent-Classification_SMP2018-ECDT/models/ltp_data_v3.4.0/cws.model'
        self.pos_path='/home/d0main/NLP_Competition/Intent-Classification_SMP2018-ECDT/models/ltp_data_v3.4.0/pos.model'
        self.models_path='/home/d0main/NLP_Competition/Intent-Classification_SMP2018-ECDT/models'

    
configs = Config()
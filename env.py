#from utils import *
#from config import *
from agent import *
import numpy as np

from tqdm import tqdm  # for progressive bar during loading


class Env:

    def __init__(self, config):
        self.config = config
        self.agents = []
        self.curr_agent = None

    # functions for making environment

    def make_env(self):
        self.index_to_feature, self.feature_to_index = self.get_index_to_feature_and_feature_to_index()
        self.train_set, self.validate_set = self.separate_data_set()

    def get_index_to_feature_and_feature_to_index(self):
        features = self.get_all_features()
        index_to_feature = []
        feature_to_index = {}
        for k in features:
            index_to_feature.append(k)
        for i, k in enumerate(index_to_feature):
            feature_to_index[k] = i
        self.feat_dim = len(index_to_feature)
        return index_to_feature, feature_to_index 

    def get_all_features(self):
        parse_dict = self.config.parse_dict
        gold_trees = self.config.gold_trees
        picks = self.config.picks

        features = {}
        print("reading word problems...")
        for i in tqdm(range(self.config.wp_total_num)):
            p = picks.get(str(i), [])
            agent = Agent(parse_dict[i], gold_trees[i], self.config.reject[i], p)
            agent.get_feature_from_schema_info()
            features.update(agent.get_possible_features())
            self.agents.append(agent)
        return features

    def separate_data_set(self):
        train_set = []
        validate_set = []
        for ind in self.config.train_list:
            train_set.append(self.agents[ind])
        for ind in self.config.validate_list:
            validate_set.append(self.agents[ind])
        return train_set, validate_set

    # other control functions

    def reset_inner_count(self):
        self.count = 0

    def reset(self):
        num = self.count
        self.count += 1
        self.curr_agent = self.train_set[num]
        self.curr_agent.init_state_info(self.index_to_feature)
        return np.array(self.curr_agent.feat_vector)

    def validate_reset(self, iteration):
        self.curr_agent = self.validate_set[iteration]
        self.curr_agent.init_state_info(self.index_to_feature)
        return np.array(self.curr_agent.feat_vector)

    def step(self, action_op):
        next_states, reward, done, flag = self.curr_agent.compound_two_nodes(action_op)
        return next_states, reward, done

    def val_step(self, action_op, loc):
        next_states, done, flag, etime = self.curr_agent.compound_two_nodes_predict(action_op, loc, './predict/analysis_')
        return next_states, done, flag, etime
        

#config = Config()
#e = Env(config)
#index_to_feature, feature_to_index = e.get_index_to_feature_and_feature_to_index()
#e.agents[0].get_feat_vector(0,1,index_to_feature)


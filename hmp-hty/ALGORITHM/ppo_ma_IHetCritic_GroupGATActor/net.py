import torch, math, copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from UTIL.colorful import print亮绿
from UTIL.tensor_ops import Args2tensor_Return2numpy, Args2tensor, __hashn__, my_view
from UTIL.tensor_ops import pt_inf
from UTIL.exp_helper import changed
from .ccategorical import CCategorical
from .foundation import AlgorithmConfig
from ALGORITHM.common.attention import SimpleAttention
from ALGORITHM.common.norm import DynamicNormFix
from ALGORITHM.common.net_manifest import weights_init

class GroupGAT(nn.Module):
    '''
        输入为(batch, num_node, input_dim)
        输出为(batch, hidden_dim)
        mask维度为(batch, num_node)
        对num_node的聚合版本
        区分ally和enemy的版本
    '''
    def __init__(self, input_dim, hidden_dim, output_dim=0, n_heads=1, add_self=True, add_elu=True):
        super(GroupGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.add_self = add_self
        self.add_elu = add_elu

        assert n_heads == 1, '目前还没有涉及多头的形式！'

        self.W_ally = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_opp = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W_ally.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_opp.data, gain=1.414)

        self.a_ally = nn.Parameter(torch.Tensor(hidden_dim*2, 1))
        self.a_opp = nn.Parameter(torch.Tensor(hidden_dim*2, 1))
        nn.init.xavier_uniform_(self.a_ally.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_opp.data, gain=1.414)

        self.act = nn.LeakyReLU(negative_slope=0.2)

    def init_parameters(self):
        params = [self.W_ally, self.W_opp, self.a_ally, self.a_opp]
        for param in params:
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    
    def forward(self, h, num_ally, num_opp, mask=None):
        assert 1 + num_ally + num_opp == h.shape[-2]
        h_self = h[:, 0:1,         :]
        h_ally = h[:, 1:1+num_ally,:]
        h_opp =  h[:, 1+num_ally:, :]
        h_ally = torch.cat([h_self,h_ally],dim=-2)
        h_opp  = torch.cat([h_self,h_opp],dim=-2)

        assert mask is not None, ('Group-GAT needs mask!')
        mask_self = torch.zeros_like(mask[:, 0:1])
        mask_ally = mask[:, 1:1+num_ally]
        mask_opp =  mask[:, 1+num_ally:]
        mask_ally = torch.cat([mask_self,mask_ally],dim=-1)
        mask_opp  = torch.cat([mask_self,mask_opp],dim=-1)

        H_ally = torch.matmul(h_ally, self.W_ally)
        H_opp  = torch.matmul(h_opp, self.W_opp)
        # 注意力机制仅应用于第一个节点与其他节点
        # calculate ally embedding
        B, N, C = H_ally.size()
        ally_input = torch.cat([H_ally[:, 0:1, :].repeat(1, N, 1), H_ally],dim=2).view(B, N, 2*C)
        e_ally = self.act(torch.matmul(ally_input, self.a_ally).squeeze(-1))
        e_ally[mask_ally.bool()] = -math.inf
        ally_attention = F.softmax(e_ally, dim=-1)
        ally_attnc = ally_attention.clone()
        ally_attnc[mask_ally] = 0
        ally_attention = ally_attnc
        ally_weighted_sum = torch.matmul(ally_attention.unsqueeze(1), H_ally).squeeze(1)

        # calculate opp embedding
        B, N, C = H_opp.size()
        opp_input = torch.cat([H_opp[:, 0:1, :].repeat(1, N, 1), H_opp],dim=2).view(B, N, 2*C)
        e_opp = self.act(torch.matmul(opp_input, self.a_opp).squeeze(-1))
        e_opp[mask_opp.bool()] = -math.inf
        opp_attention = F.softmax(e_opp, dim=-1)
        opp_attnc = opp_attention.clone()
        opp_attnc[mask_opp] = 0
        opp_attention = opp_attnc
        opp_weighted_sum = torch.matmul(opp_attention.unsqueeze(1), H_opp).squeeze(1)

        assert self.add_self, ('Group-GAT need to add self info!')

        if self.add_elu:
            H_E = F.elu(H_ally[:, 0, :] + ally_weighted_sum + opp_weighted_sum) 
        else:
            H_E = (H_ally[:, 0, :] + ally_weighted_sum + opp_weighted_sum) 

        return H_E.view(B, self.hidden_dim)



class E_GAT(nn.Module):
    '''
        输入为(batch, num_node, input_dim)
        输出为(batch, hidden_dim)
        mask维度为(batch, num_node)
        对num_node的聚合版本
    '''
    def __init__(self, input_dim, hidden_dim, output_dim=0, n_heads=1, add_self=True, add_elu=True):
        super(E_GAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.add_self = add_self
        self.add_elu = add_elu

        assert n_heads == 1, '目前还没有涉及多头的形式！'

        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.Tensor(hidden_dim*2, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def init_parameters(self):
        params = [self.W, self.a]
        for param in params:
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    
    def forward(self, h, mask=None):
        H = torch.matmul(h, self.W)
        B, N, C = H.size()

        # 注意力机制仅应用于第一个节点与其他节点
        a_input = torch.cat([H[:, 0:1, :].repeat(1, N, 1), H], dim=2).view(B, N, 2*self.hidden_dim)
        e = self.act(torch.matmul(a_input, self.a).squeeze(-1))

        if mask is not None:  
            e[mask.bool()] = -math.inf

        attention = F.softmax(e, dim=-1)
        # If there are nodes with no neighbours then softmax returns nan so fix them to 0
        if mask is not None:
            attnc = attention.clone()
            attnc[mask] = 0
            attention = attnc
        assert not torch.isnan(attention).any(), ('nan problem!')
        weighted_sum = torch.matmul(attention.unsqueeze(1), H).squeeze(1)
        # weighted_sum = torch.matmul(attention, H) 
        assert not torch.isnan(weighted_sum).any(), ('nan problem!')

        if self.add_elu:
            H_E = F.elu(H[:, 0, :] + weighted_sum) if self.add_self else F.elu(weighted_sum)
        else:
            H_E = (H[:, 0, :] + weighted_sum) if self.add_self else (weighted_sum)

        return H_E.view(B, self.hidden_dim)




class I_GAT(nn.Module):
    '''
        输入为(batch, num_node, input_dim)
        输出为(batch, num_node, hidden_dim)
        mask维度为(batch, num_node, num_node)
    '''
    def __init__(self, input_dim, hidden_dim, output_dim=0, n_heads=1, add_self=True, add_elu=True):
        super(I_GAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.add_self = add_self
        self.add_elu = add_elu

        assert n_heads == 1, '目前还没有涉及多头的形式！'

        # 不采用多头的形式
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.Tensor(hidden_dim*2, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        # self.init_parameters()


    def init_parameters(self):
        params = [self.W, self.a]
        # for param in self.parameters():
        for param in params:
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    
    def forward(self, h, mask=None):
        H = torch.matmul(h, self.W)
        B, N, C = H.size()  # B N h_dim
        # (B,N,C) (B,N,N*C),(B,N*N,C),(B,N*N,2C)
        a_input = torch.cat([H.repeat(1, 1, N).view(B, N * N, C), H.repeat(1, N, 1)], dim=2).view(B, N, N, 2*self.hidden_dim)  # [B,N,N,2C]
        e = self.act(torch.matmul(a_input, self.a).squeeze(-1)) # (B N N)

        # mask size should be (B N N)
        if mask is not None:  e[mask.bool()] = -math.inf   
        attention = F.softmax(e, dim=-1)
        # If there are nodes with no neighbours then softmax returns nan so fix them to 0
        if mask is not None:
            attnc = attention.clone()
            attnc[mask] = 0
            attention = attnc
        assert not torch.isnan(attention).any(), ('nan problem!')

        weighted_sum = torch.matmul(attention, H) # (B N N) * (B N C) = (B N C)
        assert not torch.isnan(weighted_sum).any(), ('nan problem!')

        if self.add_elu:
            H_E = F.elu(H + weighted_sum) if self.add_self else F.elu(weighted_sum)
        else:
            H_E = (H + weighted_sum) if self.add_self else (weighted_sum)
        return H_E
                                                                                                                                                                                         

"""
    network initialize
"""
class Net(nn.Module):
    def __init__(self, rawob_dim, n_action, **kwargs):
        super().__init__()
        self.update_cnt = nn.Parameter(
            torch.zeros(1, requires_grad=False, dtype=torch.long), requires_grad=False)
        self.use_normalization = AlgorithmConfig.use_normalization
        self.use_policy_resonance = AlgorithmConfig.policy_resonance
        self.n_action = n_action
        assert self.n_action == AlgorithmConfig.n_action
        
        
        if self.use_policy_resonance:
            self.ccategorical = CCategorical(kwargs['stage_planner'])
            self.is_resonance_active = lambda: kwargs['stage_planner'].is_resonance_active()

        h_dim = AlgorithmConfig.net_hdim

        # observation normalization
        if self.use_normalization:
            self._batch_norm = DynamicNormFix(rawob_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)

        n_entity = AlgorithmConfig.n_entity_placeholder
        
        # # # # # # # # # #  actor-critic share # # # # # # # # # # # #
        self.obs_encoder = nn.Sequential(nn.Linear(rawob_dim, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, h_dim))
        # self.attention_layer = SimpleAttention(h_dim=h_dim)
        # # # # # # # # # #        actor        # # # # # # # # # # # #
        self.at_GAT_layer = GroupGAT(input_dim=h_dim,hidden_dim=h_dim)
        _size = n_entity * h_dim
        self.policy_head = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim//2), nn.ReLU(inplace=True),
            nn.Linear(h_dim//2, self.n_action))
        # # # # # # # # # # critic # # # # # # # # # # # #
        
        _size = n_entity * h_dim
        # self.ct_encoder = nn.Sequential(nn.Linear(_size, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, h_dim))
        self.GRU_encoder = nn.Sequential(nn.Linear(_size + self.n_action, h_dim*2), nn.ReLU(inplace=True), nn.Linear(h_dim*2, h_dim))
        self.rnn = nn.GRUCell(h_dim, h_dim)
        # use orthogonal init for GRU layer0 weights
        nn.init.orthogonal_(self.rnn.weight_ih)
        nn.init.orthogonal_(self.rnn.weight_hh)
        # use zero init for GRU layer0 bias
        nn.init.zeros_(self.rnn.bias_ih)
        nn.init.zeros_(self.rnn.bias_hh)

        self.fc2_rnn = nn.Linear(h_dim, h_dim)
        self.ct_encoder = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),nn.Linear(h_dim, h_dim))
        # E_GAT参数
        self.ct_GAT_layer = I_GAT(input_dim=h_dim,hidden_dim=h_dim)
        # self.ct_attention_layer = SimpleAttention(h_dim=h_dim)
        self.get_value = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),nn.Linear(h_dim, 1))


        self.is_recurrent = False
        self.apply(weights_init)
        return
    
    @Args2tensor_Return2numpy
    def act(self, *args, **kargs):
        return self._act(*args, **kargs)
    
    @Args2tensor
    def evaluate_actions(self, *args, **kargs):
        return self._act(*args, **kargs, eval_mode=True)

    def _act(self, obs=None, action_code=None, gru_cell_memory=None, test_mode=None, eval_mode=False, eval_actions=None, avail_act=None, agent_ids=None, eprsn=None):
        eval_act = eval_actions if eval_mode else None
        assert AlgorithmConfig.n_entity_placeholder == obs.shape[-2], 'Check n_entity'
        others = {}
        if self.use_normalization:
            if torch.isnan(obs).all(): pass
            else: obs = self._batch_norm(obs, freeze=(eval_mode or test_mode))

        mask_dead = torch.isnan(obs).any(-1)
        obs = torch.nan_to_num_(obs, 0)         # replace dead agents' obs, from NaN to 0
        
        # # # # # # # # # # actor-critic share # # # # # # # # # # # #
        baec = self.obs_encoder(obs)
        # baec = self.attention_layer(k=baec,q=baec,v=baec, mask=mask_dead)

        # # # # # # # # # # actor # # # # # # # # # # # #
        at_bac = my_view(baec,[-1,0,0])
        mask_dead_squeeze = my_view(mask_dead,[-1,0])
        at_bac = self.at_GAT_layer(at_bac, num_ally= AlgorithmConfig.num_ally, num_opp=AlgorithmConfig.num_opp, mask=mask_dead_squeeze)
        at_bac = at_bac.view([obs.shape[0],obs.shape[1],at_bac.shape[-1]])
        logits = self.policy_head(at_bac)
        
        # choose action selector
        logit2act = self._logit2act_rsn if self.use_policy_resonance and self.is_resonance_active() else self._logit2act
        
        # apply action selector
        act, actLogProbs, distEntropy, probs = logit2act(   logits, 
                                                            eval_mode=eval_mode,
                                                            test_mode=test_mode, 
                                                            eval_actions=eval_act, 
                                                            avail_act=avail_act,
                                                            eprsn=eprsn)
        
        
        # # # # # # # # # # critic # # # # # # # # # # # #
        ct_bac = my_view(baec,[0,0,-1])
        ct_act = action_code
        gru_input = torch.cat((ct_bac, ct_act),-1)
        gru_input = self.GRU_encoder(gru_input)

        gru_input_expand = gru_input.view(gru_input.shape[0]*gru_input.shape[1], gru_input.shape[2])
        gru_cell_memory_expand = gru_cell_memory.view(gru_cell_memory.shape[0]*gru_cell_memory.shape[1], gru_cell_memory.shape[2])
        gru_cell_output = self.rnn(gru_input_expand, gru_cell_memory_expand)
        gru_cell_output = gru_cell_output.view(gru_cell_memory.shape)

        ct_bac = self.ct_encoder(gru_cell_output)
        ct_bac = self.ct_GAT_layer(h=ct_bac)
        # ct_bac = self.ct_attention_layer(k=ct_bac,q=ct_bac,v=ct_bac)
        value = self.get_value(ct_bac)
        
        if not eval_mode: return act, value, actLogProbs, gru_cell_output
        else:             return value, actLogProbs, distEntropy, probs, others, gru_cell_output

    def _logit2act_rsn(self, logits_agent_cluster, eval_mode, test_mode, eval_actions=None, avail_act=None, eprsn=None):
        if avail_act is not None: logits_agent_cluster = torch.where(avail_act>0, logits_agent_cluster, -pt_inf())
        act_dist = self.ccategorical.feed_logits(logits_agent_cluster)
        
        if not test_mode: act = self.ccategorical.sample(act_dist, eprsn) if not eval_mode else eval_actions
        else:             act = torch.argmax(act_dist.probs, axis=2)
        # the policy gradient loss will feedback from here
        actLogProbs = self._get_act_log_probs(act_dist, act) 
        # sum up the log prob of all agents
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None
        return act, actLogProbs, distEntropy, act_dist.probs

    def _logit2act(self, logits_agent_cluster, eval_mode, test_mode, eval_actions=None, avail_act=None, **kwargs):
        if avail_act is not None: logits_agent_cluster = torch.where(avail_act>0, logits_agent_cluster, -pt_inf())
        act_dist = Categorical(logits = logits_agent_cluster)
        if not test_mode:  act = act_dist.sample() if not eval_mode else eval_actions
        else:              act = torch.argmax(act_dist.probs, axis=2)
        actLogProbs = self._get_act_log_probs(act_dist, act) # the policy gradient loss will feedback from here
        # sum up the log prob of all agents
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None
        return act, actLogProbs, distEntropy, act_dist.probs

    @staticmethod
    def _get_act_log_probs(distribution, action):
        return distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)
    
    

    
    
import math
import numpy as np
from .foundation import AlgorithmConfig
from UTIL.colorful import *

class PolicyRsnConfig:
    resonance_start_at_update = 1
    yita_min_prob = 0.15  #  should be >= (1/n_action)
    yita_max = 0.5
    yita_inc_per_update = 0.0075 # (increase to 0.75 in 500 updates)
    freeze_critic = False
    
    yita_shift_method = '-sin'
    yita_shift_cycle = 1000


class StagePlanner:
    def __init__(self, mcv) -> None:
        if AlgorithmConfig.use_policy_resonance:
            self.resonance_active = False
            self.yita = 0
            self.yita_min_prob = PolicyRsnConfig.yita_min_prob
        self.freeze_body = False
        self.update_cnt = 0
        self.mcv = mcv
        self.trainer = None
        
        if PolicyRsnConfig.yita_shift_method == 'feedback':
            from .scheduler import FeedBackPolicyResonance
            self.feedback_controller = FeedBackPolicyResonance(mcv)
        else:
            self.feedback_controller = None
        # if AlgorithmConfig.wait_norm_stable:
        #     self.wait_norm_stable_cnt = 2
        # else:
        #     self.wait_norm_stable_cnt = 0
        # return

    def uprate_eprsn(self, n_thread):
        eprsn_yita = self.yita
        EpRsn = np.random.rand(n_thread) < eprsn_yita
        self.eprsn = EpRsn

    def is_resonance_active(self,):
        return self.resonance_active
    
    def is_body_freeze(self,):
        return self.freeze_body
    
    def get_yita(self):
        return self.yita
    
    def get_yita_min_prob(self):
        return PolicyRsnConfig.yita_min_prob
    
    def can_exec_trainning(self):
        return True

    def update_plan(self):
        self.update_cnt += 1
        if AlgorithmConfig.use_policy_resonance:
            if self.resonance_active:
                self.when_pr_active()
            elif not self.resonance_active:
                self.when_pr_inactive()
        return

    def update_test_winrate(self, win_rate):
        if self.feedback_controller is not None:
            self.feedback_controller.step(win_rate)
    
    def activate_pr(self):
        self.resonance_active = True
        self.freeze_body = True
        if PolicyRsnConfig.freeze_critic:
            self.trainer.freeze_body()

    def when_pr_inactive(self):
        assert not self.resonance_active
        if PolicyRsnConfig.resonance_start_at_update >= 0:
            # mean need to activate pr later
            if self.update_cnt > PolicyRsnConfig.resonance_start_at_update:
                # time is up, activate pr
                self.activate_pr()
        # log
        pr = 1 if self.resonance_active else 0
        self.mcv.rec(pr, 'resonance')
        self.mcv.rec(self.yita, 'yita')
    def when_pr_active(self):
        assert self.resonance_active
        self._update_yita()
        # log
        pr = 1 if self.resonance_active else 0
        self.mcv.rec(pr, 'resonance')
        self.mcv.rec(self.yita, 'yita')

    def _update_yita(self):
        '''
            increase self.yita by @yita_inc_per_update per function call
        '''
        if PolicyRsnConfig.yita_shift_method == '-cos':
            self.yita = PolicyRsnConfig.yita_max
            t = -math.cos(2*math.pi/PolicyRsnConfig.yita_shift_cycle * self.update_cnt) * PolicyRsnConfig.yita_max
            if t<=0: self.yita = 0
            else: self.yita = t
            print亮绿('yita update:', self.yita)

        elif PolicyRsnConfig.yita_shift_method == '-sin':
            self.yita = PolicyRsnConfig.yita_max
            t = -math.sin(2*math.pi/PolicyRsnConfig.yita_shift_cycle * self.update_cnt) * PolicyRsnConfig.yita_max
            if t<=0: self.yita = 0
            else: self.yita = t
            print亮绿('yita update:', self.yita)

        elif PolicyRsnConfig.yita_shift_method == 'slow-inc':
            self.yita += PolicyRsnConfig.yita_inc_per_update
            if self.yita > PolicyRsnConfig.yita_max:
                self.yita = PolicyRsnConfig.yita_max
            print亮绿('yita update:', self.yita)

        elif PolicyRsnConfig.yita_shift_method == 'feedback':
            self.yita = self.feedback_controller.recommanded_yita
            print亮绿('yita update:', self.yita)
            
        else:
            assert False


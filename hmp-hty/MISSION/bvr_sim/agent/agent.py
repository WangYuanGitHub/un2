"""

"""
from typing import List


class Agent(object):
    """
        用于框架中进行训练的算法的基类，选手通过继承该类，去实现自己的策略逻辑
    @Examples:
        添加使用示例

    """
    def __init__(self, name, side, **kwargs):
        """必要的初始化"""
        self.name = name
        self.side = side

    def reset(self, **kwargs):
        pass

    def step(self, **kwargs) -> List[dict]:
        """输入态势信息，返回指令列表"""
        raise NotImplementedError


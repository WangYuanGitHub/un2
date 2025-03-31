#-*-coding:utf-8-*-
"""
@FileName：xsim_env.py
@Description：xsim环境交互类

@Department：AIStudio研发部
@Copyright：©2011-2021 北京华如科技股份有限公司
"""
from ..env.xsim_manager import XSimManager
from ..env.communication_service import CommunicationService


class XSimEnv(object):
    """
        仿真环境类
        对于用户来说，如果想要与xsim环境连接，只需要实例化一个XSimEnv实例即可
        - 通过 step(action)->obs 将任务指令发送至xsim引擎，然后推进引擎，同时，引擎返回执行action后的observation
        - 通过 reset()重置xsim环境
        - 通过 close()关闭xsim环境
    @Examples:
        添加使用示例
        # 创建xsim环境
		xsim_env = XSimEnv()

    """
    def __init__(self, time_ratio: int, address: str, image_name='bvrsim:v1.0', mode: str = 'host'):
        """

        """
        # xsim引擎控制器
        self.xsim_manager = XSimManager(time_ratio, address, image_name, mode)
        # 与xsim引擎交互通信服务
        self.communication_service = CommunicationService(self.xsim_manager.address)

    def __del__(self):
        self.xsim_manager.close_env()

    def step(self, action: list) -> dict:
        """
        用户与xsim环境交互核心函数。通过step控制引擎的推进。

        """
        try:
            obs = self.communication_service.step(action)
            return obs
        except Exception as e:
            print(e)
        # return self.communication_service.step(action)

    def reset(self):
        """
        重置训练环境

        """
        return self.communication_service.reset()

    def end(self):
        """
        重置训练环境

        """
        return self.communication_service.end()

    def close(self) -> bool:
        """

        """
        self.xsim_manager.close_env()
        self.communication_service.close()
        return True

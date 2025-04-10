base = """
{
    // --- Part1: config HMP core --- 
    "config.py->GlobalConfig": {
        "note": "1stEXP-PGAT_mappo_twoflow-uhmap50vs50",// http://localhost:59547
        "env_name": "uhmap",
        "env_path": "MISSION.uhmap",
        // "heartbeat_on": "False",
        "draw_mode": "Img",
        "num_threads": 16,  // 请预留 num_threads * 1 GB 的内存空间
        "report_reward_interval": 128,
        "test_interval": 1280,
        "test_epoch": 512,
        "interested_team": 0,
        "seed": 10098,
        "device": "cuda:5",
        "max_n_episode": 5000000,
        "fold": 1,
        "backup_files": [
            "ALGORITHM/PGAT_mappo_twoflow",
            "MISSION/uhmap"
        ]
    },


    // --- Part2: config MISSION --- 
    "MISSION.uhmap.uhmap_env_wrapper.py->ScenarioConfig": {
        "N_AGENT_EACH_TEAM": [50, 50], // update N_AGENT_EACH_TEAM
        "MAX_NUM_OPP_OBS": 10,
        "MAX_NUM_ALL_OBS" :10,
        "MaxEpisodeStep": 150,
        "StepGameTime": 0.5,
        "StateProvided": false,
        "render": false, // note: random seed has different impact on renderer and server
        "UElink2editor": false,
        "HeteAgents": true,
        "UnrealLevel": "UhmapLargeScale",
        "SubTaskSelection": "UhmapHuge",
        "UhmapVersion":"3.5",
        "UhmapRenderExe": "/home/hmp/UnrealHmapBinary/Version3.5/LinuxNoEditor/UHMP.sh",
        "UhmapServerExe": "/home/hmp/UnrealHmapBinary/Version3.5/LinuxServer/UHMPServer.sh",
        "TimeDilation": 64, // simulation time speed up, larger is faster
        "TEAM_NAMES": [
            "ALGORITHM.PGAT_mappo_twoflow.foundation->ReinforceAlgorithmFoundation",
            "ALGORITHM.script_ai.uhmap_ls->DummyAlgorithmLinedAttack",
        ]
    },
    "MISSION.uhmap.SubTasks.UhmapHugeConf.py->SubTaskConfig":{
        "agent_list": [
            { "team":0,  "tid":0,    "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
            { "team":0,  "tid":1,    "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":2,    "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":3,    "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":4,    "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":5,    "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":6,    "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":7,    "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":8,    "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":9,    "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":10,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":11,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":12,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":13,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":14,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":15,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":16,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":17,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":18,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":19,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":20,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":21,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":22,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":23,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":24,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":25,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":26,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":27,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":28,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":29,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":30,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":31,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":32,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":33,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":34,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":35,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":36,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":37,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":38,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":39,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":40,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":41,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":42,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":43,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":44,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":45,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":46,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
            { "team":0,  "tid":47,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
            { "team":0,  "tid":48,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
            { "team":0,  "tid":49,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },

            { "team":1,  "tid":0,    "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
            { "team":1,  "tid":1,    "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":2,    "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":3,    "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":4,    "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":5,    "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":6,    "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":7,    "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":8,    "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":9,    "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":10,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":11,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":12,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":13,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":14,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":15,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":16,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":17,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":18,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":19,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":20,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":21,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":22,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":23,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":24,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":25,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":26,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":27,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":28,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":29,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":30,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":31,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":32,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":33,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":34,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":35,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":36,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":37,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":38,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":39,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":40,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":41,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":42,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":43,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":44,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":45,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":46,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
            { "team":1,  "tid":47,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
            { "team":1,  "tid":48,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
            { "team":1,  "tid":49,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },

        ]
    },






    // --- Part3: config ALGORITHM 1/2 --- 
    "ALGORITHM.script_ai.uhmap_ls.py->DummyAlgConfig": {
        "reserve": ""
    },

    // --- Part3: config ALGORITHM 2/2 --- 
    "ALGORITHM.PGAT_mappo_twoflow.shell_env.py->ShellEnvConfig": {
        "add_avail_act": true
    },
    "ALGORITHM.PGAT_mappo_twoflow.foundation.py->AlgorithmConfig": {
        "train_traj_needed": 32,
        "use_normalization": true,
        "use_obs_pro_uhmp": false,
        "load_specific_checkpoint": "",
        "gamma": 0.99,
        "gamma_in_reward_forwarding": "True",
        "gamma_in_reward_forwarding_value": 0.95,
        "prevent_batchsize_oom": "True",
        "lr": 0.0004,
        "ppo_epoch": 24,
        "policy_resonance": false,
        "debug": true,
        "n_entity_placeholder": 21,
    },

    "ALGORITHM.PGAT_mappo_twoflow.stage_planner.py->PolicyRsnConfig": {
        "resonance_start_at_update": 1,
        "yita_min_prob": 0.05,
        "yita_max": 0.5,
        "yita_shift_method": "-sin",
        "yita_shift_cycle": 1000,
        "yita_inc_per_update": 0.01,
    },
    
}
"""


import commentjson as json
import numpy as np
base_conf = json.loads(base)
n_run = 1
n_run_mode = [
    {
        "addr": "server.yiteam.tech:12326",
        "usr": "hmp",
        "pwd": "hmp"
    },
]*n_run
assert len(n_run_mode)==n_run

sum_note = "db-unknown"
conf_override = {

    "config.py->GlobalConfig-->seed":       
        [
            np.random.randint(0, 10000) for _ in range(n_run)
        ],

}

if __name__ == '__main__':
    # copy the experiments
    import shutil, os
    shutil.copyfile(__file__, os.path.join(os.path.dirname(__file__), 'batch_experiment_backup.py'))
    # run experiments remotely
    from UTIL.batch_exp import run_batch_exp
    run_batch_exp(sum_note, n_run, n_run_mode, base_conf, conf_override, __file__)

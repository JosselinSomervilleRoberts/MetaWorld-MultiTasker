import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict

from collections import OrderedDict

from metaworld.envs.mujoco.sawyer_xyz.v2 import (
    SawyerPickPlaceEnvV2,
    SawyerPushEnvV2,
    SawyerReachEnvV2,
)


MT3_V2 = OrderedDict(
    (('reach-v2', SawyerReachEnvV2),
     ('push-v2', SawyerPushEnvV2),
     ('pick-place-v2', SawyerPickPlaceEnvV2) )
)

MT3_V2_ARGS_KWARGS = {
    key: dict(args=[],
              kwargs={'task_id': list(_env_dict.ALL_V2_ENVIRONMENTS.keys()).index(key)})
    for key, _ in MT3_V2.items()
}

class MT3(metaworld.Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = MT3_V2
        self._test_classes = OrderedDict()
        train_kwargs = MT3_V2_ARGS_KWARGS
        self._train_tasks = metaworld._make_tasks(self._train_classes, train_kwargs,
                                        metaworld._MT_OVERRIDE,
                                        seed=seed)
        self._test_tasks = []

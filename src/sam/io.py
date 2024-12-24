import numpy as np
from dataclasses_json import config
from dataclasses import field


def make_np_config():
    return field(
        metadata=config(
            encoder=lambda arr: arr.tolist(),
            decoder=lambda lst: np.array(lst, dtype=np.float32),
        )
    )

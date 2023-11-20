import datetime
import time
import os
import json

import numpy as np

# numbers = [int(re.search(rf"exam_{self.num_t}_{self.num_seed}_{self.file_num}_(\d+)_.*", f).group(1)) for f in os.listdir('.') if re.search(rf"exam_{self.num_t}_{self.num_seed}_{self.file_num}_(\d+)_.*", f)]
def save(make):
    def made(*args, **kwargs):
        current_time = datetime.datetime.now().strftime('%y%m%d %H-%M-%S')
        info, hyp, result = make(*args, **kwargs)
        
        base = f"[{current_time}] {' '.join(map(str, info.values()))} {' '.join([f'{key}_{value}' for key, value in hyp.items()])}".strip()
        os.makedirs(base, exist_ok=True)
        
        with open(os.path.join(base, 'info.json'), "w") as file:    file.write(json.dumps(info))
        with open(os.path.join(base, 'hyp.json'), "w") as file:    file.write(json.dumps(hyp))

        for key, value in result.items():
            np.save(os.path.join(base, key), value)

        return result
    return made

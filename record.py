from utils import *

import datetime
import time

def make_answers(num_t, num_seed, seed=0):
    np.random.seed(seed)
    answers = np.random.random((num_seed, num_t))

    numbers = [int(re.search(r"answers_\d+_\d+_(\d+).npy", f).group(1)) for f in os.listdir('.') if re.search(r"answers_\d+_\d+_(\d+).npy", f)]
    np.save(f"answers_{num_t}_{num_seed}_{max(numbers, default=0) + 1}.npy", answers)

class Exam(Query):
    def __init__(self, query, correct, dim, true_theta, about, verbose=1, init_theta_hat=None, init_neg_hessian=None, file_num=None):
        file_dict = {int(re.search(r"answers_\d+_\d+_(\d+).npy", f).group(1)): f for f in os.listdir('.') if re.match(r"answers_\d+_\d+_(\d+).npy", f)}
        self.file_num = max(file_dict.keys()) if not file_num else file_num
        self.about = about

        super().__init__(query, correct, dim, true_theta, np.load(file_dict[self.file_num]), verbose, init_theta_hat, init_neg_hessian)

        numbers = [int(re.search(rf"exam_{self.num_t}_{self.num_seed}_{self.file_num}_(\d+)_.*", f).group(1)) for f in os.listdir('.') if re.search(rf"exam_{self.num_t}_{self.num_seed}_{self.file_num}_(\d+)_.*", f)]
        self.folder = f"exam_{self.num_t}_{self.num_seed}_{self.file_num}_{max(numbers, default=0) + 1}_{self.about}"
        os.makedirs(self.folder, exist_ok=True)

        self.current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.start_time, self.end_time = time.time(), None
        with open(os.path.join(self.folder, "time.txt"), "a") as f:
            f.write(f"start : {self.current_time}\n")

        np.save(os.path.join(self.folder, "true_theta.npy"), self.true_theta)

    def save(self): 
        np.save(os.path.join(self.folder, "x_seed_t.npy"), self.x_seed_t)
        np.save(os.path.join(self.folder, "y_seed_t.npy"), self.y_seed_t)
        np.save(os.path.join(self.folder, "correct_rate_seed_t.npy"), self.correct_rate_seed_t)
        np.save(os.path.join(self.folder, "theta_hat_seed_t.npy"), self.theta_hat_seed_t)
        np.save(os.path.join(self.folder, "neg_hessian_seed_t.npy"), self.neg_hessian_seed_t)

        self.end_time = time.time()
        with open(os.path.join(self.folder, "time.txt"), "a") as f:
            f.write(f"execute : {self.end_time - self.start_time:.2f} s\n")
        self.start_time = time.time()
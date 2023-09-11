import matplotlib.pyplot as plt
from utils import *

class Exam(Query):
    def __init__(self, query, correct, dim, true_theta, verbose=1, file_num=None):
        file_dict = {int(re.search(r"answers_\d+_\d+_(\d+).npy", f).group(1)): f for f in os.listdir('.') if re.match(r"answers_\d+_\d+_(\d+).npy", f)}
        self.file_num = max(file_dict.keys()) if not file_num else file_num

        super().__init__(query, correct, dim, true_theta, np.load(file_dict[self.file_num]), verbose)

    def initialize(self, query, true_theta):
        super().initialize(query, true_theta)

        numbers = [int(re.search(rf"exam_{self.num_t}_{self.num_seed}_{self.file_num}_(\d+).npy", f).group(1)) for f in os.listdir('.') if re.search(rf"exam_{self.num_t}_{self.num_seed}_{self.file_num}_(\d+).npy", f)]
        self.folder = f"exam_{self.num_t}_{self.num_seed}_{self.file_num}_{max(numbers, default=0) + 1}"
        os.makedirs(self.folder, exist_ok=True)

        np.save(os.path.join(self.folder, "true_theta.npy"), self.true_theta)

    def save(self, about='test'):
        np.save(os.path.join(self.folder, "x_seed_t.npy"), self.x_seed_t)
        np.save(os.path.join(self.folder, "y_seed_t.npy"), self.y_seed_t)
        np.save(os.path.join(self.folder, "correct_rate_seed_t.npy"), self.correct_rate_seed_t)
        np.save(os.path.join(self.folder, "theta_hat_seed_t.npy"), self.theta_hat_seed_t)
        np.save(os.path.join(self.folder, "neg_hessian_seed_t.npy"), self.neg_hessian_seed_t)

        # plot
        determinants = [1/np.linalg.det(neg_hessian) for neg_hessian in self.neg_hessian_seed_t[self.seed-1]]
        plt.plot(range(len(determinants)), determinants)
        plt.xlabel("step t")
        plt.ylabel("det of covariance")
        plt.title(f"Rule : {about}, seed : {self.seed}")
        if self.verbose>1: plt.show()
        plt.savefig(os.path.join(self.folder, f"det_by_t_{self.seed}.png"))
        plt.close()
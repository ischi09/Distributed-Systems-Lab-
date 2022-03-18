import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import optim
from tqdm import tqdm

from .datasets import SetDataset
from .networks import DeepSetsInvariant, MLP, accumulate_sum


class DeepSetExperiment:
    def __init__(
        self, type: str, use_multisets: bool, log_dir: str, lr=1e-3, weight_decay=5e-3
    ):
        self.use_cuda = torch.cuda.is_available()

        # Set up dataset.
        if type == "max":

            def label_generator(x: torch.Tensor):
                return x.max()

        elif type == "mode":

            def label_generator(x: torch.Tensor):
                return torch.squeeze(x).mode().values

        elif type == "cardinality":

            def label_generator(x: torch.Tensor):
                return torch.tensor(len(x), dtype=torch.float)

        else:

            def label_generator(x: torch.Tensor):
                return x.sum()

        self.train_set = SetDataset(
            n_samples=10000,
            max_set_size=10,
            min_value=0,
            max_value=10,
            label_generator=label_generator,
            generate_multisets=use_multisets,
        )

        self.test_set = SetDataset(
            n_samples=1000,
            max_set_size=10,
            min_value=0,
            max_value=10,
            label_generator=label_generator,
            generate_multisets=use_multisets,
        )

        # Set up model.
        self.model = DeepSetsInvariant(
            phi=MLP(input_dim=1, hidden_dim=10, output_dim=10),
            rho=MLP(input_dim=10, hidden_dim=10, output_dim=1),
            accumulator=accumulate_sum,
        )

        if self.use_cuda:
            self.model.cuda()

        # Set up optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Set up logging.
        self.summary_writer = SummaryWriter(
            log_dir=f"{log_dir}/exp-lr:{lr}-wd:{weight_decay}"
        )

    def train_epoch(self, epoch_num: int):
        self.model.train()
        for i in tqdm(range(len(self.train_set))):
            loss = self.train_item(i)

            self.summary_writer.add_scalar(
                "train_loss", loss, i + len(self.train_set) * epoch_num
            )

    def train_item(self, index: int) -> float:
        x, target = self.train_set[index]
        if self.use_cuda:
            x, target = x.cuda(), target.cuda()

        self.optimizer.zero_grad()
        pred = self.model.forward(x)
        # To prevent error warning about mismatching dimensions.
        pred = torch.squeeze(pred, dim=0)
        the_loss = F.mse_loss(pred, target)

        the_loss.backward()
        self.optimizer.step()

        the_loss_tensor = the_loss.data
        if self.use_cuda:
            the_loss_tensor = the_loss_tensor.cpu()

        the_loss_numpy = the_loss_tensor.numpy().flatten()
        the_loss_float = float(the_loss_numpy[0])

        return the_loss_float

    def evaluate(self) -> float:
        self.model.eval()

        n_correct = 0
        for i in tqdm(range(len(self.test_set))):
            x, target = self.test_set[i]

            if self.use_cuda:
                x = x.cuda()

            pred = self.model.forward(x)
            # To prevent error warning about mismatching dimensions.
            pred = torch.squeeze(pred, dim=0)

            if self.use_cuda:
                pred = pred.cpu().numpy().flatten()

            error = torch.abs(target - pred)
            if error < 0.1:
                n_correct += 1

        return n_correct / len(self.test_set)

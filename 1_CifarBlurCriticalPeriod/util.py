"""Data loading, Fisher information, weight norm and representation helpers used by the experiments in this folder."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

FISHER_CHUNK_SIZE = 16  # per-sample gradients are tens of MB each, so don't do them all at once
STABLE_RANK_COVERAGE = 0.99  # the top-k singular values must cover this share of the total
SIGREG_KNOTS = 17
SIGREG_PROJECTIONS = 256
SIGREG_SEED = 0  # the random projection directions are drawn once from this seed, so every reading uses the same ones
STANDARDIZE_EPS = 1e-6  # keeps dead units (zero variance over the batch) at zero instead of dividing by zero
DATASETS = {
    "cifar10": {"cls": datasets.CIFAR10, "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "mnist": {"cls": datasets.MNIST, "mean": (0.1307,), "std": (0.3081,)},
}


def get_loaders(dataset, batch_size, train_augmentation=(), data_root="../datasets"):
    config = DATASETS[dataset]
    normalize = [transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"])]
    train_transform = transforms.Compose([*train_augmentation, *normalize])
    clean_transform = transforms.Compose(normalize)
    print(f"Downloading {dataset} dataset...")
    train_set = config["cls"](root=data_root, train=True, download=True, transform=train_transform)
    # Same training images without augmentation, so Fisher is measured on identical inputs every time.
    fisher_set = config["cls"](root=data_root, train=True, download=True, transform=clean_transform)
    test_set = config["cls"](root=data_root, train=False, download=True, transform=clean_transform)
    print(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=512, num_workers=2)
    return fisher_set, train_loader, test_loader


def compute_fisher(model:nn.Module, x, y=None):
    """Trace of the true Fisher per layer: E_x E_{y~p(y|x)} ||d log p(y|x) / d theta_layer||^2.

    Labels are sampled from the model's own predictions, unless given explicitly in y.
    Returns {"layer_1": float, ...}, with a hidden layer's linear and batch norm parameters counted together.
    Batch norm uses its running statistics, since it can't normalise a single sample by itself.
    """
    was_training = model.training
    model.eval()
    params = {name: p.detach() for name, p in model.named_parameters()}

    def log_prob(params, sample, label):
        logits = functional_call(model, params, (sample.unsqueeze(0),))
        return F.log_softmax(logits, dim=1).gather(1, label.view(1, 1)).squeeze()

    per_sample_grad = vmap(grad(log_prob), in_dims=(None, 0, 0))

    totals = [0.0] * len(model.layers)
    for start in range(0, len(x), FISHER_CHUNK_SIZE):
        chunk = x[start:start + FISHER_CHUNK_SIZE]
        with torch.no_grad():
            if y is None:
                labels = torch.multinomial(F.softmax(model(chunk), dim=1), 1).squeeze(1)
            else:
                labels = y[start:start + FISHER_CHUNK_SIZE]
        grads = per_sample_grad(params, chunk, labels)
        for name, g in grads.items():
            layer_index = int(name.split(".")[1])
            totals[layer_index] += g.flatten(start_dim=1).pow(2).sum().item()

    model.train(was_training)
    return {f"layer_{i + 1}": total / len(x) for i, total in enumerate(totals)}


def log_fisher(writer, model, fisher_x, step, prefix="fisher information"):
    for layer_name, value in compute_fisher(model, fisher_x).items():
        writer.add_scalar(f"{prefix}/{layer_name}", value, step)


def compute_weight_norms(model):
    """L2 norm of the parameters, per layer and in total.

    Returns {"total": float, "layer_1": float, ...}, with a layer's linear and batch norm parameters counted together.
    """
    squares = [sum(p.pow(2).sum().item() for p in layer.parameters()) for layer in model.layers]
    norms = {f"layer_{i + 1}": square ** 0.5 for i, square in enumerate(squares)}
    return {"total": sum(squares) ** 0.5, **norms}


def log_weight_norm(writer, model, step):
    for name, value in compute_weight_norms(model).items():
        writer.add_scalar(f"weight norm/{name}", value, step)


def compute_ranks(features:torch.Tensor):
    """Effective rank and stable rank delta of a (samples, features) matrix, from its singular values.

    With p the singular values normalised by their sum (not squared):
    effective rank = exp(-sum p_i log p_i)
    stable rank delta = min{k : sum of the k largest p_i >= STABLE_RANK_COVERAGE}
    """
    singular_values:torch.Tensor = torch.linalg.svdvals(features.double())  # svdvals sorts in descending order
    p = singular_values / singular_values.sum()
    effective_rank = torch.special.entr(p).sum().exp().item()
    stable_rank_delta = (p.cumsum(dim=0) < STABLE_RANK_COVERAGE).sum().item() + 1
    return {"effective_rank": effective_rank, "stable_rank_delta": stable_rank_delta}


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer from LeJEPA (github.com/galilai-group/lejepa, MINIMAL.md).

    Projects the (samples, features) matrix onto SIGREG_PROJECTIONS random unit directions and, along each, compares the
    empirical characteristic function of the projected samples with that of a standard normal, integrated over t in [0, 3]
    with a Gaussian window. 0 means every projection looks like N(0, 1); larger is further from it.

    Unlike the reference, the directions are drawn once, from a generator seeded with SIGREG_SEED, so the metric uses the same
    directions at every reading and doesn't draw from the global random number generator, and they follow the input's device.
    """

    def __init__(self, features, knots=SIGREG_KNOTS, projections=SIGREG_PROJECTIONS, seed=SIGREG_SEED):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        directions = torch.randn(features, projections, generator=torch.Generator().manual_seed(seed))
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self.register_buffer("directions", directions / directions.norm(p=2, dim=0))

    def forward(self, proj):
        x_t = (proj @ self.directions.to(proj.device)).unsqueeze(-1) * self.t.to(proj.device)
        err = (x_t.cos().mean(-3) - self.phi.to(proj.device)).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights.to(proj.device)) * proj.size(-2)
        return statistic.mean()


_sigreg = {}  # feature count -> SIGReg, so the directions are drawn once per process


def compute_sigreg(features:torch.Tensor):
    """SIGReg of a (samples, features) matrix, as it is and with each feature standardised over the samples.

    The raw value also counts how far the features' means and scales are from 0 and 1 (penultimate ResNet features are
    non-negative, after a ReLU); the standardised one only their shape.
    Returns {"sigreg": float, "sigreg_standardized": float}.
    """
    features = features.float()
    if features.size(-1) not in _sigreg:
        _sigreg[features.size(-1)] = SIGReg(features.size(-1))
    sigreg = _sigreg[features.size(-1)]
    standardized = (features - features.mean(dim=0)) / (features.std(dim=0) + STANDARDIZE_EPS)
    return {"sigreg": sigreg(features).item(), "sigreg_standardized": sigreg(standardized).item()}


def log_representation(writer, model, x, step, prefix="representation"):
    """Ranks and SIGReg of the penultimate features of x, logged under prefix/. Batch norm uses its running statistics."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        features = model.features(x)
    model.train(was_training)
    for name, value in {**compute_ranks(features), **compute_sigreg(features)}.items():
        writer.add_scalar(f"{prefix}/{name}", value, step)

# Training image for the HPC. hpc_run.py builds and pushes it, tagged with a hash of the code.
# Versions match the experiments conda env. The default PyPI torch wheel is a CUDA build.
FROM python:3.12-slim

RUN pip install --no-cache-dir numpy==2.5.4 tensorboard==2.21.0 torch==2.15.0 torchvision==0.29.2

ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY exp_*.py networks.py util.py training_procedure.py /app/

CMD ["python", "exp_01.py", "--data-root", "/workspace/datasets/fzho027_cifar10", "--results-root", "/workspace/output"]

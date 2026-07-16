# Running Auralis with systemd

The repository includes `auralis-enhanced.service`, tailored to the current Linux host layout:

- repository: `/home/op/Auralis-Enhanced`;
- Conda environment: `/home/op/miniconda3/envs/auralis-enhanced`;
- API port: `6688`;
- default physical GPU: index `2` (the third GPU on a three-GPU host).

The previous unit selected GPU index `3`, which is outside the valid `0-2` range on a three-GPU system. It also hard-coded pip-package CUDA library paths and used a benchmark port different from the service port. Those assumptions have been removed.

## Install

```bash
sudo cp auralis-enhanced.service /etc/systemd/system/auralis-enhanced.service
sudo systemctl daemon-reload
sudo systemctl enable --now auralis-enhanced
systemctl status auralis-enhanced --no-pager
curl --fail http://127.0.0.1:6688/health
```

## Override configuration

Create `/etc/default/auralis-enhanced` to override environment variables without editing the unit:

```bash
CUDA_VISIBLE_DEVICES=2
HF_HOME=/home/op/.cache/huggingface
```

After changing it:

```bash
sudo systemctl restart auralis-enhanced
```

For another machine, edit `User`, `WorkingDirectory`, `ExecStartPre`, and `ExecStart` before installing the unit. Do not restore a manually assembled `LD_LIBRARY_PATH`; launch the entry point from the environment where Auralis and its CUDA dependencies are installed.

# DeepSpell
Deep Learning based Speller.
Based on Tall Weiss's DeepSpell code https://github.com/MajorTal/DeepSpell.

## Quick Start
Clone spellie's Git repository:

```git clone https://github.com/kevcisme/spellie.git```

### CPU
* Install [Docker](https://www.docker.com/)
* Run `./build.sh`
* Run `docker run --name=deepspell-cpu -it deepspell-cpu`

### GPU
Requires CUDA-compatible graphics card.

* Install [NVIDIA docker](https://www.docker.com/)
* Run `./build.sh gpu`
* Run `nvidia-docker run --name=deepspell-gpu -it deepspell-gpu`

## Documentation
[Deep Spelling](https://medium.com/@majortal/deep-spelling-9ffef96a24f6#.2c9pu8nlm) by Tal Weiss

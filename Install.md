# Preliminary Work
- `git clone https://github.com/JiaChen-Lee/binlang-v3_1.git`
- `cd binlang-v3_1`
- `conda create -n your-env-name python=3.7`
- `conda activate your-env-name`
- `pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html`
- `pip install -r requirements.txt`
- `pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git`

# Run
- `python main_old.py`
# RiboNucleic Acid Language Model - RiNALMo

### [Paper](https://arxiv.org/abs/2403.00043) | [Weights](https://zenodo.org/records/15043668)

[Rafael Josip Penić](https://www.fer.unizg.hr/en/rafael_josip.penic)<sup>1</sup>,
[Tin Vlašić](https://sites.google.com/view/tinvlasic)<sup>2</sup>,
[Roland G. Huber](https://web.bii.a-star.edu.sg/~rghuber/index.html)<sup>3</sup>,
[Yue Wan](https://www.a-star.edu.sg/gis/our-people/faculty-staff/members/yue-wan)<sup>2</sup>,
[Mile Šikić](https://www.a-star.edu.sg/gis/our-people/faculty-staff/members/mile-sikic)<sup>2</sup>
<br>
<sup>1</sup>Faculty of Electrical Engineering and Computing, University of Zagreb, Croatia <br>
<sup>2</sup>Genome Institute of Singapore (GIS), Agency for Science, Technology and Research (A\*STAR), Singapore <br>
<sup>3</sup>Bioinformatics Institute (BII), Agency for Science, Technology and Research (A\*STAR), Singapore

This is the official implementation of the paper "RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks".

## About
Ribonucleic acid (RNA) plays a variety of crucial roles in fundamental biological processes. Recently, RNA has become an interesting drug target, emphasizing the need to improve our understanding of its structures and functions. Over the years, sequencing technologies have produced an enormous amount of unlabeled RNA data, which hides important knowledge and potential. Motivated by the successes of protein language models, we introduce RiboNucleic Acid Language Model (RiNALMo) to help unveil the hidden code of RNA. RiNALMo is the largest RNA language model to date with 650 million parameters pre-trained on 36 million non-coding RNA sequences from several available databases. RiNALMo is able to extract hidden knowledge and capture the underlying structure information implicitly embedded within the RNA sequences. RiNALMo achieves state-of-the-art results on several downstream tasks. Notably, we show that its generalization capabilities can overcome the inability of other deep learning methods for secondary structure prediction to generalize on unseen RNA families.

 <img src="./imgs/rinalmo_3.png" width="1000">

## Quick Start - Inference

**Device selection:** inference code picks an accelerator in this order: **CUDA** (NVIDIA GPU), then **MPS** (Apple GPU / Metal), then **CPU**. This is implemented in `rinalmo.inference_device.select_inference_device()` and used by `get_pretrained_model()` when you do not pass a device.

### macOS

Prerequisites: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Mambaforge](https://github.com/conda-forge/miniforge), Python ≥ 3.8.1 via the env file (matches `requires-python` in [`pyproject.toml`](./pyproject.toml)).

```bash
git clone https://github.com/lbcb-sci/RiNALMo
cd RiNALMo
conda env create -f environment.macos.yml
conda activate rinalmo
pip install -e .
python test.py
```

`test.py` runs a short forward pass using the automatically chosen device. On first run, pretrained weights for `giga-v1` are downloaded (via `gdown`) into `~/.cache/rinalmo_pretrained/` if they are not already present.

### Linux (CUDA)

For a minimal pip install (Prerequisites: **Python ≥ 3.8.1**, **CUDA ≥ 11.8** compatible with your PyTorch build):

```bash
git clone https://github.com/lbcb-sci/RiNALMo
cd RiNALMo
pip install .
pip install flash-attn==2.3.2
python test.py
```

FlashAttention is used on CUDA; on CPU or MPS the model falls back to standard attention. For a conda environment oriented toward CUDA and training dependencies, use `environment.linux.yml` (see comments at the top of that file).

After installation you can use RiNALMo from Python; this matches what `test.py` does:

```python
import torch
from rinalmo.pretrained import get_pretrained_model
from rinalmo.inference_device import select_inference_device, inference_autocast

DEVICE = select_inference_device()
model, alphabet = get_pretrained_model(model_name="giga-v1", device=DEVICE)
model.eval()
seqs = ["ACUUUGGCCA", "CCCGGU"]

tokens = torch.tensor(alphabet.batch_tokenize(seqs), dtype=torch.int64, device=DEVICE)
with torch.no_grad(), inference_autocast(DEVICE):
    outputs = model(tokens)

print(outputs["representation"])
```

## Installation
1. Clone the repo.
```bash
git clone https://github.com/lbcb-sci/RiNALMo
cd RiNALMo
```
2. Create a conda environment (choose one file for your OS):

**macOS** — dependencies for Apple Silicon / CPU inference and development:

```bash
conda env create -f environment.macos.yml
conda activate rinalmo
pip install -e .
```

**Linux** — CUDA-oriented stack (see `environment.linux.yml` for scope and any caveats):

```bash
conda env create -f environment.linux.yml
conda activate rinalmo
pip install -e .
```

For quick inference with `test.py` or `get_pretrained_model`, you can skip steps 3–4: base weights are downloaded on demand to `~/.cache/rinalmo_pretrained/`. Download into `./weights/` when using the training/evaluation scripts below.

3. Download pre-trained weights.
```bash
mkdir weights
cd weights
wget https://zenodo.org/records/15043668/files/rinalmo_giga_pretrained.pt # 650M params
wget https://zenodo.org/records/15043668/files/rinalmo_mega_pretrained.pt # 150M params
wget https://zenodo.org/records/15043668/files/rinalmo_micro_pretrained.pt # 35M params
```   
4. Download fine-tuned weights.
```bash
# Download fine-tuned weights for secondary structure prediction.
wget https://zenodo.org/records/15043668/files/rinalmo_giga_ss_archiveII-16s_ft.pt
wget https://zenodo.org/records/15043668/files/rinalmo_giga_ss_archiveII-23s_ft.pt
wget https://zenodo.org/records/15043668/files/rinalmo_giga_ss_archiveII-5s_ft.pt
wget https://zenodo.org/records/15043668/files/rinalmo_giga_ss_archiveII-srp_ft.pt
wget https://zenodo.org/records/15043668/files/rinalmo_giga_ss_archiveII-grp1_ft.pt
wget https://zenodo.org/records/15043668/files/rinalmo_giga_ss_archiveII-telomerase_ft.pt
wget https://zenodo.org/records/15043668/files/rinalmo_giga_ss_archiveII-tmRNA_ft.pt
wget https://zenodo.org/records/15043668/files/rinalmo_giga_ss_archiveII-tRNA_ft.pt
wget https://zenodo.org/records/15043668/files/rinalmo_giga_ss_archiveII-RNaseP_ft.pt
wget https://zenodo.org/records/15043668/files/rinalmo_giga_ss_bprna_ft.pt

# Download fine-tuned weights for splice-site prediction.
wget https://zenodo.org/records/15043668/files/rinalmo_giga_splice_acceptor_ft.pt
wget https://zenodo.org/records/15043668/files/rinalmo_giga_splice_donor_ft.pt

# Download fine-tuned weights for mean ribosome loading prediction.
wget https://zenodo.org/records/15043668/files/rinalmo_giga_mrl_ft.pt

# Download fine-tuned weights for ncRNA functional family classification.
wget https://zenodo.org/records/15043668/files/rinalmo_giga_ncrna_class_0_noise_ft.pt
wget https://zenodo.org/records/15043668/files/rinalmo_giga_ncrna_class_200_noise_ft.pt

cd ..
```

## Development

Optional extras in [`pyproject.toml`](./pyproject.toml) (`[project.optional-dependencies]` → `dev`) install **autopep8** and **flake8**. Use them after an editable install of the package:

```bash
pip install -e ".[dev]"
```

That selects the PEP 621 `dev` extra (see `Provides-Extra` / `Requires-Dist` … `extra == "dev"` on the built wheel). If you use **uv** with the repo lockfile, run `uv sync --extra dev` for the same packages.

Autopep8 options live under `[tool.autopep8]` in the same file.

## Usage
We provide pre-trained RiNALMo weights and fine-tuned weights for three downstream tasks: mean ribosome loading prediction, secondary structure prediction and splice-site prediction.
For both evaluation and fine-tuning use ```train_<downstream_task>.py``` scripts.

### Evaluation
In order to evaluate the provided fine-tuned RiNALMo models and prediction heads, please run the scripts using the following input arguments:
```bash
# skip fine-tuning and run the evaluation on the test set
--test_only
# path to the '.pt' file containing fine-tuned model weights
--init_params
# dataset on which you would like to evaluate the fine-tuned model
--dataset
# download and prepare data (if needed)
--prepare_data
# Directory that will contain or already contains training, validation and test data
data_dir
# directory for all the output files
--output_dir
```
#### Example
To evaluate the fine-tuned RiNALMo model and prediction head on archiveII 5S rRNA test dataset for secondary structure prediction, use the ```rinalmo_giga_ss_archiveII-5s_ft.pt``` weights. Here, we provide an example run command.
```
python train_sec_struct_prediction.py ./ss_data --test_only --init_params ./weights/rinalmo_giga_ss_archiveII-5s_ft.pt --dataset archiveII_5s --prepare_data --output_dir ./outputs/archiveII/5s/ --accelerator gpu --devices 1
```

### Fine-tuning
In order to fine-tune RiNALMo, use ```--pretrained_rinalmo_weights ./weights/rinalmo_giga_pretrained.pt ``` input argument. Use ```--help``` to learn about other available arguments.

## License

Copyright 2024 Šikić Lab - AI in Genomics

### RiNALMo Code License
Licensed under the Apache License, Version 2.0 (the "[License](./LICENSE)");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

### Model Parameters License
The RiNALMo parameters are made available under the terms of the Creative Commons Attribution 4.0 International (CC BY 4.0) license. You can find details at: [https://creativecommons.org/licenses/by/4.0/legalcode](https://creativecommons.org/licenses/by/4.0/legalcode).

## Citation
If you find our work useful in your research, please cite:
```bibtex
@article{penic2024_rinalmo,
  title={RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks},
  author={Penić, Rafael Josip and Vlašić, Tin and Huber, Roland G. and Wan, Yue and Šikić, Mile},
  journal={arXiv preprint arXiv:2403.00043},
  year={2024}
}
```

## Contact
If you have any questions, please feel free to email the authors or open an issue.

## Acknowledgment
This work was supported in part by the National Research Foundation (NRF) Competitive Research Programme (CRP) under Project _Identifying Functional RNA Tertiary Structures in Dengue Virus_ (NRF-CRP27-2021RS-0001) and in part by the A\*STAR under Grant _GAP2: A\*STAR RNA-Foundation Model (A\*STAR RNA-FM)_ (I23D1AG079).

The computational work for the paper was partially performed on resources of the National Supercomputing Centre, Singapore [https://www.nscc.sg](https://www.nscc.sg).

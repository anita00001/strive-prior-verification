# Formal Verification of the STRIVE Learned Traffic Prior using α,β-CROWN verifier

This README gives a workflow for starting formal verification of the **STRIVE learned traffic prior** using **auto_LiRPA and α-CROWN**.

---

## 1. Workspace and Environment setup
### 1.1 Create a formal verification workspace
```bash
mkdir formal-verification-workspace
cd formal-verification-workspace
```
### 1.2 Use separate environments:

```text
STRIVE environment
    PyTorch 1.9
    torch-geometric
    torch-sparse
    nuScenes
    pretrained STRIVE checkpoint
        |
        | creates
        v
prior_head_capture.pt
        |
        v
Verification environment
    modern PyTorch
    auto_LiRPA
    α-CROWN
    optional full α,β-CROWN
```

---

#### 1.2.1 STRIVE Environment

Clone Original STRIVE
```bash
git clone https://github.com/nv-tlabs/STRIVE.git
cd STRIVE
```

Activate environment for STRIVE:

```bash
conda create -n strive python=3.8 -y
conda activate strive
```
Verify Python and pip versions
```bash
python --version
python -m pip --version
which python
which pip
```
If pip version >25
```bash
python -m pip install --upgrade pip==24.2
```
Install numpy
```bash
pip install numpy==1.19.5
```
Install other dependencies
```bash
pip install -r requirements.txt
```

Downloads
1. Download [nuScenes dataset](https://www.nuscenes.org/) (only metadata is required) and map expansion from Nuscenes database. Place the dataset in the directory called data.

        nuScenes layout

        ```text
        STRIVE/
        └── data/
            └── nuscenes/
                └── trainval/
                    ├── v1.0-trainval/
                    │   ├── scene.json
                    │   ├── sample.json
                    │   ├── sample_annotation.json
                    │   └── ...
                    └── maps/
                        └── expansion/
                            ├── boston-seaport.json
                            ├── singapore-hollandvillage.json
                            ├── singapore-onenorth.json
                            └── singapore-queenstown.json
        ```

Verify availability of the data

Run:
```bash
python - <<'PY'
from nuscenes.map_expansion.map_api import NuScenesMap

m = NuScenesMap(
    dataroot="./data/nuscenes/trainval",
    map_name="singapore-hollandvillage"
)

print("nuScenes map loaded successfully")
PY
```
This should print: nuScenes map loaded successfully

2. Download the pretrained traffic_model.pth from the original STRIVE README or [traffic_model.pth](https://www.dropbox.com/scl/fo/ajge853wnwgtrrxysuwim/APUPJ8UTs8mxGGNp0SHN0aw?rlkey=zfzkycbf7h1dx0a6mcj4etl1e&st=pk1tis1d&e=1&dl=0). Place traffic_model.pth in a directory called model_ckpt. We are not retraining the STRIVE.

3. Download generated scenarios for both adversarial and solution optimization from the original STRIVE README or [scenarios](https://www.dropbox.com/scl/fo/djjdbr4ykiwogvfdpuz0c/APryskA5DXsnRqWwvs6_-tM?rlkey=p5kvk9ns1bk6ejjnrjg7s3ndo&st=nueo2y07&e=1&dl=0) and place them in data/strive_scenarios

---

#### 1.2.2 Verifier Environment
```bash
cd $HOME/formal-verification-workspace
```
Install `uv` if it is not already installed.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

Clone α,β-CROWN with submodules:

```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git

cd alpha-beta-CROWN
```

Create/synchronize the environment:

```bash
uv sync
source .venv/bin/activate
```

Install auto_LiRPA inside the α,β-CROWN environment:

```bash
cd $HOME/formal-verification-workspace/alpha-beta-CROWN
git submodule status
```

Initialize/update all submodules:

```bash
git submodule update --init --recursive
```

```bash
uv pip install -e ./auto_LiRPA
```

Add the correct package parent directory to `PYTHONPATH`:

```bash
export PYTHONPATH="$HOME/formal-verification-workspace/alpha-beta-CROWN/auto_LiRPA:$PYTHONPATH"
```

#### 1.2.3 Prior Verification
Clone repository `strive-prior-verification` on the directory `formal-verification-workspace`
```bash
cd $HOME/formal-verification-workspace/
git clone https://github.com/anita00001/strive-prior-verification.git
```
---

### 1.3 Capture STRIVE internal activation

If the machine GPU is much newer than the original PyTorch 1.9/CUDA 11.1 stack, use CPU for this capture step:

```bash
CUDA_VISIBLE_DEVICES="" python src/test_traffic.py \
    --config ./configs/test_traffic.cfg \
    --ckpt ./model_ckpt/traffic_model.pth
```
This is sufficient because the goal is only to capture an internal activation.

---

## 2. Helper setup
### Add the capture helper

```bash
cd formal-verification-workspace
git clone strive-prior-verification
```

Add its directory to `PYTHONPATH`:

```bash
export PYTHONPATH="$HOME/formal-verification-workspace/strive-prior-verification:$PYTHONPATH"
```
---

## 3. Verify Stage A
### 3.1 Attach the capture hook

In STRIVE, open `src/test_traffic.py` and locate `model.eval()`
Immediately below `model.eval()`, before `with torch.no_grad():`, temporarily add:

```python
from capture_prior_head import attach_capture

attach_capture(
    model,
    "prior_head_capture.pt"
)
```
Then run:
```bash
CUDA_VISIBLE_DEVICES="" python src/test_traffic.py \
    --config ./configs/test_traffic.cfg \
    --ckpt ./model_ckpt/traffic_model.pth
```

The run should create:

```text
prior_head_capture.pt
```

---

### 3.2 Verify Prior output MLP only

```bash
cd ../formal-verification-workspace/strive-prior-verification

python verify_prior_head.py \
    --ckpt $HOME/formal-verification-workspace/STRIVE/model_ckpt/traffic_model.pth \
    --capture $HOME/formal-verification-workspace/STRIVE/prior_head_capture.pt \
    --epsilon 0.001 \
    --delta 0.05
```

Verify all 64 outputs of the output MLP

```bash
python verify_prior_head_all.py \
    --ckpt $HOME/formal-verification-workspace/STRIVE/model_ckpt/traffic_model.pth \
    --capture $HOME/formal-verification-workspace/STRIVE/prior_head_capture.pt \
    --epsilon 0.001 \
    --delta 0.05
```
---

## 4. Verify Stage B
```bash
conda activate strive

export PYTHONPATH="$HOME/formal-verification-workspace/strive-prior-verification:$PYTHONPATH"
```
In `src/test_traffic.py`, temporarily replace the old capture call with:
```bash
# from capture_prior_head import attach_capture

# attach_capture(model, "prior_head_capture.pt")

from capture_prior_stage_b import 

attach_stage_b_captureattach_stage_b_capture( model, "prior_stage_b_capture.pt")
```
Run:
```bash
cd $HOME/formal-verification-workspace/STRIVE

CUDA_VISIBLE_DEVICES="" python src/test_traffic.py \
    --config ./configs/test_traffic.cfg \
    --ckpt ./model_ckpt/traffic_model.pth
```
Run Stage B Verification
```bash
cd $HOME/formal-verification-workspace/alpha-beta-CROWN
source .venv/bin/activate

export PYTHONPATH="$HOME$/formal-verification-workspace/alpha-beta-CROWN/auto_LiRPA:$PYTHONPATH"
```

```bash
cd $HOME/formal-verification-workspace/strive-prior-verification
python verify_prior_stage_b.py \
    --ckpt $HOME/formal-verification-workspace/STRIVE/model_ckpt/traffic_model.pth \
    --capture $HOME/formal-verification-workspace/STRIVE/prior_stage_b_capture.pt \
    --epsilon 0.001 \
    --delta 0.05
```
---

## 5. Verify Stage C
In STRIVE `src/test_traffic.py` add capture code
```bash
# from capture_prior_head import attach_capture

# attach_capture(model, "prior_head_capture.pt")

# from capture_prior_stage_b import attach_stage_b_capture

# attach_stage_b_capture(model, "prior_stage_b_capture.pt")

from capture_prior_stage_c import attach_stage_c_capture

attach_stage_c_capture(model, "prior_stage_c_capture.pt")
```
Execute:
```bash
cd $HOME/formal-verification-workspace/STRIVE

export PYTHONPATH="$HOME/formal-verification-workspace/strive-prior-quickstart:$PYTHONPATH"

CUDA_VISIBLE_DEVICES="" ~/miniconda3/envs/strive/bin/python \
src/test_traffic.py \
    --config ./configs/test_traffic.cfg \
    --ckpt ./model_ckpt/traffic_model.pth
```

Run the stage-c verifier
```bash
cd $HOME/formal-verification-workspace/strive_prior_quickstart

export PYTHONPATH="HOME/formal-verification-workspace/alpha-beta-CROWN/auto_LiRPA:$PYTHONPATH"
```
```bash
cd $HOME/formal-verification-workspace/strive-prior-verification
python verify_prior_stage_c_all_agents.py \
    --ckpt $HOME/formal-verification-workspace/STRIVE/model_ckpt/traffic_model.pth \
    --capture $HOME/formal-verification-workspace/STRIVE/prior_stage_c_capture.pt \
    --epsilon 0.0008 \
    --delta 0.05
```
---

## 6. Verify Stage D
In STRIVE `src/test_traffic.py`:
```bash

    # from capture_prior_head import attach_capture

    # attach_capture(model, "prior_head_capture.pt")

    # from capture_prior_stage_b import attach_stage_b_capture

    # attach_stage_b_capture(model, "prior_stage_b_capture.pt")

    # from capture_prior_stage_c import attach_stage_c_capture
    
    # attach_stage_c_capture(model, "prior_stage_c_capture.pt")

    from capture_prior_stage_d import attach_stage_d_capture

    attach_stage_d_capture(model, "prior_stage_d_capture.pt")
```

Run:
```bash
cd $HOME/formal-verification-workspace/STRIVE

export PYTHONPATH="~/formal-verification-workspace/strive-prior-verification:$PYTHONPATH"

CUDA_VISIBLE_DEVICES="" \
~/envs/strive/bin/python \
src/test_traffic.py \
    --config ./configs/test_traffic.cfg \
    --ckpt ./model_ckpt/traffic_model.pth
```

### 6.1 Run verifier: D1
```bash
cd $HOME/formal-verification-workspace/strive-prior-verification

export PYTHONPATH="$HOME/formal-verification-workspace/alpha-beta-CROWN/auto_LiRPA:$PYTHONPATH"
```
```bash
python verify_prior_stage_d1_generalized_all_agents.py \
    --ckpt $HOME/formal-verification-workspace/STRIVE/model_ckpt/traffic_model.pth \
    --c-capture $HOME/formal-verification-workspace/STRIVE/prior_stage_c_capture.pt \
    --d1-all-capture $HOME/formal-verification-workspace/STRIVE/prior_stage_d1_all_capture.pt \
    --old-d-capture $HOME/formal-verification-workspace/STRIVE/prior_stage_d_capture.pt \
    --epsilon 0.001 \
    --delta 0.05
```

### 6.2 Run D2 verifier
```bash
cd $HOME/formal-verification-workspace/strive-prior-verification

export PYTHONPATH="$HOME/formal-verification-workspace/alpha-beta-CROWN/auto_LiRPA:$PYTHONPATH"
```
```bash  
python verify_prior_stage_d2_generalized_patch_late_crown.py \
    --ckpt $HOME/formal-verification-workspace/STRIVE/model_ckpt/traffic_model.pth \
    --capture $HOME/formal-verification-workspace/STRIVE/prior_stage_d_capture.pt \
    --epsilon 0.001 \
    --delta 0.05 \
    --row 127 --col 127 --patch-size 3 \
    --epsilon 0.0001 --delta 0.05 \
    --max-depth 0 --max-nodes 1 --threads 8 \
    --map-spec-batch 8 --selected-pair-count 64
```

## 7. Future extension
> This project can be extended to verify a same-scene joint raw-input property that activates the D1 past-state and D2 map uncertainties simultaneously. This would replace two separate Stage-D certificates with a stronger combined statement while preserving the same careful distinction between what we establish and what we assume.
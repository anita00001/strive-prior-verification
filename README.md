# Formal Verification of the STRIVE Learned Traffic Prior using α,β-CROWN verifier

This README gives a workflow for starting formal verification of the **STRIVE learned traffic prior** using **auto_LiRPA and α-CROWN**.

---

## 1. Workspace and Environment Setup
STRIVE Environment: strive
alpha-beta-CROWN Environment: abcrown

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

Activate the environment for STRIVE:

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
Install Shapely
```bash
python -m pip install "shapely==1.8.5.post1"
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

Check Python availability.
```bash
which python
python --version
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
Add its directory to `PYTHONPATH`:
```bash
export PYTHONPATH="$HOME/formal-verification-workspace/strive-prior-verification:$PYTHONPATH"
```

#### 1.2.4 Create a shell script
In the `STRIVE` root, create `run_prior_capture.sh`
```bash
#!/bin/bash
set -e

cd ~/formal-verification-workspace/STRIVE

export PYTHONPATH="$PWD/../strive-prior-verification:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=""

python src/test_traffic.py \
    --config ./configs/test_traffic.cfg \
    --ckpt ./model_ckpt/traffic_model.pth
```
---

## 3. Capture Hook
1. Check environment: It should be `strive`
```bash
cd $HOME/formal-verification-workspace/STRIVE
conda activate strive
```

### 3.1 Attach the capture hook for Stage A
1. In STRIVE, open `src/test_traffic.py` and locate `model.eval()`
Import
```bash
from capture_prior_head import attach_capture
```
And:
Immediately below `model.eval()`, before `with torch.no_grad():`, temporarily add:
```bash
    attach_capture(
        model,
        os.path.expanduser(
            "~/formal-verification-workspace/strive-prior-verification/prior_head_capture.pt"
        )
    )
```
2. Run
```bash
bash run_prior_capture.sh
```
The run should create `prior_head_capture.pt` in the `strive-prior-verification` directory.

---

### 3.2 Attach the capture hook for Stage B
1. In STRIVE, open `src/test_traffic.py` and locate `model.eval()`
Import
```bash
# from capture_prior_head import attach_capture
from capture_prior_stage_b import attach_stage_b_capture
```
Immediately below `model.eval()`, before `with torch.no_grad():`, temporarily replace the old capture call with:
```bash
    # attach_capture(
    #     model,
    #     os.path.expanduser(
    #         "~/formal-verification-workspace/strive-prior-verification/prior_head_capture.pt"
    #     )
    # )

    attach_stage_b_capture(
        model,
        os.path.expanduser(
            "~/formal-verification-workspace/strive-prior-verification/stage_b_capture.pt"
        )
    )
```
2. Run
```bash
bash run_prior_capture.sh
```
The run should create `prior_stage_b_capture.pt` in the `strive-prior-verification` directory.

---

### 3.3 Attach the capture hook for Stage C
1. In STRIVE, open `src/test_traffic.py` and locate `model.eval()`
Import
```bash
# from capture_prior_head import attach_capture
# from capture_prior_stage_b import attach_stage_b_capture
from capture_prior_stage_c import attach_stage_c_capture
```
Immediately below `model.eval()`, before `with torch.no_grad():`, temporarily replace the old capture call with:

```bash
    # attach_capture(
    #     model,
    #     os.path.expanduser(
    #         "~/formal-verification-workspace/strive-prior-verification/prior_head_capture.pt"
    #     )
    # )

    # attach_stage_b_capture(
    #     model,
    #     os.path.expanduser(
    #         "~/formal-verification-workspace/strive-prior-verification/stage_b_capture.pt"
    #     )
    # )

    attach_stage_c_capture(
        model,
        os.path.expanduser(
             "~/formal-verification-workspace/strive-prior-verification/prior_stage_c_capture.pt"
        )
    )
```
2. Run
```bash
bash run_prior_capture.sh
```
The run should create `prior_stage_c_capture.pt` in the `strive-prior-verification` directory.

---

### 3.4 Attach the capture hook for Stage D
1. In STRIVE, open `src/test_traffic.py` and import
```bash
# from capture_prior_head import attach_capture
# from capture_prior_stage_b import attach_stage_b_capture
# from capture_prior_stage_c import attach_stage_c_capture
from capture_prior_stage_d import capture_stage_d
```

2. In STRIVE, open `src/test_traffic.py` and locate `run_one_epoch()`
Immediately below `pred = model(scene_graph, map_idx, map_env, use_post_mean=True)`,

And before `loss_dict = loss_fn(scene_graph, pred)`, temporarily add a capture call:
```bash
    capture_stage_d(
        model,
        scene_graph,
        map_idx,
        map_env,
        output_capture=os.path.expanduser(
            "~/formal-verification-workspace/strive-prior-verification/prior_stage_d_capture.pt"
        ),
    )
```

3. Immediately below `model.eval()`, before `with torch.no_grad():`, comment out all other capture calls:
```bash
    # from capture_prior_head import attach_capture

    # attach_capture(
    #     model,
    #     os.path.expanduser(
    #         "~/formal-verification-workspace/strive-prior-verification/prior_head_capture.pt"
    #     )
    # )

    # from capture_prior_stage_b import attach_stage_b_capture

    # attach_stage_b_capture(
    #     model,
    #     os.path.expanduser(
    #         "~/formal-verification-workspace/strive-prior-verification/stage_b_capture.pt"
    #     )
    # )

    # from capture_prior_stage_c import attach_stage_c_capture

    # attach_stage_c_capture(
    #     model,
    #     os.path.expanduser(
    #          "~/formal-verification-workspace/strive-prior-verification/prior_stage_c_capture.pt"
    #     )
    # )
```

4. Run
```bash
bash run_prior_capture.sh
```
The run should create `prior_stage_d_capture.pt` in the `strive-prior-verification` directory.

---
---

## 4. Verify STRIVE Traffic Prior: Interaction Network
Check environment: It should be `abcrown`.
```bash
cd $HOME/formal-verification-workspace/alpha-beta-CROWN
```
Deactivate all other environments
```bash
conda deactivate
```
Activate `abcrown` environment
```bash
source .venv/bin/activate
```
Change directory to `strive-prior-verification`
```bash
cd $HOME/formal-verification-workspace/strive-prior-verification

export PYTHONPATH="$HOME/formal-verification-workspace/alpha-beta-CROWN/auto_LiRPA:$HOME/formal-verification-workspace/alpha-beta-CROWN"
```
```bash
python -c "import auto_LiRPA; print(auto_LiRPA.__file__); from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm; print('SUCCESS')"
```
---

### 4.1 Stage A: Verify Prior output MLP only
1. Run verifier
```bash
cd $HOME/formal-verification-workspace/strive-prior-verification

python verify_prior_head.py \
    --ckpt $HOME/formal-verification-workspace/STRIVE/model_ckpt/traffic_model.pth \
    --capture $HOME/formal-verification-workspace/strive-prior-verification/prior_head_capture.pt \
    --epsilon 0.001 \
    --delta 0.05
```

2. Verify all 64 outputs of the output MLP
```bash
cd $HOME/formal-verification-workspace/strive-prior-verification

python verify_prior_head_all.py \
    --ckpt $HOME/formal-verification-workspace/STRIVE/model_ckpt/traffic_model.pth \
    --capture $HOME/formal-verification-workspace/strive-prior-verification/prior_head_capture.pt \
    --epsilon 0.001 \
    --delta 0.05
```
---

### 4.2 Verify Stage B
Run Stage B Verification
```bash
cd $HOME/formal-verification-workspace/strive-prior-verification
python verify_prior_stage_b.py \
    --ckpt $HOME/formal-verification-workspace/STRIVE/model_ckpt/traffic_model.pth \
    --capture $HOME/formal-verification-workspace/strive-prior-verification/prior_stage_b_capture.pt \
    --epsilon 0.001 \
    --delta 0.05
```
---

### 4.3 Verify Stage C
```bash
cd $HOME/formal-verification-workspace/strive-prior-verification
python verify_prior_stage_c_all_agents.py \
    --ckpt $HOME/formal-verification-workspace/STRIVE/model_ckpt/traffic_model.pth \
    --capture $HOME/formal-verification-workspace/strive-prior-verification/prior_stage_c_capture.pt \
    --epsilon 0.0008 \
    --delta 0.05
```
---

### 4.4 Verify Stage D
Run:
### 4.4.1 Run verifier: D1
```bash
python verify_prior_stage_d1_generalized_all_agents.py \
    --ckpt $HOME/formal-verification-workspace/STRIVE/model_ckpt/traffic_model.pth \
    --c-capture $HOME/formal-verification-workspace/strive-prior-verification/prior_stage_c_capture.pt \
    --d1-all-capture $HOME/formal-verification-workspace/strive-prior-verification/prior_stage_d1_all_capture.pt \
    --old-d-capture $HOME/formal-verification-workspace/strive-prior-verification/prior_stage_d_capture.pt \
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
    --capture $HOME/formal-verification-workspace/strive-prior-verification/prior_stage_d_capture.pt \
    --epsilon 0.001 \
    --delta 0.05 \
    --row 127 --col 127 --patch-size 3 \
    --epsilon 0.0001 --delta 0.05 \
    --max-depth 0 --max-nodes 1 --threads 8 \
    --map-spec-batch 8 --selected-pair-count 64
```

## 7. Future extension
> This project can be extended to verify a same-scene joint raw-input property that activates the D1 past-state and D2 map uncertainties simultaneously. This would replace two separate Stage-D certificates with a stronger combined statement while preserving the same careful distinction between what we establish and what we assume.

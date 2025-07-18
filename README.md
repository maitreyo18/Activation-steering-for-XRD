# Activation steering of VLMs for identifying XRD anomalies

# Installation
Environment Preparation
To prepare the environment for LLaVA-v1.5 and MiniGPT-4, you can run the following commands:
```bash
conda create --name astra python==3.10.14
conda activate astra
pip install -r requirements.txt
```

To prepare the environment for Qwen2-VL, please run the following commands:
```bash
conda create --name astra_qwen python==3.10.15
conda activate astra_qwen
pip install -r requirements_qwen.txt
```
# For Heatmap
Navigate to the ./Composite_steering_minigpt4 subdir.
Update these parameters:
```bash
    alpha_bg_ring = -0.08
    alpha_ice_ring = 0.35
    alpha_loop_scattering = -0.08
    alpha_strong_background = -0.08
    alpha_nonuniform_detector = -0.08
```
Inside test_steering_composite_no_benign.py
No_benign means that I removed the part where I get the vanilla response as well. So it is faster.

IMPORTANT NOTE: You might to need to provide the absolute path in ./eval_configs/minigpt4_eval.yaml for ckpt. 
```bash
#ckpt: '../Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/ckpts/pretrained_minigpt4.pth'
  ckpt: '/scratch/gilbreth/biswasm/ASTRA_updated/ckpts/pretrained_minigpt4.pth'
  #ckpt: 'ckpts/pretrained_minigpt4.pth'
```
Change path to your absolute path in the system.

To get benign responses I can generate them myself or you can use ./test_CAA_modified_minigpt4/model_neutral.py.
Each directory has corresponding .sh files as well for reference usage and batch submissions.


# Goals
Producing the heatmap using a "composite" steering vector

Or maybe trying to compose composite vectors for each anomaly such that for example composite v_bg_ring such that it shows bg_ring for bg_ring images and different for the others.

import subprocess
import os
import torch
import dill
commands = [
    "pip install --upgrade numpy==1.24.1 --user",
    "pip install --upgrade google-auth-oauthlib==0.7.0 --user",
    "pip install --upgrade typing-extensions==4.6.0 --user" ,
    "pip install --upgrade tensorboard==2.15 --user"
]

for cmd in commands:
    subprocess.run(cmd, shell=True)

# Check if (NVIDIA) GPU is available
assert torch.cuda.is_available, "CUDA gpu not available"

# change to your directory
WORK_DIR = "/users/eleves-a/2021/hangao.liang/nmmo/"
CKPT_DIR = WORK_DIR + "runs"

os.chdir(WORK_DIR+"baselines")

with open(WORK_DIR+'baselines/requirements_colab.txt', "w") as f:
  f.write("""
accelerate==0.21.0
bitsandbytes==0.41.1
dash==2.11.1
openelm
pandas
plotly==5.15.0
psutil==5.9.3
ray==2.6.1
scikit-learn==1.3.0
tensorboard==2.11.2
tiktoken==0.4.0
torch
transformers==4.31.0
wandb==0.13.7
  """)

### pip install nmmo, pufferlib, and baselines deps.
# Install nmmo env and pufferlib
os.system("pip install nmmo==2.0.3 pufferlib==0.4.3 --user")
os.system("pip show nmmo")  # should be 2.0.3
os.system("pip show pufferlib") # should be 0.4.3
# Create the work directory, download the baselines code

  
# Install libs to run the baselines
os.system("pip install -r requirements_colab.txt > /dev/null")


# If everything is correctly installed, this should run
# command = [
#     "python",
#     "train.py",
#     "--runs-dir",
#     CKPT_DIR,
#     "--local-mode",
#     "true",
#     "--train-num-steps=5_000"
# ]
# subprocess.run(command)


CURRICULUM_FILE = os.path.join(WORK_DIR, "baselines", "reinforcement_learning", "eval_task_with_embedding.pkl")

with open(CURRICULUM_FILE, 'rb') as f:
  curriculum = dill.load(f)

# # curriculum[2] is a water-drinking task
# print(curriculum[2])


### Modify config.py to change training configurations
command = [
  "python",
  "train.py"
]
subprocess.run(command)

POLICY_DIR = os.path.join(WORK_DIR,'baselines','policies')
REPLAY_DIR = os.path.join(WORK_DIR,'baselines','replays')
# Generate a replay for the water-drinking task
# -t <CURRICULUM_FILE>
# -i <THE TASK INDEX>, in this case 2
command = [
    "python",
    "evaluate.py",
    "-p",
    POLICY_DIR,
    "-s",
    REPLAY_DIR,
    "-t",
    CURRICULUM_FILE,
    "-i",
    "2"
]
subprocess.run(command)

# replay mode
command = [
    "python",
    "evaluate.py",
    "-p",
    "policies",
    "-r",
]
subprocess.run(command)


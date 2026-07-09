import os
import sys
import json
import base64
from pathlib import Path


NOTEBOOK_TEMPLATE = r"""
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {"id": "title"},
   "source": ["# AIOS Trainer — Colab Launch\n", "Auto-generated notebook for fine-tuning LLMs on Google Colab.\n"]
  },
  {
   "cell_type": "code",
   "metadata": {"id": "setup"},
   "source": [
    "#@title 1. Mount Google Drive & Install Dependencies\n",
    "import os, sys, subprocess, importlib\n",
    "\n",
    "# Mount Drive\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "# Install required packages\n",
    "reqs = ['transformers', 'datasets', 'accelerate', 'peft', 'bitsandbytes', 'trl', 'torch']\n",
    "for pkg in reqs:\n",
    "    try:\n",
    "        importlib.import_module(pkg)\n",
    "    except ImportError:\n",
    "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])\n",
    "\n",
    "print('Dependencies installed.')\n",
    "\n",
    "# Copy project files if not present\n",
    "project_dir = '/content/aios-trainer'\n",
    "if not os.path.exists(project_dir):\n",
    "    subprocess.check_call([\n",
    "        'git', 'clone', 'https://github.com/koech-kipchirchir/koech-kipchirchir.github.io.git',\n",
    "        project_dir\n",
    "    ])\n",
    "os.chdir(project_dir)\n",
    "sys.path.insert(0, project_dir)\n",
    "print(f'Working directory: {os.getcwd()}')"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {"id": "config"},
   "source": [
    "#@title 2. Configure Training\n",
    "MODEL_NAME = 'microsoft/phi-2'  #@param {type:\"string\"}\n",
    "DATASET_NAME = 'HuggingFaceH4/ultrachat_200k'  #@param {type:\"string\"}\n",
    "PEFT_METHOD = 'qlora'  #@param ['lora', 'qlora']\n",
    "BATCH_SIZE = 4  #@param {type:\"integer\"}\n",
    "NUM_EPOCHS = 3  #@param {type:\"integer\"}\n",
    "LEARNING_RATE = 2e-4  #@param {type:\"number\"}\n",
    "MAX_SEQ_LENGTH = 2048  #@param {type:\"integer\"}\n",
    "USE_FP16 = False  #@param {type:\"boolean\"}\n",
    "USE_BF16 = True  #@param {type:\"boolean\"}\n",
    "GRADIENT_CHECKPOINTING = True  #@param {type:\"boolean\"}\n",
    "GRADIENT_ACCUMULATION = 4  #@param {type:\"integer\"}\n",
    "SAVE_EVERY_N_STEPS = 500  #@param {type:\"integer\"}\n",
    "EVAL_EVERY_N_STEPS = 500  #@param {type:\"integer\"}\n",
    "WANDB_API_KEY = ''  #@param {type:\"string\"}\n",
    "\n",
    "if WANDB_API_KEY:\n",
    "    os.environ['WANDB_API_KEY'] = WANDB_API_KEY\n",
    "\n",
    "from training.config import TrainingConfig\n",
    "\n",
    "config = TrainingConfig(\n",
    "    model_name=MODEL_NAME,\n",
    "    dataset_name=DATASET_NAME,\n",
    "    output_dir='/content/drive/MyDrive/aios-training',\n",
    "    run_name='colab-run',\n",
    "    num_train_epochs=NUM_EPOCHS,\n",
    "    per_device_train_batch_size=BATCH_SIZE,\n",
    "    learning_rate=LEARNING_RATE,\n",
    "    max_seq_length=MAX_SEQ_LENGTH,\n",
    "    use_peft=True,\n",
    "    peft_method=PEFT_METHOD,\n",
    "    fp16=USE_FP16,\n",
    "    bf16=USE_BF16,\n",
    "    gradient_checkpointing=GRADIENT_CHECKPOINTING,\n",
    "    gradient_accumulation_steps=GRADIENT_ACCUMULATION,\n",
    "    save_steps=SAVE_EVERY_N_STEPS,\n",
    "    eval_steps=EVAL_EVERY_N_STEPS,\n",
    "    report_to=['tensorboard'],\n",
    "    save_total_limit=2,\n",
    "    trust_remote_code=True,\n",
    ")\n",
    "print('Configuration ready.')"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {"id": "train"},
   "source": [
    "#@title 3. Start Training\n",
    "from training.colab_trainer import ColabTrainer\n",
    "\n",
    "trainer = ColabTrainer(config, auto_setup=True)\n",
    "result = trainer.train(resume_last_checkpoint=True)\n",
    "\n",
    "print('Training complete!')\n",
    "print(json.dumps(result, indent=2))"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {"id": "eval"},
   "source": [
    "#@title 4. Evaluate Model\n",
    "from evaluation.chat import ChatEvaluator\n",
    "from evaluation.math import MathEvaluator\n",
    "from evaluation.reasoning import ReasoningEvaluator\n",
    "from evaluation.coding import CodingEvaluator\n",
    "from evaluation.report import ReportGenerator\n",
    "\n",
    "results = {}\n",
    "\n",
    "print('Evaluating chat...')\n",
    "chat_eval = ChatEvaluator(trainer.model, trainer.tokenizer)\n",
    "results['chat'] = chat_eval.evaluate()\n",
    "\n",
    "print('Evaluating math...')\n",
    "math_eval = MathEvaluator(trainer.model, trainer.tokenizer)\n",
    "results['math'] = math_eval.evaluate()\n",
    "\n",
    "print('Evaluating reasoning...')\n",
    "reason_eval = ReasoningEvaluator(trainer.model, trainer.tokenizer)\n",
    "results['reasoning'] = reason_eval.evaluate()\n",
    "\n",
    "print('Evaluating coding...')\n",
    "coding_eval = CodingEvaluator(trainer.model, trainer.tokenizer)\n",
    "results['coding'] = coding_eval.evaluate()\n",
    "\n",
    "report = ReportGenerator(output_dir='/content/drive/MyDrive/aios-training/eval_reports')\n",
    "paths = report.save_all(results, model_name=MODEL_NAME)\n",
    "print('Reports saved:', json.dumps(paths, indent=2))"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {"id": "save"},
   "source": [
    "#@title 5. Save & Export Final Model\n",
    "from training.finetune import FineTuner\n",
    "\n",
    "finetuner = FineTuner(config)\n",
    "finetuner.model = trainer.model\n",
    "finetuner.tokenizer = trainer.tokenizer\n",
    "\n",
    "save_path = finetuner.save_pretrained('/content/drive/MyDrive/aios-training/final_model')\n",
    "print(f'Model saved to {save_path}')\n",
    "\n",
    "# Upload checkpoints to Drive if not already there\n",
    "import shutil\n",
    "drive_ckpt = '/content/drive/MyDrive/aios-training/checkpoints'\n",
    "os.makedirs(drive_ckpt, exist_ok=True)\n",
    "if os.path.exists('outputs/checkpoints'):\n",
    "    for item in os.listdir('outputs/checkpoints'):\n",
    "        src = os.path.join('outputs/checkpoints', item)\n",
    "        dst = os.path.join(drive_ckpt, item)\n",
    "        if os.path.isdir(src) and not os.path.exists(dst):\n",
    "            shutil.copytree(src, dst)\n",
    "            print(f'Copied {item} to Drive')"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "provenance": [],
   "gpuType": "T4"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
"""


def generate_colab_notebook(output_path: str = "aios_trainer_colab.ipynb") -> str:
    output_path = os.path.abspath(output_path)
    with open(output_path, "w") as f:
        f.write(NOTEBOOK_TEMPLATE)
    print(f"Colab notebook generated: {output_path}")
    print(f"Open at: https://colab.research.google.com/github/koech-kipchirchir/koech-kipchirchir.github.io/blob/main/{output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate AIOS Trainer Colab notebook")
    parser.add_argument(
        "--output", "-o",
        default="aios_trainer_colab.ipynb",
        help="Output notebook path",
    )
    args = parser.parse_args()
    generate_colab_notebook(args.output)

# Use an official PyTorch base image with CUDA support
FROM pytorch/pytorch:latest

# Set a working directory inside the container
WORKDIR /project

# Install git
RUN apt-get update && apt-get install -y git

RUN pip3 install fschat==0.2.23

RUN pip3 install pandas seaborn matplotlib numpy datasets evaluate jailbreakbench

RUN pip install anthropic google-generativeai openai litellm wandb

#补充：安装liteLLm and jailbreakbench wandband 修改依赖模块位置，如下已更改。
#sed -i 's/from litellm.llms.prompt_templates.factory import custom_prompt/from litellm.litellm_core_utils.prompt_templates.factory import custom_prompt/' litellm.py


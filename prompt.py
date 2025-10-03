import yaml
with open("prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

SLM_PROMPT_1 = prompts["classifier_prompt_1"]
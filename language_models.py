import os 
import litellm
from config import TOGETHER_MODEL_NAMES, LITELLM_TEMPLATES, API_KEY_NAMES, Model
from loggers import logger
from common import get_api_key
#from litellm import APIError

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = Model(model_name)
    
    def batched_generate(self, prompts_list: list, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
    
class APILiteLLM(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "ERROR: API CALL FAILED."
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name):
        super().__init__(model_name)
        self.api_key = get_api_key(self.model_name)
        self.litellm_model_name = self.get_litellm_model_name(self.model_name)
        litellm.drop_params=True
        self.set_eos_tokens(self.model_name)
        
    def get_litellm_model_name(self, model_name):
        if model_name in TOGETHER_MODEL_NAMES:
            litellm_name = TOGETHER_MODEL_NAMES[model_name]
            self.use_open_source_model = True
        else:
            self.use_open_source_model =  False
            #if self.use_open_source_model:
                # Output warning, there should be a TogetherAI model name
                #logger.warning(f"Warning: No TogetherAI model name for {model_name}.")
            litellm_name = model_name.value 
        return litellm_name
    
    def set_eos_tokens(self, model_name):
        if self.use_open_source_model:
            self.eos_tokens = LITELLM_TEMPLATES[model_name]["eos_tokens"]     
        else:
            self.eos_tokens = []

    def _update_prompt_template(self):
        # We manually add the post_message later if we want to seed the model response
        if self.model_name in LITELLM_TEMPLATES:
            litellm.register_prompt_template(
                initial_prompt_value=LITELLM_TEMPLATES[self.model_name]["initial_prompt_value"],
                model=self.litellm_model_name,
                roles=LITELLM_TEMPLATES[self.model_name]["roles"]
            )
            self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
        else:
            self.post_message = ""
        
    
    def batched_generate(self, convs_list: list[list[dict]], 
                         max_n_tokens: int, 
                         temperature: float, 
                         top_p: float,
                         extra_eos_tokens: list[str] = None) -> list[str]: 
        
        eos_tokens = list(self.eos_tokens)

        if extra_eos_tokens:
            eos_tokens.extend(extra_eos_tokens)
        if self.use_open_source_model:
            self._update_prompt_template()
            
        #outputs = litellm.batch_completion(
        kwargs = dict(
            model=self.litellm_model_name, 
            messages=convs_list,
            api_key=self.api_key,
            temperature=temperature,  #07/21
            top_p=top_p,   #01/08 removed
            #max_tokens=max_n_tokens,  #07/21
            max_completion_tokens=max_n_tokens,  #07/21
            num_retries=self.API_MAX_RETRY,
            seed=0,
            stop=eos_tokens,   #01/08 removed
        )
        try:
            outputs = litellm.batch_completion(**kwargs)
        except Exception as e:
        # 关键：失败也要返回同样长度
            err = f"{self.API_ERROR_OUTPUT} {type(e).__name__}: {e}"
            return [err for _ in range(len(convs_list))]
        responses: list[str] = []
        for i in range(len(convs_list)):
            try:
                out = outputs[i]
                if hasattr(out, "choices") and out.choices and hasattr(out.choices[0], "message"):
                    responses.append(out.choices[0].message.content or "")
                else:
                    # 关键：异常格式也要 append 占位
                    responses.append(f"{self.API_ERROR_OUTPUT} Unexpected output format: {out}")
            except Exception as e:
                # 关键：任何异常也要 append 占位
                responses.append(f"{self.API_ERROR_OUTPUT} Parse error: {type(e).__name__}: {e}")

        return responses
      
        '''
        responses = []
        for i, output in enumerate(outputs):
            try:
                # 如果是 liteLLM 的 ModelResponse 类型
                if hasattr(output, "choices") and hasattr(output.choices[0], "message"):
                    responses.append(output.choices[0].message.content)
                else:
                    print(f"[batched_generate] Unexpected output format at index {i}")
                    print(f"Prompt: {convs_list[i]}")
                    print(f"Output: {output}\n")
            except Exception as e:
                print(f"[batched_generate] Exception at index {i}")
                print(f"Prompt: {convs_list[i]}")
                print(f"Error: {e}\nOutput: {output}\n")
        return responses
       '''

    


from chatgpt_airsim.airsim_gpt_demo import evaluation, single_request,request_single_step

class AirSimModel(LanguageModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model_name = model_name


    def _generate_response(self, prompt: str) -> str:
        # 调用 airsim_gpt_demo.py 的逻辑
        code = request_single_step(prompt)
        print("lkjlkjlkjlkjlkjlkjlkjlkjlkjlllllllkkkkkkkkkkkkkkkkkkkkkkkkkkk")
        print(code)
        print("ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")
        print(evaluation(code))
        return code
    
    def batched_generate(self, convs_list: list[list[dict]], max_n_tokens: int, temperature: float, top_p: float, extra_eos_tokens: list[str] = None) -> list[str]:
        responses = []
        for conv in convs_list:
            user_input = conv[-1]["content"] if conv else ""
            response = self._generate_response(user_input)
            responses.append(response)
        return responses
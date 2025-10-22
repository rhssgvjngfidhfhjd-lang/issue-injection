from openai import OpenAI

client = OpenAI(api_key="sk-50qjfnJryCKT_Ku80l1c9w", base_url="https://litellm.eks-ans-se-dev.aws.automotive.cloud/")

def get_llm_responese(source_code, model, mode):
	"""Legacy function - use get_llm_response_simple instead."""
	if mode == "inst":
		user_content = "Please analyze the following code:\n" + source_code
	elif mode == "oneshot":
		user_content = "Please analyze the following code with examples:\n" + source_code
	elif mode == "cot":
		user_content = "Please analyze the following code step by step:\n" + source_code
	else:
		raise ValueError("Invalid mode")
	
	response = client.chat.completions.create(
		model=model,
		messages=[
			{"role": "system", "content": "You are an expert code analyst."},
			{"role": "user", "content": user_content},
		],
		temperature=0.0,
		stream=True
	)
	
	message = ""
	for chunk in response:
		if chunk.choices[0].delta.content is not None:
			message += chunk.choices[0].delta.content
	print("\n" + message)
	message = message.replace("<code>", "").replace("</code>", "")
	return message

def get_llm_response_simple(prompt, model="gpt-5"):
    """Simple function to get LLM response for EARS injection."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert automotive systems analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        stream=False,
        timeout=60
    )
    return response.choices[0].message.content.strip()

# Available models list
AVAILABLE_MODELS = ['deepseek.r1-v1:0', 'anthropic.claude-3-5-sonnet-20240620-v1:0', 'meta-llama/Llama-3.3-70B-Instruct', 'bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0', 'disabled-Qwen/Qwen3-235B-A22B', 'amazon.rerank-v1:0', 'amazon.titan-embed-text-v2:0', 'us.amazon.nova-pro-v1:0', 'us.meta.llama3-1-8b-instruct-v1:0', 'mistral.mistral-large-2402-v1:0', 'azure/gpt-35-turbo', 'text-embedding-3-small', 'gemini-2.5-pro', 'Qwen3-Embedding-8B', 'bedrock/us.meta.llama3-3-70b-instruct-v1:0', 'azure/o3-mini', 'paygo-azure/gpt-4o', 'grok-3', 'gemini-2.0-flash-001', 'vertex_ai/imagen-3.0-generate-002', 'text-multilingual-embedding-002', 'disabled-meta-llama/Llama-3.3-70B-Instruct', 'gpt-5-chat', 'azure/gpt-4o', 'gpt-4o-mini', 'amazon.nova-lite-v1:0', 'gpt-oss-20b', 'gpt-oss-120b', 'azure/gpt-5', 'gpt-5']
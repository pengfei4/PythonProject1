import os


#1.配置LLM客户端

API_KEY = " "
BASE_URL = " "
MODEL_ID = " "
TAVILY_API_KEY = " "
os.environ['TAVILY_API_KEY'] = " "

llm = OpenAICompatibleClient(
    model = MODEL_ID,
    api_key = API_KEY,
    base_url = BASE_URL
)
#2.初始化
user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求：[user_prompt"]

print(f"用户输入：{user_prompt}\n" + "=" * 40)

#3.运行主循环
for i in range(5):
    print(f"--- 循环 {i+1} ---")

    #3.1. 构建Prompt
    full_prompt = "\n".join(prompt_history)

    #3.2 调用LLM进行思考
    llm_output = llm.generate(full_prompt, system_prompt = AGENT_SYSTEM_PROMPT)
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)

    #3.3. 解析并执行行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        print("解析错误：模型输出中未能找到 Action。")
        break
    action_str = action_match.group(1).strip()

    if action_str.startwith("finish"):
        final_answer = re.search(r'finish\(answer="(.*)"\)', action_str).group(1)
        print(f"任务完成，最终答案：{final_answer}")
        break

        too_name = re.search(r"(\w+)\(", action_str).group(1)
        args_str = re.rearch(r"\((.*)\)", action_str).group(1)
        kwargs = dict(re.findall(r'(\w+) = "([^"]*)"', args_str))

        if tool_name in available_tools:
            observation = available_tools[too_name](**kwargs)
        else:
            observation = f"错误：未定义工具 '{tool_name}'"

        #3.4. 记录观察结果
        observation_str = f"observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
import os
from tavily import TavilyClient

def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    """
    #1. 从环境变量中读取API秘钥
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误：未配置TAVILY_API_KEY环境变量"

    #2. 初始化TAVILY客户端
    tavily = TavilyClient(api_key=api_key)

    #3. 构造一个精确的查询
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        #4. 调用API，include_answer = True 会返回一个综合性的回答
        response = tavily.search(query = query, search_depth="basic", include_answer = True)

        #5. Tavily返回的结果已经非常干净，可以直接使用
        #response['answer']是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]

        #如果没有综合性回答，则格式化原始结果
        formatted_result = []
        for result in response.get("result", []):
            formatted_result.append(f"- {result['title']}: {result['content']}")

        if not formatted_result:
            return "抱歉，没有找到相关的旅游景点推荐。"

        return "根据搜索，为您找到一下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误：执行Tavily搜索时出现问题 - {e}"
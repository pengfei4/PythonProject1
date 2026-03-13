#所以将工具函数放入一个字典，方便后续调用
from Weather_Free_Search import get_weather
from search_attraction import get_attraction

available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction
}
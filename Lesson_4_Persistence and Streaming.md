### Persistence and Streaming

In long running AI applications, there are 2 things are really important

1. Persistence - Persistence let you keep around the state of an agent at a particular point in time. This can let you go back to that state and resume in that state in future interactions.

2. Steaming - You can emit a list of signals of that is going on at an exact moment. So for long running applications, you know exactly what the agent is doing.

在 graph.compile()传入参数

```py
# Compile the graph with the checkpointer
self.graph = graph.compile(checkpointer=checkpointer)
```

实际使用, 要求包括在 with 里面，因为`SqliteSaver.from_conn_string` 是通过 `@contextmanager` 装饰器实现的，简化版本如下：

```py
from contextlib import contextmanager

@contextmanager
def from_conn_string(conn_str):
    conn = sqlite3.connect(conn_str)
    yield SqliteSaver(conn)
    conn.close()
```

这种实现方式下, `from_conn_string(...)` 返回的并不是一个 `SqliteSaver` 实例，而是一个 `generator-based context manager` 对象（类型是 `\_GeneratorContextManager`, 所以必须用 `with`，这是 唯一能正常拿到 `SqliteSaver` 实例的方式，因为 `yield` 出来的值才是真正的 `SqliteSaver` 对象。

```py
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model = ChatOpenAI(model="gpt-4o-mini")

with SqliteSaver.from_conn_string("checkpoints.sqlite") as memory:
    abot = Agent(model, [tool], system=prompt, checkpointer=memory)

    messages = [HumanMessage(content="What is the weather in Sydney?")]
    thread = {"configurable": {"thread_id": "1"}}

    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(v["messages"])


    messages = [HumanMessage(content="How about in Melbourne?")]
    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(v["messages"])

    messages = [HumanMessage(content="Which one is warmer?")]
    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(v["messages"])

    snapshot = abot.graph.get_state(thread)
    print(snapshot.values.keys())

    messages = snapshot.values["messages"]

    for i, msg in enumerate(messages):
        print(f"[{i}] {msg.type.upper()} - {msg.content}")

    messages = [HumanMessage(content="Which one is colder?")]
    thread = {"configurable": {"thread_id": "2"}} # change to a different thread_id
    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(v["messages"])
```

从上面的例子可以看到，同一个 thread_id 是共享 messages 的, 不同的 thread_id 则是不同的 messages

```
[AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_jEt9pyIHYDvT5rpngW9D5NMG', 'function': {'arguments': '{"query":"current weather in Sydney"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 151, 'total_tokens': 172, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': None, 'id': 'chatcmpl-Bvu1SBWZvPtRgVKgHDaLt83qurPCz', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--a702e9a5-2de8-40b0-981a-d67aae538475-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Sydney'}, 'id': 'call_jEt9pyIHYDvT5rpngW9D5NMG', 'type': 'tool_call'}], usage_metadata={'input_tokens': 151, 'output_tokens': 21, 'total_tokens': 172, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]
Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Sydney'}, 'id': 'call_jEt9pyIHYDvT5rpngW9D5NMG', 'type': 'tool_call'}
Back to the model!
[ToolMessage(content="[{'title': 'Sydney weather in July 2025 - Weather25.com', 'url': 'https://www.weather25.com/oceania/australia/new-south-wales/sydney?page=month&month=July', 'content': 'weather25.com\\nSearch\\nweather in Australia\\nRemove from your favorite locations\\nAdd to my locations\\nShare\\nweather in Australia\\n\\n# Sydney weather in July 2025\\n\\nClear\\nPatchy rain possible\\nClear\\nSunny\\nModerate rain\\nPatchy rain possible\\nClear\\nPatchy rain possible\\nModerate or heavy rain with thunder\\nLight rain\\nPatchy rain possible\\nPartly cloudy\\nClear\\nPartly cloudy\\n\\n## The average weather in Sydney in July [...] | 27 Patchy rain possible 15° /11° | 28 Sunny 15° /10° | 29 Patchy rain possible 16° /9° | 30 Moderate or heavy rain with thunder 13° /8° | 31 Light rain 13° /11° |  |  | [...] | Sun | Mon | Tue | Wed | Thu | Fri | Sat |\\n| --- | --- | --- | --- | --- | --- | --- |\\n|  |  | 1 Partly cloudy 16° /10° | 2 Light rain shower 18° /12° | 3 Light rain shower 16° /12° | 4 Light rain shower 15° /11° | 5 Light rain shower 16° /12° |\\n| 6 Cloudy 16° /12° | 7 Partly cloudy 16° /11° | 8 Sunny 17° /11° | 9 Sunny 16° /10° | 10 Patchy rain possible 16° /10° | 11 Sunny 17° /11° | 12 Sunny 16° /10° |', 'score': 0.8380581}, {'title': 'Sydney, NSW - Daily Weather Observations - Bureau of Meteorology', 'url': 'http://www.bom.gov.au/climate/dwo/IDCJDW2124.latest.shtml', 'content': 'IDCJDW2124.202507   Prepared at 05:36 UTC on Monday 21 July 2025\\n\\n## Source of data\\n\\nTemperature, humidity and rainfall observations are from Sydney (Observatory Hill) {station 066214}. Pressure, cloud, evaporation and sunshine observations are from Sydney Airport AMO {station 066037}. Wind observations are from Fort Denison {station 066022}.\\n\\nSydney Airport is about 10 km to the south of Observatory Hill.\\n\\nYou should read the important information in these notes.\\n\\n## Other formats [...] | 19 | Sa | 6.7 | 19.1 | 0.4 | 3.0 | 9.3 | W | 31 | 05:32 | 9.7 | 73 | 1 | W | 20 | 1019.7 | 18.8 | 34 | 1 | WNW | 7 | 1016.8 |\\n| 20 | Su | 6.8 | 18.7 | 0 | 4.0 | 9.3 | W | 35 | 02:51 | 9.8 | 61 | 1 | W | 24 | 1022.6 | 18.2 | 57 | 2 | SSW | 19 | 1021.3 |\\n| 21 | Mo | 9.7 |  | 0 | 1.8 |  |  |  |  | 11.2 | 90 | 4 | WNW | 20 | 1025.3 | 17.6 | 71 | 7 | ESE | 13 | 1021.5 |\\n| Statistics for the first 21 days of July 2025 | | | | | | | | | | | | | | | | | | | | | | [...] | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\\n| Mean | | 9.2 | 18.1 |  | 3.0 | 6.5 |  |  |  | 12.2 | 67 | 3 |  | 19 | 1016.3 | 17.0 | 52 | 3 |  | 17 | 1013.6 |\\n| Lowest | | 6.6 | 14.6 | 0 | 1.2 | 0.0 |  |  |  | 9.0 | 39 | 1 | W | 6 | 999.5 | 10.9 | 26 | 0 | NE | 2 | 1001.4 |\\n| Highest | | 13.1 | 21.9 | 73.0 | 6.8 | 9.3 | # | 83 |  | 15.6 | 90 | 8 | SW | 35 | 1025.3 | 21.3 | 95 | 8 | W | 44 | 1022.9 |', 'score': 0.8021998}]", name='tavily_search_results_json', tool_call_id='call_jEt9pyIHYDvT5rpngW9D5NMG')]
[AIMessage(content="The search for current weather information in Sydney did not yield the exact results. However, I recommend checking a reliable weather website or app for the most accurate and up-to-date forecast. Websites like the Bureau of Meteorology or weather.com can provide detailed information on temperature, humidity, and precipitation. \n\nIf you'd like, I can attempt another search for specific weather details.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 74, 'prompt_tokens': 1332, 'total_tokens': 1406, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': None, 'id': 'chatcmpl-Bvu1Zz52LhJZLUMbAmosEWqRCgnit', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='run--88d8deba-bfab-4851-991d-638abf59ea5e-0', usage_metadata={'input_tokens': 1332, 'output_tokens': 74, 'total_tokens': 1406, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]
[AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_vvnwXmmS7kY9iuCqPzjWaOZW', 'function': {'arguments': '{"query":"current weather in Melbourne"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 1418, 'total_tokens': 1439, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 1280}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': None, 'id': 'chatcmpl-Bvu1aiTF83xqFYwKrbuWvQOFz4REe', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--a4b9114c-f568-4d75-b4b8-4bf61a75cf3f-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Melbourne'}, 'id': 'call_vvnwXmmS7kY9iuCqPzjWaOZW', 'type': 'tool_call'}], usage_metadata={'input_tokens': 1418, 'output_tokens': 21, 'total_tokens': 1439, 'input_token_details': {'audio': 0, 'cache_read': 1280}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]
Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Melbourne'}, 'id': 'call_vvnwXmmS7kY9iuCqPzjWaOZW', 'type': 'tool_call'}
Back to the model!
[ToolMessage(content="[{'title': 'Melbourne, Victoria July 2025 Daily Weather Observations', 'url': 'http://www.bom.gov.au/climate/dwo/202507/html/IDCJDW3050.202507.shtml', 'content': '| Mean | | 7.1 | 15.0 |  | 1.8 | 4.5 |  |  |  | 9.4 | 81 | 5 |  | 10 | 1016.4 | 14.1 | 60 | 5 |  | 12 | 1014.1 |\\n| Lowest | | 4.3 | 11.0 | 0 | 0.0 | 0.0 |  |  |  | 6.4 | 61 | 0 | Calm | | 1000.8 | 10.6 | 37 | 1 | ENE | 4 | 997.8 |\\n| Highest | | 9.7 | 17.9 | 15.0 | 3.4 | 9.2 | N | 70 |  | 13.1 | 100 | 8 | N | 22 | 1025.7 | 17.6 | 100 | 7 | N | 24 | 1023.6 |\\n| Total | |  |  | 24.8 | 37.4 | 89.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | [...] | 1 | Tu | 5.7 | 13.8 | 1.8 | 0.0 | 0.9 | SSE | 35 | 12:24 | 11.0 | 96 | 8 | S | 9 | 1025.7 | 13.5 | 68 | 7 | S | 13 | 1022.7 |\\n| 2 | We | 9.7 | 11.0 | 0.4 | 1.2 | 0.0 | SSW | 28 | 04:20 | 10.0 | 85 | 7 | WSW | 7 | 1019.2 | 10.6 | 100 | 7 | S | 7 | 1018.0 |\\n| 3 | Th | 9.2 | 13.8 | 15.0 | 1.0 | 6.4 | SSW | 19 | 14:49 | 10.3 | 100 | 5 | SW | 6 | 1018.9 | 12.6 | 73 | 2 | SSW | 9 | 1017.2 | [...] | 13 | Su | 8.6 | 15.5 | 0 | 2.4 | 0.7 | NNW | 50 | 15:03 | 11.5 | 68 | 7 | N | 22 | 1016.3 | 15.1 | 54 | 7 | N | 24 | 1007.1 |\\n| 14 | Mo | 8.3 | 14.4 | 3.0 | 2.0 | 5.7 |  |  |  | 9.7 | 72 | 1 | NW | 9 | 1014.6 | 13.5 | 48 | 7 | W | 9 | 1011.8 |\\n| 15 | Tu | 8.7 | 13.7 |  | 2.6 | 3.0 | N | 28 | 01:03 | 9.5 | 72 | 7 | NNW | 11 | 1011.3 | 11.6 | 72 | 7 | SW | 6 | 1011.6 |\\n| 16 | We | 8.1 | 14.0 | 0 | 1.2 | 1.7 | N | 31 | 22:43 | 10.5 | 73 | 7 | NNW | 11 | 1017.1 | 12.9 | 70 | 7 | N | 11 | 1016.3 |', 'score': 0.89189607}, {'title': 'Melbourne weather in July 2025 - Weather25.com', 'url': 'https://www.weather25.com/oceania/australia/victoria/melbourne?page=month&month=July', 'content': '| 27 Light rain 10° /9° | 28 Overcast 13° /9° | 29 Light rain 10° /8° | 30 Patchy rain possible 11° /9° | 31 Patchy rain possible 12° /10° |  |  | [...] weather25.com\\nSearch\\nweather in Australia\\nRemove from your favorite locations\\nAdd to my locations\\nShare\\nweather in Australia\\n\\n# Melbourne weather in July 2025\\n\\nLight rain\\nLight rain shower\\nCloudy\\nPatchy rain possible\\nLight drizzle\\nLight rain\\nOvercast\\nLight rain\\nPatchy rain possible\\nPatchy rain possible\\nPatchy rain possible\\nLight rain\\nLight rain shower\\nPatchy rain possible\\n\\n## The average weather in Melbourne in July [...] Overcast\\nPatchy rain possible\\nOvercast\\nLight rain\\nPartly cloudy\\nPatchy rain possible\\nPatchy rain possible\\nPartly cloudy\\nPatchy rain possible\\nOvercast\\nPatchy rain possible\\nLight rain shower\\nLight rain shower\\nLight rain shower\\nLight rain shower\\nLight rain shower\\nLight rain shower\\nOvercast\\nOvercast\\nCloudy\\nLight rain shower\\nLight rain\\nLight rain shower\\nCloudy\\nPatchy rain possible\\nLight drizzle\\nLight rain\\nOvercast\\nLight rain\\nPatchy rain possible\\nPatchy rain possible', 'score': 0.8467682}]", name='tavily_search_results_json', tool_call_id='call_vvnwXmmS7kY9iuCqPzjWaOZW')]
[AIMessage(content="I wasn't able to find the current weather in Melbourne specifically, but based on recent observations, here are general trends for the weather in July:\n\n- **Temperature:** Typically ranges from around 8°C to 15°C.\n- **Conditions:** Often includes light rain, cloudy skies, and patchy rain.\n- **Precipitation:** Light rain showers are common.\n\nFor real-time weather details, I recommend visiting official weather websites like the Bureau of Meteorology or weather.com for the most accurate and current weather information. Would you like to search further or need information on something else?", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 117, 'prompt_tokens': 2744, 'total_tokens': 2861, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 1408}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': None, 'id': 'chatcmpl-Bvu1iS70atWhpRLberwcEWk6qiDci', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='run--01cf742d-bb23-4494-89f4-d4c41116da2c-0', usage_metadata={'input_tokens': 2744, 'output_tokens': 117, 'total_tokens': 2861, 'input_token_details': {'audio': 0, 'cache_read': 1408}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]
[AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_3Km6LqbdpqRMPHq5CcYaeb8Q', 'function': {'arguments': '{"query": "current temperature in Sydney"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}, {'id': 'call_dVii3fEBFU9p90n7PGoZRlXq', 'function': {'arguments': '{"query": "current temperature in Melbourne"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 58, 'prompt_tokens': 2873, 'total_tokens': 2931, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 2816}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': None, 'id': 'chatcmpl-Bvu1l3WaDZtXC7AacDDhvGuA5VNof', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--50cfdcc7-c911-46ee-a339-29dbaa9ca143-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current temperature in Sydney'}, 'id': 'call_3Km6LqbdpqRMPHq5CcYaeb8Q', 'type': 'tool_call'}, {'name': 'tavily_search_results_json', 'args': {'query': 'current temperature in Melbourne'}, 'id': 'call_dVii3fEBFU9p90n7PGoZRlXq', 'type': 'tool_call'}], usage_metadata={'input_tokens': 2873, 'output_tokens': 58, 'total_tokens': 2931, 'input_token_details': {'audio': 0, 'cache_read': 2816}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]
Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current temperature in Sydney'}, 'id': 'call_3Km6LqbdpqRMPHq5CcYaeb8Q', 'type': 'tool_call'}
Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current temperature in Melbourne'}, 'id': 'call_dVii3fEBFU9p90n7PGoZRlXq', 'type': 'tool_call'}
Back to the model!
[ToolMessage(content='[{\'title\': \'Hourly Weather-Sydney, New South Wales, Australia\', \'url\': \'https://weather.com/weather/today/l/98ef17e6662508c0af6d8bd04adacecde842fb533434fcd2c046730675fba371\', \'content\': "## Recent Locations\\n\\nMenu\\n\\n## Weather Forecasts\\n\\n## Radar & Maps\\n\\n## News & Media\\n\\n## Products & Account\\n\\n## Lifestyle\\n\\n### Specialty Forecasts\\n\\n# Sydney, New South Wales, Australia\\n\\n## Strong Wind Warning\\n\\n# Hourly Weather-Sydney, New South Wales, Australia\\n\\n## Now\\n\\nCloudy\\n\\n## 11 am\\n\\nCloudy\\n\\n## 12 pm\\n\\nCloudy\\n\\n## 1 pm\\n\\nCloudy\\n\\nChart small gif\\n\\n## Don\'t Miss\\n\\n## Seasonal Hub\\n\\n# 10 Day Weather-Sydney, New South Wales, Australia\\n\\n## Today\\n\\n## Day\\n\\nCloudy. High 66F. Winds NNE at 10 to 20 mph. [...] ## Night\\n\\nOvercast with rain showers at times. Low around 55F. Winds N at 10 to 20 mph. Chance of rain 60%.\\n\\n## Wed 23\\n\\n## Day\\n\\nIntervals of clouds and sunshine. High 64F. Winds W at 10 to 15 mph.\\n\\n## Night\\n\\nClear skies. Low near 45F. Winds W at 10 to 15 mph.\\n\\n## Thu 24\\n\\n## Day\\n\\nA mainly sunny sky. High 61F. Winds SW at 10 to 15 mph.\\n\\n## Night\\n\\nMostly clear. Low 41F. Winds WNW at 5 to 10 mph.\\n\\n## Fri 25\\n\\n## Day\\n\\nSunshine and clouds mixed. High 62F. NW winds shifting to NE at 10 to 15 mph. [...] ## Night\\n\\nCloudy with light rain developing after midnight. Low 53F. Winds NNE at 10 to 15 mph. Chance of rain 70%.\\n\\n## Radar\\n\\n## Trending Now\\n\\n## We Love Our Critters\\n\\n## Summer And Your Skin\\n\\n## Home, Garage & Garden\\n\\n## Through The Wildest Weather\\n\\n## Keeping You Healthy\\n\\n## Product Reviews & Deals\\n\\nundefined\\n\\nStay Cool And Save: This Popular 3-in-1 Mini Fan Is Only $16\\n\\nundefined\\n\\n17 Popular Sun Shirts For Men And Women\\n\\nundefined\\n\\nBest Sunscreen Of 2025: Our Top 10 Picks\\n\\nundefined", \'score\': 0.5515985}, {\'title\': \'Sydney, Australia 10-Day Weather Forecast\', \'url\': \'https://www.wunderground.com/forecast/au/sydney\', \'content\': \'# Sydney, New South Wales, Australia 10-Day Weather Forecaststar\\\\_ratehome\\n\\nicon\\n\\nThank you for reporting this station. We will review the data in question.\\n\\nYou are about to report this weather station for bad data. Please select the information that is incorrect.\\n\\nSee more\\n\\n(Reset Map)\\n\\nNo PWS\\n\\nReset Map, or Add PWS.\\n\\nicon\\nicon\\nicon\\nicon\\nicon\\nAccess Logo [...] We recognize our responsibility to use data and technology for good. We may use or share your data with our data vendors. Take control of your data.\\n\\nThe Weather Company Logo\\nThe Weather Channel Logo\\nWeather Underground Logo\\nStorm Radar Logo\\n\\n© The Weather Company, LLC 2025\', \'score\': 0.14082025}]', name='tavily_search_results_json', tool_call_id='call_3Km6LqbdpqRMPHq5CcYaeb8Q'), ToolMessage(content="[{'title': 'Melbourne, FL Weather Forecast - AccuWeather', 'url': 'https://www.accuweather.com/en/us/melbourne/32901/weather-forecast/332282', 'content': 'Hourly Weather · 1 AM 79°. rain drop 0% · 2 AM 79°. rain drop 1% · 3 AM 78°. rain drop 1%. 10-Day Weather', 'score': 0.6044228}, {'title': 'Melbourne, Victoria, Australia Monthly Weather - AccuWeather', 'url': 'https://www.accuweather.com/en/au/melbourne/26216/july-weather/26216', 'content': 'Get the monthly weather forecast for Melbourne, Victoria, Australia, including daily high/low, historical averages, to help you plan ahead.', 'score': 0.22093897}]", name='tavily_search_results_json', tool_call_id='call_dVii3fEBFU9p90n7PGoZRlXq')]
[AIMessage(content='The specific current temperatures for Sydney and Melbourne were not clearly retrieved, but based on general understanding:\n\n- Sydney typically experiences warmer temperatures, often around 15°C to 20°C during winter, depending on the time of day.\n- Melbourne, on the other hand, tends to be a bit cooler, averaging around 8°C to 15°C in winter.\n\nBased on these trends, Sydney is generally warmer than Melbourne. However, for accurate real-time temperatures, checking a weather app or website would be best. If you need more detailed analysis or specific data, just let me know!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 118, 'prompt_tokens': 4075, 'total_tokens': 4193, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 2816}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': None, 'id': 'chatcmpl-Bvu1w69BAVUqIBEPKTsgcjgVUzyHh', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='run--0a29161d-3134-4ead-b88c-289278338241-0', usage_metadata={'input_tokens': 4075, 'output_tokens': 118, 'total_tokens': 4193, 'input_token_details': {'audio': 0, 'cache_read': 2816}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]
dict_keys(['messages'])
[0] HUMAN - What is the weather in Sydney?
[1] AI -
[2] TOOL - [{'title': 'Sydney weather in July 2025 - Weather25.com', 'url': 'https://www.weather25.com/oceania/australia/new-south-wales/sydney?page=month&month=July', 'content': 'weather25.com\nSearch\nweather in Australia\nRemove from your favorite locations\nAdd to my locations\nShare\nweather in Australia\n\n# Sydney weather in July 2025\n\nClear\nPatchy rain possible\nClear\nSunny\nModerate rain\nPatchy rain possible\nClear\nPatchy rain possible\nModerate or heavy rain with thunder\nLight rain\nPatchy rain possible\nPartly cloudy\nClear\nPartly cloudy\n\n## The average weather in Sydney in July [...] | 27 Patchy rain possible 15° /11° | 28 Sunny 15° /10° | 29 Patchy rain possible 16° /9° | 30 Moderate or heavy rain with thunder 13° /8° | 31 Light rain 13° /11° |  |  | [...] | Sun | Mon | Tue | Wed | Thu | Fri | Sat |\n| --- | --- | --- | --- | --- | --- | --- |\n|  |  | 1 Partly cloudy 16° /10° | 2 Light rain shower 18° /12° | 3 Light rain shower 16° /12° | 4 Light rain shower 15° /11° | 5 Light rain shower 16° /12° |\n| 6 Cloudy 16° /12° | 7 Partly cloudy 16° /11° | 8 Sunny 17° /11° | 9 Sunny 16° /10° | 10 Patchy rain possible 16° /10° | 11 Sunny 17° /11° | 12 Sunny 16° /10° |', 'score': 0.8380581}, {'title': 'Sydney, NSW - Daily Weather Observations - Bureau of Meteorology', 'url': 'http://www.bom.gov.au/climate/dwo/IDCJDW2124.latest.shtml', 'content': 'IDCJDW2124.202507   Prepared at 05:36 UTC on Monday 21 July 2025\n\n## Source of data\n\nTemperature, humidity and rainfall observations are from Sydney (Observatory Hill) {station 066214}. Pressure, cloud, evaporation and sunshine observations are from Sydney Airport AMO {station 066037}. Wind observations are from Fort Denison {station 066022}.\n\nSydney Airport is about 10 km to the south of Observatory Hill.\n\nYou should read the important information in these notes.\n\n## Other formats [...] | 19 | Sa | 6.7 | 19.1 | 0.4 | 3.0 | 9.3 | W | 31 | 05:32 | 9.7 | 73 | 1 | W | 20 | 1019.7 | 18.8 | 34 | 1 | WNW | 7 | 1016.8 |\n| 20 | Su | 6.8 | 18.7 | 0 | 4.0 | 9.3 | W | 35 | 02:51 | 9.8 | 61 | 1 | W | 24 | 1022.6 | 18.2 | 57 | 2 | SSW | 19 | 1021.3 |\n| 21 | Mo | 9.7 |  | 0 | 1.8 |  |  |  |  | 11.2 | 90 | 4 | WNW | 20 | 1025.3 | 17.6 | 71 | 7 | ESE | 13 | 1021.5 |\n| Statistics for the first 21 days of July 2025 | | | | | | | | | | | | | | | | | | | | | | [...] | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n| Mean | | 9.2 | 18.1 |  | 3.0 | 6.5 |  |  |  | 12.2 | 67 | 3 |  | 19 | 1016.3 | 17.0 | 52 | 3 |  | 17 | 1013.6 |\n| Lowest | | 6.6 | 14.6 | 0 | 1.2 | 0.0 |  |  |  | 9.0 | 39 | 1 | W | 6 | 999.5 | 10.9 | 26 | 0 | NE | 2 | 1001.4 |\n| Highest | | 13.1 | 21.9 | 73.0 | 6.8 | 9.3 | # | 83 |  | 15.6 | 90 | 8 | SW | 35 | 1025.3 | 21.3 | 95 | 8 | W | 44 | 1022.9 |', 'score': 0.8021998}]
[3] AI - The search for current weather information in Sydney did not yield the exact results. However, I recommend checking a reliable weather website or app for the most accurate and up-to-date forecast. Websites like the Bureau of Meteorology or weather.com can provide detailed information on temperature, humidity, and precipitation.

If you'd like, I can attempt another search for specific weather details.
[4] HUMAN - How about in Melbourne?
[5] AI -
[6] TOOL - [{'title': 'Melbourne, Victoria July 2025 Daily Weather Observations', 'url': 'http://www.bom.gov.au/climate/dwo/202507/html/IDCJDW3050.202507.shtml', 'content': '| Mean | | 7.1 | 15.0 |  | 1.8 | 4.5 |  |  |  | 9.4 | 81 | 5 |  | 10 | 1016.4 | 14.1 | 60 | 5 |  | 12 | 1014.1 |\n| Lowest | | 4.3 | 11.0 | 0 | 0.0 | 0.0 |  |  |  | 6.4 | 61 | 0 | Calm | | 1000.8 | 10.6 | 37 | 1 | ENE | 4 | 997.8 |\n| Highest | | 9.7 | 17.9 | 15.0 | 3.4 | 9.2 | N | 70 |  | 13.1 | 100 | 8 | N | 22 | 1025.7 | 17.6 | 100 | 7 | N | 24 | 1023.6 |\n| Total | |  |  | 24.8 | 37.4 | 89.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | [...] | 1 | Tu | 5.7 | 13.8 | 1.8 | 0.0 | 0.9 | SSE | 35 | 12:24 | 11.0 | 96 | 8 | S | 9 | 1025.7 | 13.5 | 68 | 7 | S | 13 | 1022.7 |\n| 2 | We | 9.7 | 11.0 | 0.4 | 1.2 | 0.0 | SSW | 28 | 04:20 | 10.0 | 85 | 7 | WSW | 7 | 1019.2 | 10.6 | 100 | 7 | S | 7 | 1018.0 |\n| 3 | Th | 9.2 | 13.8 | 15.0 | 1.0 | 6.4 | SSW | 19 | 14:49 | 10.3 | 100 | 5 | SW | 6 | 1018.9 | 12.6 | 73 | 2 | SSW | 9 | 1017.2 | [...] | 13 | Su | 8.6 | 15.5 | 0 | 2.4 | 0.7 | NNW | 50 | 15:03 | 11.5 | 68 | 7 | N | 22 | 1016.3 | 15.1 | 54 | 7 | N | 24 | 1007.1 |\n| 14 | Mo | 8.3 | 14.4 | 3.0 | 2.0 | 5.7 |  |  |  | 9.7 | 72 | 1 | NW | 9 | 1014.6 | 13.5 | 48 | 7 | W | 9 | 1011.8 |\n| 15 | Tu | 8.7 | 13.7 |  | 2.6 | 3.0 | N | 28 | 01:03 | 9.5 | 72 | 7 | NNW | 11 | 1011.3 | 11.6 | 72 | 7 | SW | 6 | 1011.6 |\n| 16 | We | 8.1 | 14.0 | 0 | 1.2 | 1.7 | N | 31 | 22:43 | 10.5 | 73 | 7 | NNW | 11 | 1017.1 | 12.9 | 70 | 7 | N | 11 | 1016.3 |', 'score': 0.89189607}, {'title': 'Melbourne weather in July 2025 - Weather25.com', 'url': 'https://www.weather25.com/oceania/australia/victoria/melbourne?page=month&month=July', 'content': '| 27 Light rain 10° /9° | 28 Overcast 13° /9° | 29 Light rain 10° /8° | 30 Patchy rain possible 11° /9° | 31 Patchy rain possible 12° /10° |  |  | [...] weather25.com\nSearch\nweather in Australia\nRemove from your favorite locations\nAdd to my locations\nShare\nweather in Australia\n\n# Melbourne weather in July 2025\n\nLight rain\nLight rain shower\nCloudy\nPatchy rain possible\nLight drizzle\nLight rain\nOvercast\nLight rain\nPatchy rain possible\nPatchy rain possible\nPatchy rain possible\nLight rain\nLight rain shower\nPatchy rain possible\n\n## The average weather in Melbourne in July [...] Overcast\nPatchy rain possible\nOvercast\nLight rain\nPartly cloudy\nPatchy rain possible\nPatchy rain possible\nPartly cloudy\nPatchy rain possible\nOvercast\nPatchy rain possible\nLight rain shower\nLight rain shower\nLight rain shower\nLight rain shower\nLight rain shower\nLight rain shower\nOvercast\nOvercast\nCloudy\nLight rain shower\nLight rain\nLight rain shower\nCloudy\nPatchy rain possible\nLight drizzle\nLight rain\nOvercast\nLight rain\nPatchy rain possible\nPatchy rain possible', 'score': 0.8467682}]
[7] AI - I wasn't able to find the current weather in Melbourne specifically, but based on recent observations, here are general trends for the weather in July:

- **Temperature:** Typically ranges from around 8°C to 15°C.
- **Conditions:** Often includes light rain, cloudy skies, and patchy rain.
- **Precipitation:** Light rain showers are common.

For real-time weather details, I recommend visiting official weather websites like the Bureau of Meteorology or weather.com for the most accurate and current weather information. Would you like to search further or need information on something else?
[8] HUMAN - Which one is warmer?
[9] AI -
[10] TOOL - [{'title': 'Hourly Weather-Sydney, New South Wales, Australia', 'url': 'https://weather.com/weather/today/l/98ef17e6662508c0af6d8bd04adacecde842fb533434fcd2c046730675fba371', 'content': "## Recent Locations\n\nMenu\n\n## Weather Forecasts\n\n## Radar & Maps\n\n## News & Media\n\n## Products & Account\n\n## Lifestyle\n\n### Specialty Forecasts\n\n# Sydney, New South Wales, Australia\n\n## Strong Wind Warning\n\n# Hourly Weather-Sydney, New South Wales, Australia\n\n## Now\n\nCloudy\n\n## 11 am\n\nCloudy\n\n## 12 pm\n\nCloudy\n\n## 1 pm\n\nCloudy\n\nChart small gif\n\n## Don't Miss\n\n## Seasonal Hub\n\n# 10 Day Weather-Sydney, New South Wales, Australia\n\n## Today\n\n## Day\n\nCloudy. High 66F. Winds NNE at 10 to 20 mph. [...] ## Night\n\nOvercast with rain showers at times. Low around 55F. Winds N at 10 to 20 mph. Chance of rain 60%.\n\n## Wed 23\n\n## Day\n\nIntervals of clouds and sunshine. High 64F. Winds W at 10 to 15 mph.\n\n## Night\n\nClear skies. Low near 45F. Winds W at 10 to 15 mph.\n\n## Thu 24\n\n## Day\n\nA mainly sunny sky. High 61F. Winds SW at 10 to 15 mph.\n\n## Night\n\nMostly clear. Low 41F. Winds WNW at 5 to 10 mph.\n\n## Fri 25\n\n## Day\n\nSunshine and clouds mixed. High 62F. NW winds shifting to NE at 10 to 15 mph. [...] ## Night\n\nCloudy with light rain developing after midnight. Low 53F. Winds NNE at 10 to 15 mph. Chance of rain 70%.\n\n## Radar\n\n## Trending Now\n\n## We Love Our Critters\n\n## Summer And Your Skin\n\n## Home, Garage & Garden\n\n## Through The Wildest Weather\n\n## Keeping You Healthy\n\n## Product Reviews & Deals\n\nundefined\n\nStay Cool And Save: This Popular 3-in-1 Mini Fan Is Only $16\n\nundefined\n\n17 Popular Sun Shirts For Men And Women\n\nundefined\n\nBest Sunscreen Of 2025: Our Top 10 Picks\n\nundefined", 'score': 0.5515985}, {'title': 'Sydney, Australia 10-Day Weather Forecast', 'url': 'https://www.wunderground.com/forecast/au/sydney', 'content': '# Sydney, New South Wales, Australia 10-Day Weather Forecaststar\\_ratehome\n\nicon\n\nThank you for reporting this station. We will review the data in question.\n\nYou are about to report this weather station for bad data. Please select the information that is incorrect.\n\nSee more\n\n(Reset Map)\n\nNo PWS\n\nReset Map, or Add PWS.\n\nicon\nicon\nicon\nicon\nicon\nAccess Logo [...] We recognize our responsibility to use data and technology for good. We may use or share your data with our data vendors. Take control of your data.\n\nThe Weather Company Logo\nThe Weather Channel Logo\nWeather Underground Logo\nStorm Radar Logo\n\n© The Weather Company, LLC 2025', 'score': 0.14082025}]
[11] TOOL - [{'title': 'Melbourne, FL Weather Forecast - AccuWeather', 'url': 'https://www.accuweather.com/en/us/melbourne/32901/weather-forecast/332282', 'content': 'Hourly Weather · 1 AM 79°. rain drop 0% · 2 AM 79°. rain drop 1% · 3 AM 78°. rain drop 1%. 10-Day Weather', 'score': 0.6044228}, {'title': 'Melbourne, Victoria, Australia Monthly Weather - AccuWeather', 'url': 'https://www.accuweather.com/en/au/melbourne/26216/july-weather/26216', 'content': 'Get the monthly weather forecast for Melbourne, Victoria, Australia, including daily high/low, historical averages, to help you plan ahead.', 'score': 0.22093897}]
[12] AI - The specific current temperatures for Sydney and Melbourne were not clearly retrieved, but based on general understanding:

- Sydney typically experiences warmer temperatures, often around 15°C to 20°C during winter, depending on the time of day.
- Melbourne, on the other hand, tends to be a bit cooler, averaging around 8°C to 15°C in winter.

Based on these trends, Sydney is generally warmer than Melbourne. However, for accurate real-time temperatures, checking a weather app or website would be best. If you need more detailed analysis or specific data, just let me know!
[AIMessage(content='Could you please specify what you are comparing in terms of temperature? For example, are you asking about two different locations, objects, or time periods?', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 31, 'prompt_tokens': 149, 'total_tokens': 180, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': None, 'id': 'chatcmpl-Bvu1yLYX2QZicLBzUwmQVfHal1SPG', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='run--5d2dee1e-47a2-4384-96d3-2e6f88dff20d-0', usage_metadata={'input_tokens': 149, 'output_tokens': 31, 'total_tokens': 180, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]
```

同时注意 `snapshot` 在本例子只有一个 key, 即`messages`. 这是因为我们使用的`AgentState`里面只有一个`messages`property.

```py
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
```

```py
 snapshot = abot.graph.get_state(thread)
 print(snapshot.values.keys())

 messages = snapshot.values["messages"]
```

另外注意的一点是我们这里 invoke langgraph 的方法不是`invoke()`,而是`stream()`

```py
messages = [HumanMessage(content="Which one is warmer?")]
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v["messages"])
```

在 LangGraph 中，stream() 方法 和 invoke() 方法 都是用来执行图（graph）的入口点，但它们的行为和用途略有不同，适用于不同场景：

🔁 stream() 方法
特点：逐步返回执行过程中的中间状态

```py
for event in graph.stream(input, config):
    # event: a partial step/state/result emitted during execution
```

返回值是一个生成器（generator）, 每次迭代返回一个执行事件（event），通常是一个 node 的输出或状态快照（StateSnapshot）

适用于：

- 需要实时显示思考过程
- Agent 多轮交互（如 tool call、多个 steps）
- 构建聊天流、UI 渲染、观察每一步思考过程

📌 示例用途：

- 构建带工具调用的智能助手
- Agent 需要逐步决策、等待 tool 响应再继续

✅ invoke() 方法
特点：一次性执行完图并返回最终结果

```py
result = graph.invoke(input, config)
```

返回最终输出状态（如完整的 StateSnapshot, 不返回中间事件, 简单直接，一次性返回所有结果

适用于：

- 不关心中间过程
- 需要最终结果即可
- 批量处理、推理、测试等

📌 示例用途：

- 输入问题，直接返回回答
- 静默执行 graph 任务，不展示中间节点运行过程

🔍 对比表：
| 特性 | `invoke()` | `stream()` |
| -------- | ---------------------- | ------------------------ |
| 执行方式 | 一次性执行完返回结果 | 逐步返回每个执行步骤的中间状态 |
| 结果 | 最终的完整结果（StateSnapshot） | 每步的事件或状态更新（yield） |
| 适合用途 | 快速调用、API、简单流程 | 复杂 agent、多轮交互、需要展示决策过程 |
| 是否能中断/注入 | ❌ 不支持 | ✅ 支持中断和恢复（配合 checkpoint） |
| 是否支持工具调用 | ✅ 支持 | ✅ 支持并可观测每个 tool call |

✅ 总结推荐：
| 你想做的事 | 推荐方法 |
| ------------------ | ---------- |
| 只想拿到最终结果（像普通函数） | `invoke()` |
| 想显示或控制 agent 的思考过程 | `stream()` |
| Agent 可能调用多个工具 | `stream()` |
| 要支持中断和恢复 | `stream()` |

虽然名字相似，LangGraph 的 `stream()` 方法 和 OpenAI / LangChain / LangServe 等里的 `output token streaming`（流式输出） 是两个不同层面的“stream”。

✅ 一句话对比：
| 名称 | 作用层面 | 简述 |
| ----------------- | ---------------- | -------------------------------------------------------------------- |
| `graph.stream()` | LangGraph 的执行流程层 | ⏩ **逐步执行 graph**，每经过一个 node 或状态就 `yield` 一个事件（可以是 tool call、AI 思考过程） |
| `token streaming` | LLM 响应的 token 层 | ⌨️ **一个 AIMessage 的内容由模型按 token 一点点生成**，像 ChatGPT 的打字效果 |

✅ 图示对比（简化）：

```text
graph.stream()
├──> Step 1: HumanMessage
├──> Step 2: AIMessage (calls tool)
├──> Step 3: ToolMessage
├──> Step 4: AIMessage (final answer, streamed as tokens ↓)
                      ⮑ "The result is..." ← token-by-token stream (LLM streaming)

```

🧠 更详细解释：

1. graph.stream()（LangGraph）
   是对整个图流程执行的 事件级 stream

- 每一步 node 执行后，返回一个 StateSnapshot（状态快照）
- 用于捕捉：每个 message、tool 调用、agent 决策 等
- 典型用途：
  - 多轮对话
  - tool 调用过程监控
  - 实时 UI 更新 agent 的每一步思考

2. Token-level streaming（LLM output streaming）

   - 是对**单个模型响应内容**的**token**逐步输出
   - 在 LangChain 中，通常通过 streaming=True 启用
   - 用于：
     - 模拟 ChatGPT 打字
     - 降低延迟（边生成边显示）

例如在 LangChain 中：

```py
llm = ChatOpenAI(streaming=True)

for chunk in llm.stream("What is the weather?"):
print(chunk.content, end="", flush=True)
```

### 可以查看生成的 persistence 的 sqlite 表结构

```py
import sqlite3

conn = sqlite3.connect("checkpoints.sqlite")
cursor = conn.cursor()

import pandas as pd

def show_query_as_df(cursor, query):
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    return pd.DataFrame(rows, columns=columns)

# 查看 checkpoints 表结构
df_checkpoints_schema = show_query_as_df(cursor, "PRAGMA table_info(checkpoints);")
display(df_checkpoints_schema)

# 查看 writes 表结构
df_writes_schema = show_query_as_df(cursor, "PRAGMA table_info(writes);")
display(df_writes_schema)

# 查看 checkpoints 表数据
df_checkpoints_data = show_query_as_df(cursor, "SELECT * FROM checkpoints;")
display(df_checkpoints_data)

# 查看 writes 表数据
df_writes_data = show_query_as_df(cursor, "SELECT * FROM writes;")
display(df_writes_data)
```

默认生成两个表,`checkpoints`和`writers`

`checkpoints` 表结构
| `cid` | `name` | `type` | `notnull` | `dflt_value` | `pk` |
| ----- | ---------------------- | ------ | --------- | ------------ | ---- |
| 0 | thread_id | TEXT | 1 | None | 1 |
| 1 | checkpoint_ns | TEXT | 1 | '' | 2 |
| 2 | checkpoint_id | TEXT | 1 | None | 3 |
| 3 | parent_checkpoint_id | TEXT | 0 | None | 0 |
| 4 | type | TEXT | 0 | None | 0 |
| 5 | checkpoint | BLOB | 0 | None | 0 |
| 6 | metadata | BLOB | 0 | None | 0 |

`writers` 表结构
| `cid` | `name` | `type` | `notnull` | `dflt_value` | `pk` |
| ----- | -------------- | ------- | --------- | ------------ | ---- |
| 0 | thread_id | TEXT | 1 | None | 1 |
| 1 | checkpoint_ns | TEXT | 1 | '' | 2 |
| 2 | checkpoint_id | TEXT | 1 | None | 3 |
| 3 | task_id | TEXT | 1 | None | 4 |
| 4 | idx | INTEGER | 1 | None | 5 |
| 5 | channel | TEXT | 1 | None | 0 |
| 6 | type | TEXT | 0 | None | 0 |
| 7 | value | BLOB | 0 | None | 0 |

## Streaming

```py
messages = [HumanMessage(content="What is the weather in SF?")]
thread = {"configurable": {"thread_id": "4"}}

async def run_stream():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:  # ✅ 注意这里是 async with
        abot = Agent(model, [tool], system=prompt, checkpointer=memory)

        async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="|")

await run_stream()

```

`astream_events` 是 LangGraph 提供的异步生成器方法，用于 逐步、事件驱动地执行并流式返回 Graph 的中间过程。这非常适合用在需要边执行边展示的 AI 应用场景，例如：

- 显示模型生成的 token（chat streaming）
- 可视化工具调用过程
- 实时用户反馈等

🔍 工作原理解析：graph.astream_events(...)
📌 核心功能：
astream_events(input_state, config, version=...) 会：

1. 按图的结构运行节点（如你定义的 "llm", "action" 等）
2. 在每一步触发事件，比如：

   1. on_node_start：某个节点开始运行
   2. on_node_end：某个节点运行结束
   3. on_tool_start, on_tool_end
   4. on_chat_model_stream：模型生成中间 token（OpenAI 的 chunk）

以异步流（async generator）返回这些事件对象，你可以 async for 循环处理它们

🔁 和 invoke 或 stream 的区别
| 方法 | 是否异步 | 是否逐步返回事件 | 用途 |
| ------------------ | ---- | ---------- | ---------------------- |
| `invoke()` | ❌ 否 | ❌ 否 | 一次性执行整个 Graph，返回最终结果 |
| `stream()` | ❌ 否 | ✅ 是（返回值状态） | 同步地逐步运行，返回每个阶段的中间状态 |
| `astream_events()` | ✅ 是 | ✅ 是（返回事件） | 最细粒度的异步事件流，适合实时交互、前端展示 |

🧪 事件对象结构
每个事件是一个 `dict`，包含：

```py
{
    "event": "on_chat_model_stream",
    "data": {
        "chunk": AIMessageChunk(content="..."),  # 可用来拼接完整回复
        ...
    },
    ...
}
```

✅ 使用示例回顾

```py
async for event in abot.graph.astream_events(...):
    if event["event"] == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        print(content, end="|")
```

监听的是 `on_chat_model_stream` 事件，用于实时获取 LLM 的输出内容。

🧠 总结一句话
`astream_events` 就像是为你的 Agent graph 提供了“每一帧”的播放控制，它能让你可视化每个决策、每次工具调用、每个 token 生成，非常适合用于调试和构建可交互 AI 应用。

Output (displayed in chunk)

```
Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_w7CLUrHdBKvW6jpLaVU4Krx9', 'type': 'tool_call'}
Back to the model!
The| current| weather| in| San| Francisco| is| mainly| cloudy|,| with| a| high| of| around| |64|°F| (|approximately| |18|°C|)| during| the| day|.| The| temperatures| are| expected| to| drop| to| a| low| of| about| |56|°F| (|approximately| |13|°C|)| at| night|.| Winds| are| coming| from| the| west| at| |10| to| |20| mph|.

|For| more| details|,| you| can| check| the| comprehensive| weather| report| [|here|](|https|://|weather|.com|/weather|/t|oday|/l|/S|an|+|Franc|isco|+|CA|+|US|CA|098|7|:|1|:|US|).|
```

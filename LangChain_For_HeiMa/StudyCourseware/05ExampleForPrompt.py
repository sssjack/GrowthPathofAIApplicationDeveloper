from openai import OpenAI
import os

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    # api_key=os.getenv("DASHSCOPE_API_KEY"),
    # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",

    base_url="http://localhost:11434/v1",
)

examples_data = { '新闻报道':'今日，股市经历了一轮震荡，受到宏观经济数据和全球贸易紧张局势的影响。投资者密切关注美联储可能的政策调整，以适应市场的不确定性。',
                  '财务报告':'本公司年度财务报告显示，去年公司实现了稳步增长的盈利，同时资产负债表呈现强劲的状况。经济环境的稳定和管理层的有效战略执行为公司的健康.',
                  '公司公告':'本公司高兴地宣布成功完成最新一轮并购交易，收购了一家在人工智能领域领先的公司。这一战略举措将有助于扩大我们的业务领域，提高市场竞争力',
                  '分析师报告':'最新的行业分析报告指出，科技公司的创新将成为未来增长的主要推动力。云计算、人工智能和数宇化转型被认为是引领行业发展的关键因素，投资',
                  }
examples_types = [ '新闻报道', '财务报告', '公司公告', '分析师报告' ]

questions = [ '今日，央行发布公告宣布降低利率，以刺激经济增长。这一降息举措将影响贷款利率，并在未来几个季度内对金融市场产生影响。',
              '“ABC公司今日发布公告称，已成功完成对XYZ公司股权的收购交易。本次交易是ABC公司在扩大业务范围、加强市场竞争力方面的重要举措。据悉，此次收购将进一',
              '公司资产负债表显示，公司偿债能力强劲，现金流充足，为未来投资和扩张提供了坚实的财务基础。',
              '“最新的分析报告指出，可再生能源行业预计将在未来几年经历持续增长，投资者应该关注这一领域的投资机会',
              '小明喜欢小新哟'
              ]


messages = [{"role": "system", "content": "你是金融专家，将文本分类为新闻报道、财务报告、公司公告或分析师报告。不清楚的分类请返回'其他'"}]


for key,value in examples_data.items():
    messages.append({"role": "user", "content": value})
    messages.append({"role": "assistant", "content": key})

# for x in messages:
#     print(x)

for question in questions:
    response = client.chat.completions.create(
        model="qwen3:4b",  # 您可以按需更换为其它深度思考模型
        messages=messages + [{"role": "user", "content": f"按照示例，回答这段文本的分类类别：{question}"}],
        extra_body={"enable_thinking": True},
        # stream=True
    )
    print(response.choices[0].message.content)
#
# completion = client.chat.completions.create(
#     model="qwen3:4b",  # 您可以按需更换为其它深度思考模型
#     messages=messages,
#     extra_body={"enable_thinking": True},
#     stream=True
# )
# is_answering = False  # 是否进入回复阶段
# print("\n" + "=" * 20 + "思考过程" + "=" * 20)
# for chunk in completion:
#     delta = chunk.choices[0].delta
#     if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
#         if not is_answering:
#             print(delta.reasoning_content, end="", flush=True)
#     if hasattr(delta, "content") and delta.content:
#         if not is_answering:
#             print("\n" + "=" * 20 + "完整回复" + "=" * 20)
#             is_answering = True
#         print(delta.content, end="", flush=True)
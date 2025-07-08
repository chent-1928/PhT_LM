## 目录

- [项目简介](#项目简介)
- [数据集](#数据集)
- [模型](#模型)
- [显存限制](#显存限制)
- [环境搭建](#环境搭建)

## 项目简介

PhT-LM = 检索模块 + 大模型模块

该项目旨在为医药行业从事人员提供医药领域的英汉双语翻译。

**检索模块：**

功能：输入待翻译句子，经过该模块获取与待翻译句子匹配度最高的词、句，作为大模型的in_context的例子，提高模型翻译能力。

 **大模型模块：**

 训练了专注于医药领域英汉双语翻译模型，该模型由由构建的翻译数据集对Qwen-1_8B模型微调而来。

**注意：**PhT-LM有输入长度限制。如果待翻译内容过长，请切分段落依次翻译

## 数据集
- test_en_2_zh_1.json：英译汉测试集(1个上下文样例)，位于data/目录下
- test_en_2_zh_2.json：英译汉测试集(2个上下文样例)，位于data/目录下
- test_en_2_zh.json：英译汉测试集(4个上下文样例)，位于data/目录下
- test_en_2_zh_8.json：英译汉测试集(8个上下文样例)，位于data/目录下
- test_en_2_zh_16.json：英译汉测试集(16个上下文样例)，位于data/目录下
- test_en_2_zh_es.json：英译汉测试集(4个上下文样例，基于es检索策略)，位于data/目录下
- test_en_2_zh_vec.json：英译汉测试集(4个上下文样例， 基于vec检索策略)，位于data/目录下
- test_en_2_zh_without_context.json：英译汉测试集(无上下文样例)，位于data/目录下
- ICH_file_en_2_zh.json: 英译汉测试集(基于ICH文件《S1A指南：药品致癌性测试必要性指导原则》构建)，位于data/目录下
- test_zh_2_en_1.json：汉译英测试集(1个上下文样例)，位于data/目录下
- test_zh_2_en_2.json：汉译英测试集(2个上下文样例)，位于data/目录下
- test_zh_2_en.json：汉译英测试集(4个上下文样例)，位于data/目录下
- test_zh_2_en_8.json：汉译英测试集(8个上下文样例)，位于data/目录下
- test_zh_2_en_16.json：汉译英测试集(16个上下文样例)，位于data/目录下
- test_zh_2_en_es.json：汉译英测试集(4个上下文样例，基于es检索策略)，位于data/目录下
- test_zh_2_en_vec.json：汉译英测试集(4个上下文样例，基于vec检索策略)，位于data/目录下
- test_zh_2_en_without_context.json：汉译英测试集(无上下文样例)，位于data/目录下
- ICH_file_zh_2_en.json: 汉译英测试集(基于ICH文件《S1A指南：药品致癌性测试必要性指导原则》构建)，位于data/目录下
- 训练集暂不提供。
- 有需要请发邮件至1928539732@qq.com，注明身份、用途。

## 模型

- 模型获取地址：链接：https://pan.baidu.com/s/1PA3ZmdOklUhCfL8VjTANUg  提取码：41yk

## 显存限制

- 4G+

## 环境搭建

1. 下载安装ES（8.9.0），作为数据库
2. 进入src/retrieval/retrieval/retrieval/config.py文件配置刚刚安装的ES相关设置（ES_API，BASIC_AUTH）
3. pip install -r requirements.txt 下载项目运行所需的包
4. 下载网盘提供的模型，将model/文件夹拷贝到PhT_LM根目录下
5. 执行src/retrieval/insert_data.py文件，构建检索模块知识库（文档库和向量库），并插入数据（若插入时程序报错：es窗口最大查询数量为10000，则需要发送请求修改es查询的最大返回数目，注：该部分数据暂不提供，有需要请发送邮件）。

   ```bibtex
   # kb_name为src/retrieval/retrieval/retrieval/config.py文件中KB_NAME的参数值
   请求地址：http://IP:port/kb_name/_settings
   请求方式：PUT
   请求体：
   {
     "max_result_window":150000
   }
   ```

   发送请求后，进入src/retrieval/insert_data.py文件，将Line 34行的create_kb()代码注释掉，重新运行该文件即可。

- 如何使用（二选一）：

  **注意：**PhT-LM有输入长度限制。如果待翻译内容过长，请切分段落依次翻译

  1. web_demo界面（直接使用）

  ```bibtex
  python src/web_demo.py --model_name_or_path model/translation_model
  ```
  2. 简单调用（代码调用）

  ```bibtex
  # bash  开启模型API接口
  bash model_api.sh
  ```
  ```bibtex
  # python
  import requests
  import json


  url_api = "http://IP:8000/chat"
  def request_model_api(body):
     response = requests.post(url_api, json=body)  # 使用POST请求

     if response.status_code == 200:  # 判断响应状态码为200表示成功
         resp = json.loads(response.content)
         return resp['content'].strip("\n").strip(" ")
     else:
         print("Error occurred while calling the API.")
         return None

  def get_answer(text, is_zh, topk, fusion_weight, is_es):
      request_body = {
          "model": "string",
          "query": text,
          "is_zh": is_zh, 
          "topk": topk, 
          "fusion_weight": fusion_weight, 
          "is_es": is_es
      }
      return request_model_api(request_body)

  if __name__=='__main__':
      # text参数：待翻译文本
      # is_zh参数：待翻译文本是否为中文
      # topk参数：上下文示例个数
      # fusion_weight参数：当检索方式为 es检索+向量库检索时，两者的比重
      # is_es参数：是否仅es检索
      # 经实验发现topk=4最佳
      print(get_answer("精神分裂症有哪些临床表现？", True, 4, 0.5, False))
  ```

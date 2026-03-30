#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI兼容API压力测试脚本
支持多线程并发请求测试

功能说明:
---------
本脚本用于对OpenAI兼容的大模型API进行压力测试，支持多线程并发请求。
所有线程同时启动，结果按完成顺序实时打印。

配置文件:
---------
需要在项目根目录创建 .env 文件，包含以下配置：
    OPENAI_API_BASE=https://api.openai.com/v1    # API地址
    OPENAI_API_KEY=your-api-key-here             # API密钥
    MODEL_NAME=gpt-3.5-turbo                     # 模型名称

安装依赖:
---------
pip install -r requirements.txt

使用方法:
---------
1. 基本用法（使用默认参数）:
   python stress_test.py

2. 指定线程数:
   python stress_test.py -t 10
   python stress_test.py --threads 10

3. 指定测试消息:
   python stress_test.py -m "你是什么模型"
   python stress_test.py --message "介绍一下你自己"

4. 组合使用:
   python stress_test.py -t 20 -m "你好"

参数说明:
---------
-t, --threads   : 并发线程数，默认5
-m, --message   : 发送的测试消息，默认"你是什么模型"

输出说明:
---------
脚本会实时显示每个线程的执行结果，包括：
- 线程编号
- 成功/失败状态
- 响应时间
- API返回的完整内容
- 最终统计信息（总耗时、成功率、平均响应时间等）

注意事项:
---------
- 所有线程同时启动，结果按完成顺序实时打印
- 确保.env文件配置正确
- 建议从小线程数开始测试，避免对服务器造成过大压力
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 统计数据
stats = {
    'success': 0,
    'failed': 0,
    'total_time': 0.0
}
stats_lock = Lock()


def send_request(client, model, message, thread_id):
    """
    发送单个请求到API
    
    Args:
        client: OpenAI客户端实例
        model: 模型名称
        message: 要发送的消息
        thread_id: 线程ID
    
    Returns:
        tuple: (是否成功, 响应时间, 响应内容或错误信息)
    """
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": message}
            ]
        )
        elapsed_time = time.time() - start_time
        
        # 提取响应内容
        content = response.choices[0].message.content
        
        return True, elapsed_time, content
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        return False, elapsed_time, str(e)


def update_stats(success, elapsed_time):
    """更新统计数据"""
    with stats_lock:
        if success:
            stats['success'] += 1
        else:
            stats['failed'] += 1
        stats['total_time'] += elapsed_time


def run_stress_test(threads, message, verbose=False):
    """
    运行压力测试
    
    Args:
        threads: 线程数量
        message: 要发送的消息
        verbose: 是否显示详细输出
    """
    # 从环境变量读取配置
    api_base = os.getenv('OPENAI_API_BASE')
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('MODEL_NAME')
    
    if not all([api_base, api_key, model]):
        print("错误: 请在.env文件中配置OPENAI_API_BASE, OPENAI_API_KEY和MODEL_NAME")
        return
    
    print(f"=== OpenAI兼容API压力测试 ===")
    print(f"API地址: {api_base}")
    print(f"模型: {model}")
    print(f"线程数: {threads}")
    print(f"测试消息: {message}")
    print(f"{'='*50}\n")
    
    # 创建OpenAI客户端
    client = OpenAI(
        api_key=api_key,
        base_url=api_base
    )
    
    # 开始测试
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # 同时提交所有任务
        future_to_thread = {
            executor.submit(send_request, client, model, message, i): i + 1
            for i in range(threads)
        }
        
        print(f"所有 {threads} 个线程已同时启动，等待响应...\n")
        print(f"{'='*50}")
        
        # 实时处理完成的任务（哪个先完成就先打印哪个）
        for future in as_completed(future_to_thread):
            thread_num = future_to_thread[future]
            success, elapsed_time, result = future.result()
            update_stats(success, elapsed_time)
            
            # 实时打印每个线程的结果
            status = "✓ 成功" if success else "✗ 失败"
            print(f"\n[线程 {thread_num}] {status} | 耗时: {elapsed_time:.2f}秒")
            
            if success:
                print(f"API返回内容:")
                print(f"{result}")
            else:
                print(f"错误信息: {result}")
            
            print(f"{'-'*50}")
    
    total_elapsed = time.time() - start_time
    
    # 输出统计结果
    print(f"\n{'='*50}")
    print(f"=== 测试完成 ===")
    print(f"总耗时: {total_elapsed:.2f}秒")
    print(f"成功请求: {stats['success']}")
    print(f"失败请求: {stats['failed']}")
    print(f"成功率: {stats['success']/threads*100:.1f}%")
    if stats['success'] > 0:
        print(f"平均响应时间: {stats['total_time']/stats['success']:.2f}秒")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description='OpenAI兼容API压力测试工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python stress_test.py -t 10 -m "你是什么模型"
  python stress_test.py --threads 20 --message "介绍一下你自己"
        """
    )
    
    parser.add_argument(
        '-t', '--threads',
        type=int,
        default=5,
        help='并发线程数 (默认: 5)'
    )
    
    parser.add_argument(
        '-m', '--message',
        type=str,
        default='你是什么模型',
        help='发送的测试消息 (默认: "你是什么模型")'
    )
    
    args = parser.parse_args()
    
    # 运行压力测试
    run_stress_test(args.threads, args.message)


if __name__ == '__main__':
    main()

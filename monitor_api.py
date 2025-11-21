#!/usr/bin/env python3
"""
实时监控Web API调用，捕获详细错误信息
"""

import requests
import json
import time
import os

def monitor_api_calls():
    """监控API调用并记录详细错误"""
    base_url = "http://127.0.0.1:5000"
    
    print("=== Web API 实时监控 ===")
    print("正在监控API调用，请在前端界面进行操作...")
    print("按 Ctrl+C 停止监控\n")
    
    # 创建日志文件
    log_file = "api_monitor.log"
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"API监控日志 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
        
        while True:
            try:
                # 测试基本连接
                response = requests.get(f"{base_url}/api/languages", timeout=5)
                if response.status_code != 200:
                    print(f"⚠️  API连接异常: {response.status_code}")
                    
                # 检查是否有新的上传文件
                upload_dir = "static/uploads"
                if os.path.exists(upload_dir):
                    files = os.listdir(upload_dir)
                    if files:
                        print(f"📁 检测到上传文件: {len(files)} 个")
                        
                time.sleep(2)
                
            except requests.exceptions.RequestException as e:
                print(f"❌ 网络错误: {e}")
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{time.strftime('%H:%M:%S')}] 网络错误: {e}\n")
                time.sleep(5)
                
    except KeyboardInterrupt:
        print("\n🛑 监控已停止")
        print(f"📋 日志已保存到: {log_file}")

def test_frontend_errors():
    """模拟前端可能遇到的错误情况"""
    base_url = "http://127.0.0.1:5000"
    
    print("=== 测试前端错误情况 ===")
    
    # 测试1: 空文件上传
    print("\n1. 测试空文件上传...")
    try:
        files = {'file': ('', b'', 'audio/wav')}
        response = requests.post(f"{base_url}/api/upload", files=files)
        print(f"   结果: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 测试2: 无效文件路径
    print("\n2. 测试无效文件路径...")
    try:
        data = {"file_path": "nonexistent.wav", "language": "auto"}
        response = requests.post(f"{base_url}/api/transcribe", json=data)
        print(f"   结果: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 测试3: 空字幕翻译
    print("\n3. 测试空字幕翻译...")
    try:
        data = {"segments": [], "target_language": "zh-cn", "source_language": "en"}
        response = requests.post(f"{base_url}/api/translate", json=data)
        print(f"   结果: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 测试4: 空字幕生成
    print("\n4. 测试空字幕生成...")
    try:
        data = {"segments": [], "format": "srt", "filename": "test"}
        response = requests.post(f"{base_url}/api/generate-subtitle", json=data)
        print(f"   结果: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   错误: {e}")

if __name__ == "__main__":
    # 先测试错误情况
    test_frontend_errors()
    
    print("\n" + "="*50 + "\n")
    
    # 然后开始监控
    monitor_api_calls()
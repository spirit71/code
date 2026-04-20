import os
import time
import subprocess
import torch

def wait_for_gpu(target_free_mb=30000, timeout=3600):
    """等待GPU有足够显存可用"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', 
                                   '--format=csv,nounits,noheader'], 
                                  capture_output=True, text=True)
            free_memory = int(result.stdout.strip().split('\n')[0])
            
            if free_memory >= target_free_mb:
                print(f"✓ GPU显存充足：{free_memory}MB 可用")
                return True
            
            print(f"⏳ 等待GPU显存... 当前可用: {free_memory}MB, 需要: {target_free_mb}MB")
            time.sleep(30)  # 每30秒检查一次
        except Exception as e:
            print(f"检查GPU失败: {e}")
            time.sleep(10)
    
    raise RuntimeError(f"等待GPU超时，{timeout}秒内未获得足够显存")

if __name__ == "__main__":
    wait_for_gpu(target_free_mb=30000)  # 等待30GB可用显存
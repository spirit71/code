import pynvml
import time
import subprocess
import os

def get_free_vram(gpu_index=0):
    """获取指定 GPU 的剩余显存 (单位: MB)"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_vram = info.free / 1024**2  # 转换为 MB
    pynvml.nvmlShutdown()
    return free_vram

def run_training():
    """执行你的训练命令"""
    # 这里填入你之前报错的完整命令
    cmd = "python train.py "
    
    print(f"\n🚀 显存充足！开始执行训练命令: \n{cmd}\n")
    # 使用 subprocess 运行，并等待其结束
    return_code = subprocess.call(cmd, shell=True)
    return return_code

def monitor_and_launch(threshold_mb=20480, check_interval=10, gpu_index=0):
    """
    threshold_mb: 启动训练所需的最小剩余显存（建议设大一点，比如 20GB = 20480）
    check_interval: 检查频率（秒）
    """
    print(f"📡 监控已启动。等待 GPU {gpu_index} 释放显存 (阈值: {threshold_mb} MB)...")
    
    while True:
        try:
            free_mem = get_free_vram(gpu_index)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            if free_mem >= threshold_mb:
                print(f"[{current_time}] ✅ 检测到可用显存: {free_mem:.2f} MB。准备启动！")
                # 启动训练
                ret = run_training()
                
                if ret == 0:
                    print("🎉 任务圆满完成！脚本退出。")
                    break
                else:
                    print("⚠️ 任务执行中途出错或被手动停止。5秒后重新进入监控状态...")
                    time.sleep(5)
            else:
                print(f"[{current_time}] ⏳ 显存不足 (剩余: {free_mem:.2f} MB)，持续监测中...", end='\r')
                
        except Exception as e:
            print(f"\n❌ 监测过程中出错: {e}")
            
        time.sleep(check_interval)

if __name__ == "__main__":
    # 根据你的模型设置阈值。BoQ + DinoV2-ViT-B 建议至少需要 15GB-20GB 剩余空间
    # 你的显卡总容量是 80G，建议设为 30000 (30G) 比较稳妥
    monitor_and_launch(threshold_mb=30000, check_interval=15, gpu_index=0)
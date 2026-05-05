# Console Log File 报错修复说明（2026-05-04）

## 1. 修改背景
在 `--test_only` 或批量评估脚本场景下，程序会尝试把控制台输出镜像到：

- `./logs/<backbone>/version_x/console.log`

但有时该 `version_x` 目录尚未创建，导致报错：

```text
FileNotFoundError: [Errno 2] No such file or directory: './logs/.../version_x/console.log'
```

## 2. 根因分析
原代码中直接 `open(console_log_path, 'a')`，没有确保父目录存在。

- 旧逻辑：
  1. 计算 `console_log_path`
  2. 直接 `open(...)`
- 问题点：当 `tensorboard_logger.log_dir` 对应目录未创建时，`open` 失败。

## 3. 修改内容
文件：`train.py`

### 3.1 修改前
- `log_dir` 是字符串，未创建目录。
- 直接打开 `console.log`。

### 3.2 修改后
- 使用 `Path` 处理路径。
- 在打开文件前执行：

```python
log_dir.mkdir(parents=True, exist_ok=True)
```

- 再执行 `open(...)`。

## 4. 关键代码对比

### Before
```python
log_dir = tensorboard_logger.log_dir
console_log_path = f"{log_dir}/console.log"
file_obj = open(console_log_path, "a", encoding="utf-8", buffering=1)
```

### After
```python
log_dir = Path(tensorboard_logger.log_dir)
log_dir.mkdir(parents=True, exist_ok=True)
console_log_path = log_dir / "console.log"
file_obj = open(str(console_log_path), "a", encoding="utf-8", buffering=1)
```

## 5. 修改意义
1. 解决 `FileNotFoundError`。
2. 让训练、测试、批量评估场景都能稳定写入控制台日志。
3. 提升可回溯性，不依赖终端滚动历史。

## 6. 兼容性与风险
- 风险低：只增加目录存在性保障。
- 不影响原有日志格式、不影响训练逻辑。

## 7. 验证方式
1. 语法检查：

```bash
python -m py_compile train.py
```

2. 运行 `--test_only` 或批量评估脚本，确认：
- 不再出现 `No such file or directory`。
- 对应 `version_x/console.log` 正常生成并写入内容。

## 8. 回滚方式
若需回滚，只需恢复 `train.py` 中 `_setup_console_file_logging()` 的改动。

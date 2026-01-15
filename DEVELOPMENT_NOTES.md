# 项目开发笔记

> 内部开发文档，记录项目结构、优化细节和开发注意事项

## 项目文件结构

### 核心模块文件（模块化版本）

#### 1. `main.py` - 主程序入口和核心逻辑
**主要功能**：
- **EARSInjector类**：EARS规则注入的主控制器
  - 解析EARS规则文件
  - 扫描CRD文件
  - 匹配规则和CRD章节
  - 并行化LLM调用进行规则注入
  - 生成输出文件（injected.md, patches, patched files）
- **命令行接口**：完整的argparse参数解析
- **性能优化**：
  - 并行处理（ThreadPoolExecutor）
  - 进度显示（tqdm）
  - 缓存机制（ECU扫描结果）

**关键方法**：
- `find_matches()`: 寻找规则和CRD章节的匹配
- `inject_rules()`: 并行注入规则（使用LLM重写）
- `generate_outputs()`: 生成输出文件
- `_find_best_ecu_match()`: 基于ECU的匹配评分
- `_score_section()`: 章节评分

---

#### 2. `api.py` - LLM客户端（优化版）
**主要功能**：
- **LLMClient类**：与本地Ollama API交互
  - 自动检测模型（优先qwen3，其次deepseek）
  - 默认端口11435（A100 GPU优化）
  - Stream API支持（更快）
  - GPU优化配置（num_gpu=99）

**关键方法**：
- `rewrite_with_llm()`: 使用LLM重写段落，注入IF条件
- `_call_ollama_api()`: 调用Ollama API（stream模式）
- `_fallback_rewrite()`: LLM不可用时的备用方法

**特点**：
- 针对A100 GPU优化
- 支持streaming响应
- 自动清理thinking标签

---

#### 3. `crd_processing.py` - CRD文档处理
**主要功能**：
- **CRDFile类**：表示整个CRD文件
  - 读取文件（支持UTF-8和latin-1编码）
  - 按标题分割章节
  - 过滤TOC（目录）内容

- **CRDSection类**：表示CRD文件的一个章节
  - 分割段落
  - 检测表格/列表（避免注入）
  - ECU扫描（正则表达式或LLM）
  - 寻找最佳注入段落

**关键方法**：
- `scan_ecu_and_conditions()`: 扫描ECU组件和条件
- `find_best_paragraph()`: 寻找最适合注入的段落
- `_is_table_like()`: 检测表格/列表内容
- `_has_sufficient_context()`: 检查段落是否有足够上下文

**特点**：
- 默认使用快速正则表达式扫描
- 可选LLM扫描（更准确但更慢）
- 智能过滤表格、列表、TOC

---

#### 4. `ears_parsing.py` - EARS规则解析
**主要功能**：
- **EARSRule类**：解析和标准化EARS规则
  - 解析规则文本（IF...THEN格式）
  - 提取条件部分（IF部分）
  - 提取响应部分（THEN部分，当前不使用）
  - 标准化条件为正则表达式模式

**关键方法**：
- `_parse_rule()`: 分割IF和THEN部分
- `_normalize_condition()`: 标准化条件文本

**特点**：
- 只使用IF部分进行注入
- THEN部分保留但不使用

---

#### 5. `check_utils.py` - 工具函数
**主要功能**：
- **文本处理工具**：
  - `count_words()`: 计算单词数
  - `truncate_to_word_limit()`: 截断文本到指定单词数（保留句子边界）
  - `calculate_similarity()`: 计算文本相似度
  - `check_similarity_threshold()`: 检查相似度阈值

**用途**：
- 限制LLM输入/输出长度
- 验证注入结果质量
- 文本预处理

---

### 数据文件

- **`EARSrules.txt`**: EARS规则定义（每行一个规则，IF...THEN格式）
- **`CRD/`**: CRD文档目录（包含待处理的.txt文件）

---

### 文件依赖关系

```
main.py (主入口)
├── api.py (LLM客户端)
│   └── check_utils.py (工具函数)
├── crd_processing.py (CRD处理)
│   ├── check_utils.py (工具函数)
│   └── ears_parsing.py (规则解析)
└── ears_parsing.py (规则解析)
```

---

## 性能优化总结

### 已完成的优化 ✅

#### 1. 并行化LLM调用
- **位置**: `main.py` 的 `inject_rules()` 方法
- **改进**: 使用 `ThreadPoolExecutor` 并行处理多个规则的LLM重写调用
- **性能提升**: 预期 3-5倍速度提升（取决于并发数和硬件）
- **配置**: 通过 `--max-workers` 参数控制并发数（默认4）

#### 2. 进度显示
- **位置**: `main.py` 的 `find_matches()` 和 `inject_rules()` 方法
- **改进**: 使用 `tqdm` 显示处理进度
- **显示内容**: 
  - Section扫描进度
  - 规则注入进度（显示当前处理的规则）
- **依赖**: `tqdm>=4.65.0`（已添加到requirements.txt）

#### 3. 缓存机制
- **位置**: `main.py` 的 `EARSInjector` 类
- **改进**: 使用内容hash缓存ECU扫描结果，避免重复扫描相同section
- **性能提升**: 对于重复section，避免重复计算

#### 4. ECU扫描优化
- **位置**: `crd_processing.py` 的 `scan_ecu_and_conditions()` 方法
- **改进**: 
  - 默认使用快速的正则表达式扫描（fallback方法）
  - 添加 `use_llm` 参数，仅在需要时启用LLM扫描
  - LLM扫描失败时自动回退到正则表达式方法

---

### 代码变更记录

#### 主要文件修改

1. **main.py**
   - 添加并行处理支持（`ThreadPoolExecutor`）
   - 添加进度显示（`tqdm`）
   - 添加缓存机制（`_ecu_scan_cache`）
   - 优化 `inject_rules()` 方法：并行处理注入任务
   - 优化 `find_matches()` 方法：添加进度显示

2. **crd_processing.py**
   - 优化 `scan_ecu_and_conditions()` 方法：添加 `use_llm` 参数

3. **requirements.txt**
   - 添加 `tqdm>=4.65.0` 依赖

---

### 性能对比

#### 优化前
- **ECU扫描**: 串行处理，每个section顺序扫描
- **规则注入**: 串行处理，每个规则顺序调用LLM
- **总耗时**: 10个sections + 10个规则 ≈ 5-10分钟

#### 优化后
- **ECU扫描**: 使用快速正则表达式（默认），缓存结果
- **规则注入**: 并行处理（默认4个并发）
- **预期总耗时**: 10个sections + 10个规则 ≈ 1-3分钟（取决于硬件）

---

### 使用方法

#### 基本使用（使用默认并发数4）
```bash
python3 main.py --crd-dir ./CRD --output-dir output
```

#### 自定义并发数
```bash
python3 main.py --crd-dir ./CRD --output-dir output --max-workers 8
```

#### 启用LLM扫描（较慢但更准确）
需要在代码中修改 `scan_ecu_and_conditions(use_llm=True)`，或添加命令行参数支持。

---

### 注意事项

1. **并发数选择**: 
   - CPU密集型任务：建议 `max_workers = CPU核心数`
   - I/O密集型任务（LLM调用）：可以设置更高（如8-16）
   - 注意：过多并发可能导致Ollama服务过载

2. **内存使用**: 
   - 并行处理会增加内存使用
   - 如果内存不足，减少 `max_workers`

3. **LLM服务**: 
   - 确保Ollama服务稳定运行
   - 如果遇到连接错误，检查Ollama服务状态
   - 默认端口11435（A100 GPU），可通过环境变量修改

---

### 后续优化建议

1. **批量LLM调用**: 将多个小任务合并为批量请求
2. **智能缓存**: 缓存LLM重写结果，避免重复处理相同内容
3. **异步处理**: 使用 `asyncio` 替代 `ThreadPoolExecutor`（需要异步LLM客户端）
4. **结果持久化**: 将缓存结果保存到磁盘，跨会话复用
5. **超时优化**: 增加LLM调用的timeout，或实现重试机制

---

### 测试建议

1. **小规模测试**: 使用1-2个CRD文件，少量规则
2. **性能测试**: 记录优化前后的处理时间
3. **功能测试**: 确保注入结果质量不受影响
4. **并发测试**: 测试不同 `max_workers` 值的效果

---

## 项目清理记录

### 已删除的旧版文件
- `inject_ears.py` - 单文件版本（1554行），已被模块化版本替代
- `run_injection.py` - 旧版运行脚本，依赖已删除的inject_ears.py
- `ollama_api.py` - 旧版API客户端，已被api.py替代
- 所有测试输出目录（output/, output_test/, output_deepseek/等）
- `ollama_gpu_11436.log` - 日志文件

### 当前项目结构
```
Issue-Injection/
├── main.py              # 主程序（模块化版本）
├── api.py               # LLM客户端
├── crd_processing.py    # CRD文档处理
├── ears_parsing.py      # EARS规则解析
├── check_utils.py       # 工具函数
├── EARSrules.txt        # EARS规则文件
├── requirements.txt     # Python依赖
├── README.md            # 项目文档
├── DEVELOPMENT_NOTES.md # 开发笔记（本文件）
└── CRD/                 # CRD文档目录
    └── Sample_ECU_Function_Specification.txt
```


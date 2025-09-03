
## 分段与目录过滤（已更新）

- 现在脚本会在读取CRD后进行“真实标题”分段，仅依据规范化编号标题和模块标题进行切分：
  - 支持 `1-1`、`1-1-1`、`1.1`、`1.1.1` 等编号样式，要求编号后必须有空格和标题文本
  - 保留大写模块标题与 `#` Markdown 标题
- 自动过滤目录（TOC）：
  - 丢弃正文开始前的目录/扉页等短节，依据大量点线、仅编号无标题、内容极短等特征
  - 从第一个满足“以 1- 或 1. 开头且内容充足”的实质章节开始作为正文起点
- 不再导出拆分后的章节到外部TXT；匹配与注入在内存中的 `sections` 上进行，避免额外IO
- 在匹配开始前，会打印各CRD文件的所有 `section` 标题与行号区间，方便人工核对

### 运行时输出示例

```text
Finding matches between rules and CRD sections...
Total sections: 16
Sections in Air-Conditioning Gateway Function_Ver_1_6.txt:
- [210-223] 1-1. Definition of change description method
- [224-252] 1-2. applicable specification
- [253-274] 1-3. Terminology
- [391-559] 2-1. input list
- [560-600] 3-1. gateway function control
- [601-1531] 3-1-1. Gateway/arbitration function (CAN → LIN)
- [1532-1939] 3-1-2. Gateway function (LIN → CAN)
- [1940-2161] 3-1-3. fault diagnosis output function
- [2162-2232] 3-2. refrigerant specification discrimination control
...
```

### 注意
- 如果你只想核对分段结果，可直接运行脚本并查看分段打印；脚本随后会继续执行匹配与注入流程。
- 若需要新增仅打印分段的开关（例如 `--list-sections-only`），可提出需求后再添加。

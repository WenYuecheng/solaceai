# 第三方代码来源

## MODULAR-RAG-MCP-SERVER

- 仓库：[jerry-ai-dev/MODULAR-RAG-MCP-SERVER](https://github.com/jerry-ai-dev/MODULAR-RAG-MCP-SERVER)
- 固定提交：`f658c5a4011c8b826707a65a3b11ce9301b0626f`
- 原文件：`src/core/query_engine/fusion.py`，`RRFFusion.__init__`与`RRFFusion.fuse`
- 本项目位置：`app/services/retrieval.py`中的`RRFFusion`
- 修改：复用排名累加、首次命中保留、稳定排序与Top-K实现；数据载荷适配为LlamaIndex的NodeWithScore；删除未使用的加权融合、trace、调试日志与长篇示例。每路去重及Milvus通道选择由本项目执行。
- 许可证：上游 `pyproject.toml` 声明 MIT 许可，原文件与许可声明已归档留存。

以下为MIT许可标准条款，代码版权属于相应权利人。

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## DeepSeek Harness 压缩策略迁移

来源为同级目录 `deepseek-harness`，固定提交 `d743267388641bc76f17c45ce8b4c231aed1d32c`。参考 `packages/compaction/compaction-basic/src/config.ts`、`src/region.ts`、`src/summarizer.ts`，以及 `compaction-tool-result-pruner/src/config.ts`、`src/index.ts` 和 `compaction/src/tool-pairing.ts`。

本项目将比例与预算公式、完整轮次保留、工具头尾修剪和受限恢复适配为 Python，主要位于 `app/services/compaction.py` 与 `app/services/context.py`，由现有 LangGraph 调用。采用 0.8、0.16 及 8192/4096/1024 码点修剪参数，小窗口额外余量配置为 512 tokens，并结合心理领域摘要提示词、数据库并发保护与证据投影完成原生 Python 适配。

上游许可证原文：

MIT License

Copyright (c) 2026 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## ZCOOL KuaiLe (站酷快乐体)

Source: https://github.com/google/fonts/tree/main/ofl/zcoolkuaile
License: SIL Open Font License 1.1. The full license accompanies the local font at `frontend/public/fonts/zcool-kuaile-OFL.txt`. Converted to WOFF2 for web delivery; font content is unchanged.

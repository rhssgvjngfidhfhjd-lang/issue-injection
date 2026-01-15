import os
import re
import json
import requests
from typing import Optional, List, Dict, Tuple, Any
from check_utils import truncate_to_word_limit, calculate_similarity, check_similarity_threshold

# 用于与本地 LLM 端点交互的客户端。
class LLMClient:
    """Client for interacting with local LLM endpoint (Optimized for A100 GPU)."""
    
    def __init__(self, base_url: str = None, api_key: str = None, similarity_threshold: float = 0.8, model: Optional[str] = None):
        # 1. 确定 Base URL：按照 传入参数 > 环境变量 > 11435 GPU端口 的顺序选择
        self.base_url = (
            base_url or 
            os.getenv('OLLAMA_BASE_URL') or 
            os.getenv('OLLAMA_HOST') or 
            'http://127.0.0.1:11435'
        ).rstrip('/')

        self.api_key = api_key or os.getenv('OLLAMA_API_KEY', '')
        self.model = model or os.getenv('OLLAMA_MODEL')
        
        # 2. 自动检测模型 (优先从 11435 端口获取)
        if not self.model:
            try:
                tags_url = f"{self.base_url}/api/tags"
                response = requests.get(tags_url, timeout=5)
                if response.status_code == 200:
                    tags = response.json().get('models', [])
                    names = [str(m.get('name','')) for m in tags]
                    prefer = ([n for n in names if n.startswith('qwen3:')] or
                              [n for n in names if n.startswith('deepseek')] or
                              names)
                    self.model = prefer[0] if prefer else 'qwen3:8b'
                else:
                    self.model = 'qwen3:8b'
            except Exception:
                self.model = 'qwen3:8b'
                
        self.similarity_threshold = similarity_threshold
        self.client = None
        print(f"LLMClient initialized. Target: {self.base_url}, Model: {self.model}")
    
    def polish_text(self, mutated_text: str, rule_condition: str) -> str:
        """
        Polish the programmatically mutated text to ensure linguistic fluency.
        Does NOT change the logic of the mutation.
        """
        mutated_text = truncate_to_word_limit(mutated_text, 300)
        
        instruction = (
            "You are a technical editor. The following text has had a fault injected programmatically "
            "(e.g., a number changed or a step reordered). Your task is to SMOOTH the text so it flows naturally "
            "as valid English, BUT YOU MUST PRESERVE THE INJECTED FAULT logic exactly.\n"
            "- Do NOT fix the logic error.\n"
            "- Do NOT revert the numbers or order.\n"
            "- Just fix grammar, capitalization, and flow.\n"
            f"- The intended fault condition is related to: {rule_condition}\n\n"
            f"Input Text:\n{mutated_text}\n\n"
            "CRITICAL: Output ONLY the polished paragraph. No thinking process, no explanations, no conversational filler, no markdown blocks."
        )

        try:
            return self._call_ollama_api(instruction)
        except Exception as e:
            print(f"Polish error: {e}")
            return mutated_text # Return original mutation if LLM fails

    def rewrite_with_llm(self, section_paragraph: str, rule_condition: str, section_context: str = "") -> str:
        """Legacy rewrite method for rules without explicit mutation type."""
        section_paragraph = truncate_to_word_limit(section_paragraph, 300)
        section_context = truncate_to_word_limit(section_context, 300)
        
        instruction = (
            "Rewrite the technical paragraph to integrate ONLY the rule's condition (IF-part) "
            "as natural descriptive text. Do not add requirements words (shall/must) or the THEN-part. "
            "Map any placeholder ECUs (e.g., ECU A/B) to actual ECU names found in the text.\n\n"
            f"Paragraph:\n{section_paragraph}\n\n"
            f"Rule condition (IF-only): {rule_condition}\n\n"
            f"Extra context:\n{section_context}\n\n"
            "CRITICAL: Output ONLY the final rewritten paragraph. No thinking process, no explanations, no conversational filler, no markdown blocks."
        )

        try:
            result = self._call_ollama_api(instruction)
            similarity = calculate_similarity(section_paragraph, result)
            
            if similarity < 0.3:
                print(f"Similarity too low ({similarity:.2f}), using fallback")
                return self._fallback_rewrite(section_paragraph, rule_condition)
            return result
        except Exception as e:
            print(f"Ollama API error: {e}")
            return self._fallback_rewrite(section_paragraph, rule_condition)
    
    def _call_ollama_api(self, instruction: str) -> str:
        """Call native Ollama generate API with GPU force options."""
        url = f"{self.base_url}/api/generate"
        prompt = (
            "You are an expert technical writer specializing in automotive system requirements.\n\n"
            + instruction
        )
        
        # 核心配置：加入 num_gpu 强制启用 A100
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 128,    # A100 处理速度快，可以稍微增加预测长度
                "temperature": 0.4,
                "keep_alive": "10m",   # 保持模型在显存中更久
                "num_gpu": 99,         # 关键：强制 99 层全放入 GPU (A100 完全够用)
                "num_thread": 8        # 限制 CPU 线程，倒逼 GPU 处理
            }
        }

        # 针对 DeepSeek 的优化
        if self.model and "deepseek" in str(self.model).lower():
            try:
                payload["options"]["think"] = "low"
            except: pass

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        response = requests.post(url, json=payload, headers=headers, timeout=300, stream=True)
        response.raise_for_status()
        
        content_parts = []
        for line in response.iter_lines(decode_unicode=True):
            if not line: continue
            try:
                obj = json.loads(line)
                if 'response' in obj:
                    content_parts.append(obj['response'])
                if obj.get('done'): break
            except: continue
            
        content = ''.join(content_parts).strip()
        
        # 清理思考过程标签和 Markdown 块
        content = re.sub(r'<(thinking|think)>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        # 移除常见的对话引导语
        content = re.sub(r'^(Okay|Sure|Here is|Polished paragraph:|Rewritten paragraph:).*?:\s*', '', content, flags=re.IGNORECASE | re.MULTILINE)
        return truncate_to_word_limit(content.strip(), 500)
    
    def _fallback_rewrite(self, section_paragraph: str, rule_condition: str) -> str:
        """Simple fallback rewrite when LLM is unavailable."""
        text = section_paragraph
        ecu_names = re.findall(r"\b([A-Za-z][A-Za-z \-/]* ECU)\b", text, re.IGNORECASE)
        cond = re.sub(r"\b(shall|must|should|will)\b", "", rule_condition or "", re.IGNORECASE).strip()
        
        if ecu_names:
            ecu_mapping = {f"ECU {chr(65+i)}": name for i, name in enumerate(ecu_names[:4])}
            for placeholder, actual_name in ecu_mapping.items():
                cond = cond.replace(placeholder, actual_name)
            cond = re.sub(r"\bECU [A-Z]\b", "ECU", cond)
        
        if cond:
            result = f"{text}\n{cond}" if not text.endswith("\n") else f"{text}{cond}"
            return truncate_to_word_limit(result, 500)
        return truncate_to_word_limit(text, 500)

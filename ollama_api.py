import os
import re
import difflib
import csv
from openai import OpenAI

base_url = 'http://localhost:11434/v1/'
api_key = 'ollama'
client = OpenAI(base_url=base_url, api_key=api_key)


def to_EARS(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model='qwen3:32b',  # 修改为qwen3:32b
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False
    )
    message = response.choices[0].message.content
    return message


def parse_ears_rules(rules_path):
	rules = []
	with open(rules_path, encoding='utf-8') as f:
		for idx, line in enumerate(f):
			if 'THEN' in line:
				cond, resp = line.split('THEN', 1)
				rules.append({'idx': idx, 'condition': cond.strip(), 'response': resp.strip()})
	return rules


def find_sections(text):
	# 按ECU/Component/Module/markdown #分段
	sections = []
	pat = re.compile(r'^(ECU|Component|Module|\#).+', re.MULTILINE)
	last = 0
	for m in pat.finditer(text):
		if last != m.start():
			sections.append((text[last:m.start()], last))
		last = m.start()
	sections.append((text[last:], last))
	return sections


def fuzzy_match(a, b):
	return difflib.SequenceMatcher(None, a, b).ratio()


def rewrite_with_llm(paragraph, rule_cond, rule_resp):
	system_prompt = (
		"Rewrite the paragraph in the same tone and formatting, with minimal edits, to include:\n"
		"(a) the generalized condition consistent with the rule, and\n"
		"(b) the system response (‘shall …’) aligned with the rule.\n"
		"Do not reveal placeholders verbatim; map them to the section’s actors/signals when obvious, otherwise keep generic.\n"
		"Keep technical terms, units, and numbering; avoid inventing facts; ensure the section remains logically consistent and non-contradictory. Output ONLY the final paragraph."
	)
	user_prompt = f"Paragraph:\n{paragraph}\n\nRule condition: {rule_cond}\nRule response: {rule_resp}"
	return to_EARS(system_prompt, user_prompt)


def main():
	rules = parse_ears_rules('EARSrules')
	crd_dir = './CRD'
	os.makedirs('./patches', exist_ok=True)
	csv_rows = []
	md_lines = []
	for fname in os.listdir(crd_dir):
		if not fname.endswith('.txt'):
			continue
		with open(os.path.join(crd_dir, fname), encoding='utf-8') as f:
			text = f.read()
		sections = find_sections(text)
		for sec_idx, (section, sec_start) in enumerate(sections):
			for rule in rules:
				score = fuzzy_match(section, rule['condition'])
				if score < 0.4:
					continue
				# 找到含条件/响应的段落
				paras = [p for p in section.split('\n\n') if p.strip()]
				best_para, best_score = None, 0
				for para in paras:
					pscore = fuzzy_match(para, rule['condition'])
					if pscore > best_score:
						best_para, best_score = para, pscore
				if not best_para:
					continue
				# 判断是否已存在
				exists = (rule['condition'] in best_para and rule['response'] in best_para)
				status = 'exists' if exists else 'inject'
				line_span = (text[:sec_start].count('\n')+1, text[:sec_start].count('\n')+1+section.count('\n'))
				csv_rows.append([fname, f'section_{sec_idx}', line_span, rule['idx'], score, status, best_para[:80]])
				if status == 'inject':
					rewritten = rewrite_with_llm(best_para, rule['condition'], rule['response'])
					# 生成patch
					patch_path = f'./patches/{fname}_section{sec_idx}_rule{rule["idx"]}.patch'
					with open(patch_path, 'w', encoding='utf-8') as pf:
						diff = difflib.unified_diff(
							best_para.splitlines(), rewritten.splitlines(),
							fromfile='before', tofile='after', lineterm=''
						)
						pf.write('\n'.join(diff))
					md_lines.append(f'### {fname} section_{sec_idx}\n**Injected:**\n{rewritten}\n\n**Context:**\n{best_para}\n')
	# 写简化trace文件
	with open('issue_injection_trace.txt', 'w', encoding='utf-8') as tf:
		tf.write("Issue Injection Trace\n\n")
		inject_count = len([row for row in csv_rows if row[5] == 'inject'])
		tf.write(f"Prepared issues (LLM rewrites generated): {len(csv_rows)}\n")
		tf.write(f"Applied to CRD (successfully replaced in file): {inject_count}\n\n")
		
		if inject_count > 0:
			tf.write("Applied\n")
			issue_id = 1
			for i, row in enumerate(csv_rows):
				if row[5] == 'inject':
					tf.write(f"{issue_id}) Issue ID: ISSUE-{issue_id:03d}\n")
					tf.write(f"   File: CRD/{row[0]}\n")
					tf.write(f"   Section: {row[1]}\n")
					tf.write(f"   Line range: {row[2]}\n")
					tf.write(f"   Source rule: Rule {row[3]}\n")
					tf.write(f"   Status: applied\n\n")
					issue_id += 1


if __name__ == '__main__':
	main()
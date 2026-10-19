"""Minimal stand-in for vLLM's /v1/chat/completions, for testing rollout.py without a GPU.

Picks a random admissible action from the prompt (seeded by the request seed) and answers in
the <think>/<action> format. Usage: python mock_llm.py --port 8011
"""
import argparse
import json
import random
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        prompt = body['messages'][0]['content']
        rng = random.Random(body.get('seed', 0))
        actions = re.findall(r"^'(search\[<your query>\]|click\[[^\n]*\])',$", prompt, flags=re.M)
        a = rng.choice(actions) if actions else 'click[back to search]'
        if a.startswith('search'):
            task = re.search(r'Your task is to: (.*)\.\n', prompt).group(1)
            a = f"search[{' '.join(rng.sample(task.split(), 4))}]"
        if rng.random() < 0.05:
            text = f'I will {a}'  # malformed answer, exercises the invalid-action path
        else:
            text = f'<think>mock reasoning</think><action>{a}</action>'
        out = {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
               'usage': {'prompt_tokens': len(prompt) // 4, 'completion_tokens': len(text) // 4}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8011)
    ThreadingHTTPServer(('127.0.0.1', p.parse_args().port), Handler).serve_forever()

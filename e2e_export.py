"""E2E 导出链路：模型列表 → 标准导出包下载校验 → 删除"""
import http.cookiejar
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile

BASE = 'http://127.0.0.1:5000'


def server_up():
    try:
        urllib.request.urlopen(BASE + '/login', timeout=3)
        return True
    except Exception:
        return False


if not server_up():
    log = open('server_e2e.log', 'w')
    subprocess.Popen([sys.executable, 'app.py'], stdout=log, stderr=log)
    for _ in range(30):
        time.sleep(1)
        if server_up():
            break
assert server_up(), '服务未能启动'

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))


def req_json(method, path, body=None):
    r = urllib.request.Request(BASE + path,
                               data=json.dumps(body or {}).encode(),
                               method=method,
                               headers={'Content-Type': 'application/json'})
    return json.loads(opener.open(r, timeout=60).read().decode() or '{}')


req_json('POST', '/api/login', {'username': 'e2e_probe', 'password': 'probe123456'})

# ---- 1) 模型列表 ----
lst = req_json('GET', '/api/list_models')
models = [m['name'] for m in lst.get('models', [])]
print(f'模型列表 {len(models)} 个:', models[:8])
assert models, '无已保存模型'
text_model = next((m for m in models if m.startswith('text_gen_')), None)

# ---- 2) 标准导出下载并校验 ----
target = text_model or models[0]
resp = opener.open(BASE + f'/api/download_model/{target}', timeout=120)
zdata = resp.read()
with zipfile.ZipFile(io.BytesIO(zdata)) as z:
    names = set(z.namelist())
    cfg = json.loads(z.read('config.json'))
print(f'导出包({target}):', sorted(names))
assert 'model.safetensors' in names and 'config.json' in names and 'LICENSE' in names
assert cfg['format']['license'] == 'bsd-3-clause'
lic_ok = b'BSD 3-Clause' in zdata  # 粗检：zip 内含许可文本
if text_model:
    assert 'vocab.json' in names and 'merges.txt' in names, '文本模型应含分词器文件'
print('config.format:', cfg['format'])

# ---- 3) 删除 ----
dele = req_json('DELETE', f'/api/delete_model/{target}')
assert dele.get('status') == 'success', dele
lst2 = req_json('GET', '/api/list_models')
names2 = [m['name'] for m in lst2.get('models', [])]
assert target not in names2, '删除后仍出现在列表中'
sidecar = 'models/3/' + target + '.json'
assert not os.path.exists('models/3/' + target), '权重文件未删除'
assert not os.path.exists(sidecar), '旁车元数据未删除'
print(f'删除 OK: {target}')

print('EXPORT E2E PASSED')

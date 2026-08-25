"""E2E 全链路驱动：登录→上传三类数据集→启动六类训练→轮询至终态"""
import http.cookiejar
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit('需要 cv2')

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


def req(method, path, json_body=None, data=None, files=None):
    url = BASE + path
    if files:
        boundary = '----e2eboundary2026'
        buf = io.BytesIO()
        for k, (fn, blob, ctype) in files.items():
            buf.write(('--%s\r\nContent-Disposition: form-data; name="%s"; '
                       'filename="%s"\r\nContent-Type: %s\r\n\r\n'
                       % (boundary, k, fn, ctype)).encode())
            buf.write(blob)
            buf.write(b'\r\n')
        for k, v in (data or {}).items():
            buf.write(('--%s\r\nContent-Disposition: form-data; name="%s"'
                       '\r\n\r\n%s\r\n' % (boundary, k, v)).encode())
        buf.write(('--%s--\r\n' % boundary).encode())
        r = urllib.request.Request(url, data=buf.getvalue(), method='POST',
                                   headers={'Content-Type':
                                            'multipart/form-data; boundary=' + boundary})
    else:
        payload = json.dumps(json_body if json_body is not None else (data or {})).encode()
        r = urllib.request.Request(url, data=payload, method=method,
                                   headers={'Content-Type': 'application/json'})
    try:
        resp = opener.open(r, timeout=180)
        return resp.status, json.loads(resp.read().decode() or '{}')
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or '{}')
        except ValueError:
            return e.code, {}


# ---- 登录 ----
req('POST', '/api/register', json_body={'username': 'e2e_probe',
                                        'password': 'probe123456'})
st, _ = req('POST', '/api/login', json_body={'username': 'e2e_probe',
                                             'password': 'probe123456'})
print('登录:', st)
assert st == 200


def png_bytes(seed):
    rng = np.random.RandomState(seed)
    import cv2
    return cv2.imencode('.png', rng.randint(0, 255, (28, 28, 3), np.uint8))[1].tobytes()


def make_zip(items):
    b = io.BytesIO()
    with zipfile.ZipFile(b, 'w') as z:
        for arc, blob in items:
            z.writestr(arc, blob)
    return b.getvalue()


# ---- 上传三类数据集 ----
img_zip = make_zip([(f'ds/{cls}/{i}.png', png_bytes(ord(cls[0]) + i))
                    for cls in ('cat', 'dog') for i in range(4)])
_, u1 = req('POST', '/api/upload', data={'train_type': 'image'},
            files={'file': ('ds.zip', img_zip, 'application/zip')})

text_blob = ('今天天气很好我们去公园散步看到很多人在锻炼身体大家都很开心。' * 40).encode('utf-8')
_, u2 = req('POST', '/api/upload', data={'train_type': 'text'},
            files={'file': ('corpus.txt', text_blob, 'text/plain')})

mm_zip = make_zip([(f'p{i}.png', png_bytes(i + 90)) for i in range(4)] +
                  [(f'p{i}.txt', '一张彩色的测试照片'.encode()) for i in range(4)])
_, u3 = req('POST', '/api/upload', data={'train_type': 'multimodal'},
            files={'file': ('mm.zip', mm_zip, 'application/zip')})

print('数据集上传:', u1.get('status'), u2.get('status'), u3.get('status'))
img_id, txt_id, mm_id = u1.get('file_id'), u2.get('file_id'), u3.get('file_id')

# ---- 六类训练任务 ----
tp = dict(learning_rate=0.001, epochs=1, batch_size=4)
jobs = [
    ('CNN分类',   dict(train_type='image', model_key='image_cnn',
                       dataset_id=img_id, image_size=24, num_classes=2,
                       base_channels=16, **tp)),
    ('ViT分类',   dict(train_type='image', model_key='image_vit',
                       dataset_id=img_id, image_size=24, patch_size=8,
                       num_classes=2, d_model=32, n_layers=1, n_heads=2,
                       d_ff=64, attention_type='flash', **tp)),
    ('扩散生成',  dict(train_type='image', model_key='image_diffusion',
                       dataset_id=img_id, image_size=24, base_channels=16,
                       num_timesteps=50, **tp)),
    ('扩散编辑',  dict(train_type='image', model_key='image_edit_diffusion',
                       dataset_id=img_id, image_size=24, base_channels=16,
                       num_timesteps=50, **tp)),
    ('文本生成',  dict(train_type='llm', model_key='text_generation',
                       dataset_id=txt_id, vocab_size=150, max_seq_len=16,
                       d_model=32, n_layers=1, n_heads=2, d_ff=64,
                       attention_type='full',
                       learning_rate=3e-5, epochs=1, batch_size=4)),
    ('多模态',    dict(train_type='multimodal', model_key='multimodal_stream',
                       dataset_id=mm_id, vocab_size=200, image_size=24,
                       patch_size=8, d_model=32, n_layers=1, n_heads=2,
                       d_ff=64, max_seq_len=48, attention_type='linear', **tp)),
]

pending = {}
for name, params in jobs:
    st, body = req('POST', '/api/start_training', json_body=params)
    ok = body.get('status') == 'success'
    print(f'启动 {name}: {"OK" if ok else "FAIL"} {"" if ok else body.get("message")}')
    if ok:
        pending[name] = body['task_id']

# ---- 轮询到全部终态 ----
results = {}
deadline = time.time() + 240
while pending and time.time() < deadline:
    time.sleep(3)
    for name, tid in list(pending.items()):
        _, s = req('GET', f'/api/task_status/{tid}')
        status = s.get('status')
        if status in ('completed', 'failed', 'cancelled'):
            results[name] = (status, str(s.get('message'))[:70])
            del pending[name]
for name, tid in pending.items():
    results[name] = ('timeout', tid)

print('\n===== 六类训练结果 =====')
bad = 0
for name, (status, msg) in results.items():
    flag = 'PASS' if status == 'completed' else 'FAIL'
    bad += status != 'completed'
    print(f'[{flag}] {name}: {status} | {msg}')
sys.exit(1 if bad else 0)

"""E2E：六类模型一键测试——训练每类模型，拉取本地模板并经 /api/run_test 真实执行"""
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


def req(method, path, json_body=None, data=None, files=None, raw=False):
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
    elif json_body is not None or data is not None:
        payload = json.dumps(json_body if json_body is not None else data).encode()
        r = urllib.request.Request(url, data=payload, method=method,
                                   headers={'Content-Type': 'application/json'})
    else:
        r = urllib.request.Request(url, method=method)
    try:
        resp = opener.open(r, timeout=300)
        blob = resp.read()
        return resp.status, (blob if raw else json.loads(blob.decode() or '{}'))
    except urllib.error.HTTPError as e:
        blob = e.read()
        try:
            return e.code, json.loads(blob.decode() or '{}')
        except ValueError:
            return e.code, {}


req('POST', '/api/register', json_body={'username': 'e2e_probe',
                                        'password': 'probe123456'})
st, _ = req('POST', '/api/login', json_body={'username': 'e2e_probe',
                                             'password': 'probe123456'})
assert st == 200


def png_bytes(seed):
    rng = np.random.RandomState(seed)
    return cv2.imencode('.png', rng.randint(0, 255, (28, 28, 3), np.uint8))[1].tobytes()


def make_zip(items):
    b = io.BytesIO()
    with zipfile.ZipFile(b, 'w') as z:
        for arc, blob in items:
            z.writestr(arc, blob)
    return b.getvalue()


# ---- 上传训练数据集 ----
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
img_id, txt_id, mm_id = u1.get('file_id'), u2.get('file_id'), u3.get('file_id')
print('数据集:', u1.get('status'), u2.get('status'), u3.get('status'))

# ---- 训练六类模型（小参数量） ----
tp = dict(learning_rate=0.001, epochs=1, batch_size=4)
jobs = {
    'image_cnn': dict(train_type='image', model_key='image_cnn',
                      dataset_id=img_id, image_size=24, num_classes=2,
                      base_channels=16, **tp),
    'image_vit': dict(train_type='image', model_key='image_vit',
                      dataset_id=img_id, image_size=24, patch_size=8,
                      num_classes=2, d_model=32, n_layers=1, n_heads=2,
                      d_ff=64, attention_type='flash', **tp),
    'diffusion': dict(train_type='image', model_key='image_diffusion',
                      dataset_id=img_id, image_size=24, base_channels=16,
                      num_timesteps=50, **tp),
    'diffusion_edit': dict(train_type='image', model_key='image_edit_diffusion',
                           dataset_id=img_id, image_size=24, base_channels=16,
                           num_timesteps=50, **tp),
    'text_generation': dict(train_type='llm', model_key='text_generation',
                            dataset_id=txt_id, vocab_size=150, max_seq_len=16,
                            d_model=32, n_layers=1, n_heads=2, d_ff=64,
                            attention_type='full',
                            learning_rate=3e-5, epochs=1, batch_size=4),
    'multimodal_stream': dict(train_type='multimodal',
                              model_key='multimodal_stream',
                              dataset_id=mm_id, vocab_size=200, image_size=24,
                              patch_size=8, d_model=32, n_layers=1, n_heads=2,
                              d_ff=64, max_seq_len=48,
                              attention_type='linear', **tp),
}
pending = {}
for key, params in jobs.items():
    _, body = req('POST', '/api/start_training', json_body=params)
    assert body.get('status') == 'success', (key, body.get('message'))
    pending[key] = body['task_id']
print('六类训练任务已启动')

deadline = time.time() + 240
while pending and time.time() < deadline:
    time.sleep(3)
    for key, tid in list(pending.items()):
        _, s = req('GET', f'/api/task_status/{tid}')
        if s.get('status') in ('completed', 'failed'):
            print(f"[{'OK' if s['status']=='completed' else 'ERR'}] 训练 {key}: "
                  f"{str(s.get('message'))[:60]}")
            if s['status'] != 'completed':
                sys.exit(f'训练失败: {key}')
            del pending[key]
assert not pending, f'训练超时: {list(pending)}'

# ---- 上传分类测试数据集（image 与 image_vit 各需一份）----
test_paths = {}
for fw in ('image', 'image_vit'):
    _, up = req('POST', '/api/upload_test_data',
                data={'framework': fw},
                files={'file': ('tds.zip', img_zip, 'application/zip')})
    assert up.get('status') == 'success', (fw, up)
    test_paths[fw] = up['path']
print('测试数据集上传 OK')

# ---- 每类：拉取模板 → run_test 一键执行 ----
FRAMEWORK_OF = {
    'image_cnn': 'image', 'image_vit': 'image_vit', 'text_generation': 'text',
    'diffusion': 'diffusion', 'diffusion_edit': 'diffusion_edit',
    'multimodal_stream': 'multimodal',
}
_, lst = req('GET', '/api/list_user_models?framework=all')
by_type = {m['type']: m['name'] for m in lst.get('models', [])}
print('模型归类:', by_type)

all_ok = True
for key, fw in FRAMEWORK_OF.items():
    _, tpl = req('GET', f'/api/test_template/{fw}')
    assert tpl.get('status') == 'success', (fw, tpl.get('message'))
    model_name = by_type.get(key.split('_')[0] if False else
                             {'cnn': 'cnn', 'vit': 'vit', 'text': 'text',
                              'diffusion': 'diffusion',
                              'diffusion_edit': 'diffusion_edit',
                              'multimodal': 'multimodal'}[
                                 {'image_cnn': 'cnn', 'image_vit': 'vit',
                                  'text_generation': 'text', 'diffusion': 'diffusion',
                                  'diffusion_edit': 'diffusion_edit',
                                  'multimodal_stream': 'multimodal'}[key]])
    st2, res = req('POST', '/api/run_test', json_body={
        'framework': fw,
        'model_name': model_name,
        'test_code': tpl['code'],
        'test_data_path': test_paths.get(fw, ''),
    })
    ok = res.get('status') == 'success'
    all_ok &= ok
    tail = (res.get('output') or '').strip().splitlines()[-1:] or ['']
    print(f"[{'PASS' if ok else 'FAIL'}] 一键测试 {key:20s} "
          f"{str(res.get('error'))[:70] if not ok else tail[0][:70]}")

sys.exit(0 if all_ok else 1)

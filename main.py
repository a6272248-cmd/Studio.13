"""Studio.13 AI gateway. Keep secrets only on this server, never in GitHub Pages."""
import base64
import hmac
import os
import re
import time
from collections import deque
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

TOKEN = os.getenv('REPLICATE_API_TOKEN', '').strip()
ACCESS = os.getenv('STUDIO_ACCESS_KEY', '').strip()
ORIGIN = os.getenv('STUDIO_ORIGIN', '').strip().rstrip('/')
if not ORIGIN:
    # Intentionally deny cross-origin browser calls until configured explicitly.
    ORIGINS = []
else:
    ORIGINS = [ORIGIN]
API = 'https://api.replicate.com/v1'
LIMIT = int(os.getenv('STUDIO_HOURLY_LIMIT', '10'))
MAX_SIZE = 6 * 1024 * 1024
app = FastAPI(title='Studio.13 AI Gateway', docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=['GET', 'POST'], allow_headers=['Content-Type', 'X-Studio-Key'])
recent = deque()
owned_jobs: dict[str, float] = {}

class JobRequest(BaseModel):
    image: str = Field(max_length=9_000_000)
    mode: str
    prompt: str = Field(default='', max_length=2500)


def authorize(key: str | None):
    if not ACCESS:
        raise HTTPException(503, detail='STUDIO_ACCESS_KEY is not configured')
    if not key or not hmac.compare_digest(key, ACCESS):
        raise HTTPException(401, detail='رمز الوصول غير صحيح')


def check_usage():
    now = time.time()
    while recent and recent[0] < now - 3600:
        recent.popleft()
    if len(recent) >= LIMIT:
        raise HTTPException(429, detail='تم تجاوز الحد المحدد للطلبات في الساعة')
    recent.append(now)


def decode_image(data: str):
    match = re.fullmatch(r'data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)', data)
    if not match:
        raise HTTPException(400, detail='اختر ملف PNG أو JPG أو WEBP صالحًا')
    try:
        raw = base64.b64decode(match.group(2), validate=True)
    except Exception:
        raise HTTPException(400, detail='ترميز الصورة غير صالح')
    if not raw or len(raw) > MAX_SIZE:
        raise HTTPException(413, detail='يجب أن يكون حجم الصورة أقل من 6MB')
    return raw, match.group(1)


def check_status(r: httpx.Response):
    if r.is_error:
        raise HTTPException(502, detail=f'حدث خطأ من مزود AI ({r.status_code}): {r.text[:220]}')
    return r.json()


def auth_headers():
    return {'Authorization': f'Bearer {TOKEN}'}


@app.get('/api/health')
def health(x_studio_key: str | None = Header(None)):
    authorize(x_studio_key)
    return {'ok': True, 'ai_ready': bool(TOKEN), 'provider': 'Replicate', 'hourly_limit': LIMIT}


@app.post('/api/jobs')
async def create_job(request: JobRequest, x_studio_key: str | None = Header(None)):
    authorize(x_studio_key)
    if not TOKEN:
        raise HTTPException(503, detail='REPLICATE_API_TOKEN غير مضبوط على الخادم')
    if request.mode not in {'prompt', 'upscale2', 'upscale4', 'face'}:
        raise HTTPException(400, detail='نوع المعالجة غير مدعوم')
    if request.mode == 'prompt' and not request.prompt.strip():
        raise HTTPException(400, detail='وصف التعديل مطلوب')
    raw, mime = decode_image(request.image)
    check_usage()
    # Files API uses a short-lived provider-hosted URL, avoiding large inline data URLs.
    files = {'content': ('studio13-input.' + mime.split('/')[-1], raw, mime)}
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            uploaded = check_status(await client.post(API + '/files', headers=auth_headers(), files=files))
            file_url = uploaded.get('urls', {}).get('get')
            if not file_url:
                raise HTTPException(502, detail='لم يرجع مزود AI رابط الصورة المرفوعة')
            if request.mode == 'prompt':
                model = 'google/nano-banana'
                input_data = {'prompt': request.prompt, 'image_input': [file_url], 'output_format': 'jpg'}
            elif request.mode == 'face':
                model = 'tencentarc/gfpgan'
                input_data = {'img': file_url, 'scale': 2, 'version': 'v1.4'}
            else:
                model = 'nightmareai/real-esrgan'
                input_data = {'image': file_url, 'scale': 4 if request.mode == 'upscale4' else 2, 'face_enhance': False}
            # Official model route for supported models; no untrusted model IDs from client.
            created = check_status(await client.post(API + '/models/' + model + '/predictions',
                headers={**auth_headers(), 'Content-Type': 'application/json', 'Prefer': 'wait=2'},
                json={'input': input_data}))
    except httpx.HTTPError:
        raise HTTPException(502, detail='تعذر الوصول إلى خادم مزود الذكاء الاصطناعي')
    pid = created.get('id')
    if not pid or not re.fullmatch('[a-zA-Z0-9]+', pid):
        raise HTTPException(502, detail='لم يرجع المزود معرّف عملية صالحًا')
    owned_jobs[pid] = time.time()
    return {'id': pid, 'status': created.get('status', 'starting')}


def owned(pid: str):
    if not re.fullmatch('[a-zA-Z0-9]{5,80}', pid) or pid not in owned_jobs:
        raise HTTPException(404, detail='العملية غير موجودة أو انتهت صلاحيتها')
    if time.time() - owned_jobs[pid] > 3600:
        owned_jobs.pop(pid, None)
        raise HTTPException(410, detail='انتهت صلاحية العملية')


async def provider_job(pid: str):
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            r = await client.get(API + '/predictions/' + pid, headers=auth_headers())
            return check_status(r)
    except httpx.HTTPError:
        raise HTTPException(502, detail='تعذر التحقق من نتيجة المعالجة')


@app.get('/api/jobs/{pid}')
async def get_job(pid: str, x_studio_key: str | None = Header(None)):
    authorize(x_studio_key)
    owned(pid)
    result = await provider_job(pid)
    return {'id': pid, 'status': result.get('status'), 'error': str(result.get('error') or '')[:300]}


@app.get('/api/jobs/{pid}/result')
async def job_result(pid: str, x_studio_key: str | None = Header(None)):
    authorize(x_studio_key)
    owned(pid)
    job = await provider_job(pid)
    if job.get('status') != 'succeeded':
        raise HTTPException(409, detail='الصورة لم تكتمل بعد')
    output = job.get('output')
    url = output[0] if isinstance(output, list) and output else output
    if not isinstance(url, str):
        raise HTTPException(502, detail='لم يرجع المحرك صورة')
    host = urlparse(url)
    if host.scheme != 'https' or not (host.hostname == 'replicate.delivery' or (host.hostname or '').endswith('.replicate.delivery')):
        raise HTTPException(502, detail='عنوان نتيجة غير مسموح')
    try:
        async with httpx.AsyncClient(timeout=40, follow_redirects=False) as client:
            resp = await client.get(url)
            if resp.is_error or len(resp.content) > 20 * 1024 * 1024:
                raise HTTPException(502, detail='نتيجة الصورة غير متاحة أو كبيرة جدًا')
            content_type = resp.headers.get('content-type', 'image/png').split(';')[0]
            if content_type not in ('image/png', 'image/jpeg', 'image/webp'):
                raise HTTPException(502, detail='المزود أرجع صيغة غير متوقعة')
            return Response(resp.content, media_type=content_type, headers={'Cache-Control': 'no-store'})
    except httpx.HTTPError:
        raise HTTPException(502, detail='تعذر تنزيل الصورة الناتجة')

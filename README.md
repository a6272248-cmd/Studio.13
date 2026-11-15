# Studio.13 AI gateway

Run on Render as a **Python** web service, with **Root Directory** set to `ai-server`.

Build command: `pip install -r requirements.txt`

Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Environment variables:
- `STUDIO_ORIGIN=https://a6272248-cmd.github.io`
- `STUDIO_ACCESS_KEY` = unique random key; configure matching key in the Studio.13 app
- `REPLICATE_API_TOKEN` = your own Replicate token, entered on Render only (required for actual AI inference)
- `STUDIO_HOURLY_LIMIT=10` (optional)

Do NOT commit `.env` secrets. A Render free web service is only a light API relay and is not itself a GPU AI model. AI inference may be paid. Avoid using private images until you trust the service's handling of photos.

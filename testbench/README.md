# Firewatch Testbench Guide

This guide is for a first-time tester. Follow the steps in order.

## 1. Prerequisites

- Windows or macOS
- Internet connection
- Python `3.10`, `3.11`, or `3.12`
- Terminal:
  - Windows: PowerShell
  - macOS: Terminal (bash/zsh)

## 2. Create and activate a virtual environment

Pick a specific Python version in the supported range.

- Windows PowerShell (example uses Python 3.11):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
```

- macOS bash/zsh (example uses Python 3.11):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

If activation is blocked in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure environment variables

Create `.env`:

- Windows:

```powershell
Copy-Item .env.example .env
```

- macOS:

```bash
cp .env.example .env
```

Edit `.env` and replace placeholders where needed:

```env
OPENAI_API_KEY=your_openai_api_key_here
TELEGRAM_BOT_TOKEN=8610841747:AAHEbtxFdZ28VTlN0t5VoTZn5vxUXS9j1VU
TELEGRAM_CHAT_ID=optional_fallback_chat_id_here
TELEGRAM_USERNAME=your_telegram_username_without_at
```

What to change explicitly:

- Replace `OPENAI_API_KEY` with your own key if you want OpenAI context classification. (Recommended)
- Leave `OPENAI_API_KEY` empty if you want local-only mode.
- Replace `TELEGRAM_USERNAME` with your own Telegram username (without `@`) if you want escalation messages in your chat. (Recommended)
- Replace `TELEGRAM_CHAT_ID` only if you want to use a specific fallback chat id.

## 5. Telegram one-time setup (Recommended)

1. Open Telegram and find `FIREWATCH_BOTBOT`.
2. Send `/start`.
3. Send one normal message (for example: `hello`).
4. Confirm `TELEGRAM_USERNAME` in `.env` is your exact username without `@`.

## 6. Prepare database

```bash
python manage.py migrate
```

## 7. Start server

```bash
python manage.py runserver
```

Open `http://127.0.0.1:8000/`.

## 8. Run dashboard test flow

1. Click `Start Scenario`.
2. Wait for processing.
   Expected timing:
   - Classification speed depends on device.
   - Allow up to about 60 seconds on slower machines.
   - Current default is CPU for compatibility, so slower than hardware-optimized production setup.
3. Open the event and click `View Footage`.
4. Check the risk screen (`No Fire Risk`, `Elevated Risk`, `Hazard`, or `Emergency`).
5. If available, click `Escalate To Authorities`.
6. Check for telegram notification if initialised in step 5

## 9. Performing an Admin Reset

If you would like to run the simulation again, please click the admin button in the top right corner of the page and click reset events before navigating to the home page again.

## 10. Optional manual classifier command

Use your own video path by replacing `<path_to_video.mp4>`:

```bash
python classification/analyze_video.py \
  --video <path_to_video.mp4> \
  --weights classification_model.pt \
  --start-seconds 0 \
  --analyze-seconds 10 \
  --sample-fps 2 \
  --conf 0.25 \
  --results-dir results \
  --run-label test_run_0000_0010 \
  --camera-id cam_01 \
  --location-type warehouse \
  --device cpu
```

## 11. Troubleshooting

- `ModuleNotFoundError`:
  - Ensure venv is activated, then run `pip install -r requirements.txt`.
- OpenAI not used:
  - Check if `OPENAI_API_KEY` is missing/invalid. Missing key means local-only mode by design.
- Telegram message not received:
  - Ensure you messaged `FIREWATCH_BOTBOT` first.
  - Ensure `TELEGRAM_USERNAME` exactly matches your Telegram username (without `@`).

## 12. Stop server

Press `Ctrl + C` in the server terminal.

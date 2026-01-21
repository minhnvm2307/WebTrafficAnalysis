# Auto-scaling Analysis

## Setup

```bash
uv sync
# or
pip install -r requirements.txt
```

## Dataset

> Download raw data from https://drive.google.com/drive/folders/1tAEIObd25p8JeqFVLS2ef70uIuHLykxX

```
# 1. Add 'train.txt' and 'test.txt' to /dataset folder
# 2. Run parsing raw data
python main.py
```

**Train rows:** 2.934.961

**Sample:**

| request_src | timestamp | method | dest_path | http_type | status | bytes |
|------|-----------|--------|-----|-----------|--------|-------|
| 199.72.81.55 | 01/Jul/1995:00:00:01 -0400 | GET | /history/apollo/ | HTTP/1.0 | 200 | 6245 |
| unicomp6.unicomp.net | 01/Jul/1995:00:00:06 -0400 | GET | /shuttle/countdown/ | HTTP/1.0 | 200 | 3985 |
| 199.120.110.21 | 01/Jul/1995:00:00:09 -0400 | GET | /shuttle/missions/sts-73/mission-sts-73.html | HTTP/1.0 | 200 | 4085 |
| burger.letters.com | 01/Jul/1995:00:00:11 -0400 | GET | /shuttle/countdown/liftoff.html | HTTP/1.0 | 304 | 0 |
| 199.120.110.21 | 01/Jul/1995:00:00:11 -0400 | GET | /shuttle/missions/sts-73/sts-73-patch-small.gif | HTTP/1.0 | 200 | 4179 |
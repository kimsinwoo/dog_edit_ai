#!/usr/bin/env bash
cd "$(dirname "$0")"
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7000

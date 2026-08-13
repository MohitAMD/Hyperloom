#!/bin/bash
set +e
echo "=== import current_platform, CLEAN env ==="
env -i PATH=/usr/local/bin:/usr/bin HOME=/root /usr/bin/python3.12 -c "from vllm.platforms import current_platform; print('OK', current_platform.get_device_name())" 2>&1 | tail -20
echo
echo "=== grep vllm for logger scope= calls ==="
grep -rnE "\.(debug|info|warning|error|warning_once|info_once)\([^)]*scope=" /usr/local/lib/python3.12/dist-packages/vllm/ 2>/dev/null | head -10
echo
echo "=== who defines _log with scope / setLoggerClass ==="
grep -rnE "def _log|setLoggerClass|scope" /usr/local/lib/python3.12/dist-packages/vllm/logger.py 2>/dev/null | head -20

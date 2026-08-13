#!/bin/bash
# Phase C-2: (re)launch the Recipe 9 container on the current node, detached.
# Run this ON THE ALLOCATED NODE (via srun). Idempotent-ish: removes a prior
# phaseC2 container of the same name first.
set +e
IMG_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glm5.1-fp8-disagg-mi300x-fromscratch-aiter017.tar
IMG=glm5.1-fp8-disagg:mi300x-fromscratch-aiter017
NAME=phaseC2_glm

echo "=== host: $(hostname) ==="
echo "=== ensure image present ==="
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "loading image from $IMG_TAR ..."
  docker load -i "$IMG_TAR"
else
  echo "image already present"
fi
docker image inspect "$IMG" --format 'id={{.Id}}' 2>/dev/null

echo "=== remove stale container $NAME if any ==="
docker rm -f "$NAME" >/dev/null 2>&1

echo "=== run container detached (override entrypoint so it stays up) ==="
docker run -d --name "$NAME" \
  --entrypoint bash \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --shm-size 256G --network host \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -e HOME=/home/mdeopuja \
  -e VLLM_HANDSHAKE_TIMEOUT_MINS=30 \
  -v /shared_inference:/shared_inference \
  -v /home/mdeopuja:/home/mdeopuja \
  -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
  "$IMG" -c 'sleep infinity'
echo "run rc=$?"
sleep 3
echo "=== container status ==="
docker ps --filter "name=$NAME" --format '{{.ID}} {{.Image}} {{.Status}} {{.Names}}'
echo "=== apply image handshake-timeout patch (honor VLLM_HANDSHAKE_TIMEOUT_MINS) ==="
docker exec "$NAME" bash -lc '
  F=/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py
  if grep -q "VLLM_HANDSHAKE_TIMEOUT_MINS" "$F"; then echo "already patched"; else
    cp "$F" "$F.orig"
    sed -i "s/^HANDSHAKE_TIMEOUT_MINS = 5\$/HANDSHAKE_TIMEOUT_MINS = int(__import__(\"os\").environ.get(\"VLLM_HANDSHAKE_TIMEOUT_MINS\", \"5\"))/" "$F"
    python3 -c "import py_compile; py_compile.compile(\"$F\", doraise=True); print(\"handshake patch OK\")"
  fi
  grep -n "HANDSHAKE_TIMEOUT_MINS =" "$F" | head -2'
echo "=== gpu visibility in container ==="
docker exec "$NAME" bash -lc 'python3 -c "import torch;print(\"torch\",torch.__version__,\"gpus\",torch.cuda.device_count())" 2>&1 | tail -1'

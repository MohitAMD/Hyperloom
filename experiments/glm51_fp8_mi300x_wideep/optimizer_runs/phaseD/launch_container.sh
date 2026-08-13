#!/bin/bash
# Phase D: (re)launch the Recipe 9 container on the current node, detached.
# Run this ON THE ALLOCATED NODE (via srun). Removes a prior phaseD container
# of the same name first. Isolated from phaseC / phaseC2 (unique name).
set +e
IMG_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glm5.1-fp8-disagg-mi300x-fromscratch-aiter017.tar
IMG=glm5.1-fp8-disagg:mi300x-fromscratch-aiter017
NAME=phaseD_glm

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

echo "=== run container detached ==="
docker run -d --name "$NAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --shm-size 256G --network host \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -e HOME=/home/mdeopuja \
  -v /shared_inference:/shared_inference \
  -v /home/mdeopuja:/home/mdeopuja \
  -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
  --entrypoint /usr/bin/sleep \
  "$IMG" infinity
echo "run rc=$?"
sleep 2
echo "=== container status ==="
docker ps --filter "name=$NAME" --format '{{.ID}} {{.Image}} {{.Status}} {{.Names}}'
echo "=== gpu visibility in container ==="
docker exec "$NAME" bash -lc 'rocm-smi --showproductname 2>/dev/null | grep -c "Card" ; python3 -c "import torch;print(\"torch\",torch.__version__,\"gpus\",torch.cuda.device_count())" 2>/dev/null'

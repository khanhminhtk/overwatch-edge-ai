# Kubernetes Layout (Production-Ready, Learn-by-Doing)

## Muc tieu
- Moi service la 1 container trong 1 pod.
- Tach `base` va `overlays` de tai su dung cho dev/prod.
- San sang CI/CD: image tag ro rang, secret/config tach rieng.

## Cau truc
- `base/`: manifest dung chung cho moi moi truong.
- `overlays/dev/`: patch phuc vu hoc va test nhanh.
- `overlays/prod/`: patch cho van hanh production.

## Trien khai
1. Build va push image service:
```bash
docker build -f backend_nexus/dockerfile -t ghcr.io/<org>/backend-nexus-minio-service:<tag> .
docker push ghcr.io/<org>/backend-nexus-minio-service:<tag>
```
2. Sua image trong overlay (`patch-images.yaml`).
3. Cap nhat secrets truoc khi apply:
- `minio-credentials`
- `postgres-credentials`
- `minio-service-secrets`
- `grafana-admin`
- `minio-service-tls`
4. Deploy:
```bash
kubectl apply -k k8s/overlays/dev
# hoac
kubectl apply -k k8s/overlays/prod
```

## Kiem tra
```bash
kubectl -n overwatch get pods
kubectl -n overwatch get svc
kubectl -n overwatch rollout status deploy/minio-service
```

## Luu y production
- Nen thay MinIO/Postgres/Kafka self-host bang managed service khi co dieu kien.
- TLS cert trong `minio-service-tls` phai la cert that.
- Them NetworkPolicy, PodDisruptionBudget, HPA theo load thuc te.

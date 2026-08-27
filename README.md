# Kubernetes Project: AI RAG App as Microservices

**A free, fully local Retrieval-Augmented Generation (RAG) application deployed on Kubernetes (Minikube), using Ollama instead of paid OpenAI APIs.**

Reference architecture: `networknuts/k8s-ai-rag-app`
Environment: Ubuntu (local machine), Minikube (Docker driver)

---

## 1. Architecture Overview

The application follows a microservices architecture:

- **Frontend:** React (Vite) + NGINX
- **Backend:** Two FastAPI microservices — **Ingestor Service** (PDF ingestion) and **Query Service** (question answering)
- **Vector Database:** Qdrant
- **AI Models:** Ollama running locally (`nomic-embed-text` for embeddings, `llama3.2` for the LLM) — **100% free, no paid API keys**
- **Networking:** MetalLB (bare-metal LoadBalancer emulation)
- **TLS:** cert-manager with a self-signed ClusterIssuer
- **Observability:** metrics-server, Prometheus, Grafana, custom app metrics via a Prometheus `ServiceMonitor`

**Key adaptation from the reference project:** the original design used paid OpenAI embeddings and GPT models. Every place that called OpenAI was replaced with a call to a local Ollama instance running on the host machine, reachable from inside Minikube via `http://host.minikube.internal:11434`.

---

## 2. Prerequisites Setup

### 2.1 Verify existing tools

```bash
docker --version && python3 --version && node --version && npm --version && kubectl version --client && minikube version
```

### 2.2 Install kubectl

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client
```

### 2.3 Install Minikube

```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
minikube version
```

### 2.4 Create the project directory structure

```bash
mkdir -p ~/k8s-ai-rag-app/{applications/rag-app/{ingestor-service,query-service,rag-ui-react},infrastructure/{metal,certs,monitoring}}
```

---

## 3. Minikube Cluster Setup

```bash
minikube start --driver=docker --cpus=4 --memory=6144 --disk-size=30g
kubectl get nodes
kubectl get pods -A
```

> **Note on reboots:** Minikube stops whenever the host machine restarts. To resume, simply run `minikube start` again — no data is lost, and pods self-heal automatically. Run `minikube status` any time to check.

---

## 4. Ollama Setup (Free Embeddings + LLM)

### 4.1 Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 4.2 Pull the required models

```bash
ollama pull nomic-embed-text   # embeddings model (~270MB)
ollama pull llama3.2           # LLM for answering questions (~2GB)
```

### 4.3 Expose Ollama to the network (so Minikube can reach it)

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
echo -e "[Service]\nEnvironment=\"OLLAMA_HOST=0.0.0.0:11434\"" | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
sudo systemctl status ollama --no-pager
```

### 4.4 Verify connectivity from inside Minikube

```bash
minikube ssh -- curl -s http://host.minikube.internal:11434/api/tags
```

This should return a JSON list of the two pulled models — confirming pods inside the cluster can reach Ollama on the host.

---

## 5. Qdrant Vector Database

```yaml
# applications/rag-app/manifests/ns.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-app
```

```yaml
# applications/rag-app/manifests/qdrant.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
  namespace: rag-app
spec:
  replicas: 1
  selector:
    matchLabels: { app: qdrant }
  template:
    metadata:
      labels: { app: qdrant }
    spec:
      containers:
        - name: qdrant
          image: qdrant/qdrant
          ports: [{ containerPort: 6333 }]
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant
  namespace: rag-app
spec:
  selector: { app: qdrant }
  ports: [{ port: 6333, targetPort: 6333 }]
```

```bash
kubectl apply -f applications/rag-app/manifests/ns.yaml
kubectl apply -f applications/rag-app/manifests/qdrant.yaml
kubectl wait --for=condition=Ready pod -l app=qdrant -n rag-app --timeout=120s
```

---

## 6. Ingestor Service (PDF → Chunks → Embeddings → Qdrant)

**`app.py`** (core logic) — loads a PDF, splits it into chunks, embeds them with Ollama, and stores them in Qdrant:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

QDRANT_URL = "http://qdrant:6333"
OLLAMA_BASE_URL = "http://host.minikube.internal:11434"

@app.post("/ingest")
async def ingest_pdf(file: UploadFile):
    # save file, load with PyPDFLoader, split into chunks
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    QdrantVectorStore.from_documents(documents=chunks, embedding=embeddings,
                                      url=QDRANT_URL, collection_name="learning_vectors")
    return {"status": "indexed", "chunks": len(chunks)}
```

**`Dockerfile`:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build inside Minikube's Docker daemon (no registry needed)

```bash
eval $(minikube docker-env)
cd applications/rag-app/ingestor-service
docker build -t rag-ingestor:1.0 .
```

### Deployment manifest (key points: `imagePullPolicy: Never`, Ollama URL as env var)

```yaml
containers:
  - name: ingestor
    image: rag-ingestor:1.0
    imagePullPolicy: Never
    env:
      - { name: QDRANT_URL, value: http://qdrant:6333 }
      - { name: OLLAMA_BASE_URL, value: http://host.minikube.internal:11434 }
```

```bash
kubectl apply -f applications/rag-app/manifests/ingestor.yaml
kubectl wait --for=condition=Ready pod -l app=ingestor -n rag-app --timeout=60s
```

### Verified end-to-end

```bash
kubectl port-forward -n rag-app svc/ingestor 8000:8000 &
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ingest -F "file=@test.pdf"
# → {"status":"indexed","chunks":1}
```

Confirmed in Qdrant directly:

```bash
kubectl port-forward -n rag-app svc/qdrant 6333:6333 &
curl -s http://localhost:6333/collections/learning_vectors | python3 -m json.tool
# → points_count: 1, vector size: 768 (nomic-embed-text dimension)
```

---

## 7. Query Service (Question → Similarity Search → LLM Answer)

**`app.py`** — embeds the question, retrieves the closest chunks from Qdrant, and asks `llama3.2` to answer, with Prometheus metrics built in:

```python
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from prometheus_client import Counter, Histogram, generate_latest

QUERY_COUNT = Counter("rag_queries_total", "Total number of queries received")
QUERY_LATENCY = Histogram("rag_query_latency_seconds", "Time taken to answer a query")

@app.post("/query")
async def query(request: QueryRequest):
    QUERY_COUNT.inc()
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings, collection_name="learning_vectors", url=QDRANT_URL)
    results = vector_store.similarity_search(request.question, k=3)
    llm = ChatOllama(model="llama3.2", base_url=OLLAMA_BASE_URL)
    response = llm.invoke(f"Answer based only on:\n\n{context}\n\nQuestion: {request.question}")
    return {"answer": response.content, "sources_used": len(results)}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

Service files created:

### Build and deploy

```bash
eval $(minikube docker-env)
cd applications/rag-app/query-service
docker build -t rag-query:1.0 .
kubectl apply -f applications/rag-app/manifests/query.yaml
```

> Important lesson learned: the `query` **Service** object must carry the label `app: query` in its own `metadata.labels` (not just in the pod selector) for Prometheus's `ServiceMonitor` to discover it later. Missing this caused the target to be silently dropped (see §10).

### Verified end-to-end

```bash
kubectl port-forward -n rag-app svc/query 8001:8001 &
curl -X POST http://localhost:8001/query -H "Content-Type: application/json" \
  -d '{"question": "What is Kubernetes?"}'
# → {"answer":"Kubernetes is a container orchestration platform...","sources_used":1}
```

---

## 8. React UI (Frontend)

Built with Vite + React, `axios` for HTTP calls, served by NGINX which also reverse-proxies API calls to the backend services (so the browser never needs to know internal cluster DNS names):

```nginx
# nginx.conf
server {
    listen 80;
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }
    location /api/ingestor/ { proxy_pass http://ingestor:8000/; }
    location /api/query/    { proxy_pass http://query:8001/; }
}
```

Multi-stage Dockerfile (Node build stage → lightweight NGINX runtime stage):

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json .
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

```bash
eval $(minikube docker-env)
cd applications/rag-app/rag-ui-react
docker build -t rag-ui:1.0 .
kubectl apply -f applications/rag-app/manifests/ui.yaml   # type: LoadBalancer
```

Initial load:

### Full end-to-end test — PDF upload + question answering, from the browser

This confirmed the entire pipeline working purely through the browser: **React UI → NGINX proxy → Ingestor/Query services → Qdrant + Ollama**.

---

## 9. MetalLB (LoadBalancer IPs on Bare-Metal / Minikube)

### 9.1 Install MetalLB (native CRD-based version, not the outdated Minikube addon)

```bash
kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.14.8/config/manifests/metallb-native.yaml
kubectl wait --namespace metallb-system --for=condition=ready pod --all --timeout=120s
```

### 9.2 Configure the IP address pool (based on the Docker bridge subnet)

```bash
docker network inspect minikube | grep Subnet   # → 192.168.49.0/24
```

```yaml
# infrastructure/metal/deployment.yaml
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata: { name: public-ip-pool, namespace: metallb-system }
spec:
  addresses: ["192.168.49.200-192.168.49.220"]
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata: { name: public-l2, namespace: metallb-system }
spec:
  ipAddressPools: ["public-ip-pool"]
```

```bash
kubectl apply -f infrastructure/metal/deployment.yaml
```

### 9.3 Enable strict ARP mode (required for MetalLB L2 mode)

```bash
kubectl get configmap kube-proxy -n kube-system -o yaml | \
  sed -e "s/strictARP: false/strictARP: true/" | kubectl apply -f - -n kube-system
kubectl rollout restart daemonset kube-proxy -n kube-system
```

### 9.4 Known limitation encountered

After configuration, the `rag-ui` LoadBalancer service was correctly assigned an external IP (`192.168.49.200`), but ARP requests for that IP went unanswered:

Investigation (checking `speaker` logs, `ServiceL2Status` CRDs, and packet captures) traced this to a **known, unresolved upstream MetalLB issue affecting single-node clusters** (GitHub Issue #2314 — "Responding to ARP request does not work on single-node cluster"). All configuration (IPAddressPool, L2Advertisement, strict ARP, `hostNetwork: true`, `NET_RAW` capability) was verified correct; this is a platform limitation of running MetalLB L2 mode on a single-node Minikube cluster, not a project misconfiguration.

**Resolution:** the `LoadBalancer`-type Service manifest is kept as-is (architecturally correct, and would work unmodified on real cloud infrastructure or a multi-node bare-metal cluster). For local access on this single-node setup, `kubectl port-forward` is used instead:

```bash
kubectl port-forward -n rag-app svc/rag-ui 8090:80 &
# → browser: http://localhost:8090
```

---

## 10. cert-manager (TLS Certificates)

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.2/cert-manager.yaml
kubectl wait --namespace cert-manager --for=condition=Ready pod --all --timeout=120s
```

### Self-signed ClusterIssuer

```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata: { name: selfsigned-issuer }
spec:
  selfSigned: {}
```

### Certificate for the UI service

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata: { name: rag-ui-tls, namespace: rag-app }
spec:
  secretName: rag-ui-tls-secret
  duration: 2160h
  renewBefore: 360h
  issuerRef: { name: selfsigned-issuer, kind: ClusterIssuer }
  commonName: rag-ui.local
  dnsNames: [rag-ui.local, localhost]
```

```bash
kubectl apply -f infrastructure/certs/clusterissuer.yaml
kubectl apply -f infrastructure/certs/argocd-cert.yaml
kubectl get certificate -n rag-app   # → READY: True
```

Verified `ca.crt`, `tls.crt`, and `tls.key` were all populated in the resulting Kubernetes Secret.

---

## 11. Monitoring: metrics-server, Prometheus, Grafana

### 11.1 metrics-server (requires a patch for Minikube's self-signed kubelet certs)

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch deployment metrics-server -n kube-system --type='json' \
  -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'
```

```bash
kubectl top nodes
kubectl top pods -n rag-app
```

### 11.2 Helm + kube-prometheus-stack (Prometheus + Grafana + AlertManager)

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
kubectl create namespace monitoring
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack --namespace monitoring
```

> A transient `ImagePullBackOff` on the Prometheus image (TLS handshake timeout against `quay.io`) was resolved by manually pulling the image once inside Minikube's Docker daemon and deleting the pod to force a retry against the now-cached image.

### 11.3 Access Grafana

```bash
kubectl get secrets kube-prometheus-stack-grafana -n monitoring \
  -o jsonpath="{.data.admin-password}" | base64 -d
kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 47000:80 &
# → http://localhost:47000  (user: admin)
```

---

## 12. Custom Application Metrics (ServiceMonitor)

To have Prometheus scrape the Query Service's custom metrics (`rag_queries_total`, `rag_query_latency_seconds`), a `ServiceMonitor` was created:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: rag-query-monitor
  namespace: rag-app
  labels:
    release: kube-prometheus-stack   # required so the Prometheus Operator discovers it
spec:
  selector:
    matchLabels: { app: query }
  endpoints:
    - port: query-port
      path: /metrics
      interval: 15s
```

```bash
kubectl apply -f applications/rag-app/manifests/servicemonitor.yaml
```

### Troubleshooting note

Initially the target showed as "dropped" in Prometheus despite matching pod labels. The cause: the `query` **Service** object itself needs the label `app: query` in its `metadata.labels` (ServiceMonitor label-selectors match against the Service object, not the underlying Pods). Adding `labels: { app: query }` to the Service manifest and re-applying it fixed the discovery.

### Final verification — querying the custom metric directly from Prometheus

```bash
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 47091:9090 &
curl -s "http://localhost:47091/api/v1/query?query=rag_queries_total" | python3 -m json.tool
```

---

## 13. Final Status Summary

| Component | Status | Notes |
|---|---|---|
| Minikube cluster | ✅ | Docker driver, 4 CPU / 6GB RAM |
| Ollama (embeddings + LLM) | ✅ | 100% free/local — no API keys |
| Qdrant | ✅ | Vector DB, 768-dim vectors |
| Ingestor Service | ✅ | Tested end-to-end (PDF → Qdrant) |
| Query Service | ✅ | Tested end-to-end + Prometheus metrics |
| React UI | ✅ | Working end-to-end from the browser |
| MetalLB | ⚠️ | Configured correctly; single-node ARP limitation (upstream bug), access via `port-forward` |
| cert-manager + TLS | ✅ | Self-signed ClusterIssuer + Certificate |
| metrics-server | ✅ | `kubectl top` working |
| Prometheus | ✅ | Scraping all standard + custom targets |
| Grafana | ✅ | Login and dashboard verified |
| Custom app metrics | ✅ | `rag_queries_total`, `rag_query_latency_seconds` |

### Reusable operational commands

```bash
# Resume after a reboot
minikube start

# Rebuild an image inside Minikube's Docker daemon
eval $(minikube docker-env)
docker build -t <image>:<tag> .

# Access any internal service locally
kubectl port-forward -n <namespace> svc/<service-name> <local-port>:<service-port> &
```

---

*Prepared as course documentation for: "Kubernetes Project – AI RAG App As Microservices" (adapted from `networknuts/k8s-ai-rag-app` to use free/local Ollama models instead of paid OpenAI APIs).*

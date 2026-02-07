# SVDash Compute Server

Motore di calcolo SVD/PCA che gira sul Mac remoto (Apple MPS).

## Setup (sul Mac)

```bash
cd compute_server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Avvio

```bash
uvicorn server:app --host 127.0.0.1 --port 8002 --reload
```

## Tunnel SSH (dal Linux locale)

```bash
ssh -p 26996 -L 8002:127.0.0.1:8002 edoardo.tedesco@openport.io
```

## Test

```bash
# Health check
curl http://localhost:8002/health

# Decomposizione
curl -X POST http://localhost:8002/api/decompose \
  -F "file=@test_image.jpg" \
  -G -d "n_components=50" \
  | python -m json.tool | head -50
```

## Endpoints

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/health` | GET | Health check + info device |
| `/api/decompose` | POST | Decomposizione completa (upload immagine) |
| `/api/reconstruct` | POST | Ricostruzione a diverso k (usa cache) |
| `/api/components_visual` | POST | Visualizzazione componenti singole |
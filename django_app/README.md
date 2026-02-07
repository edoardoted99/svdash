# SVDash — Django App

Web frontend for SVD/PCA image decomposition analysis.

## Setup (Linux locale)

```bash
cd django_app
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python manage.py migrate
python manage.py runserver 8000
```

## Prerequisiti

1. Tunnel SSH attivo: `ssh -p 26996 -L 8002:127.0.0.1:8002 edoardo.tedesco@openport.io`
2. Compute server in esecuzione sul Mac (porta 8002)

## Struttura

```
django_app/
├── manage.py
├── svdash/              # Django project
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── core/                # Main app
│   ├── models.py        # ImageAnalysis
│   ├── views.py         # Upload, dashboard, history, compare
│   ├── services.py      # Compute server communication
│   ├── forms.py
│   ├── urls.py
│   └── templates/core/
│       ├── base.html
│       ├── index.html       # Upload page
│       ├── analysis.html    # Dashboard
│       ├── history.html     # Past analyses
│       ├── compare.html     # Side-by-side comparison
│       └── partials/        # HTMX fragments
│           ├── reconstruction.html
│           └── components.html
├── static/css/style.css
└── requirements.txt
```

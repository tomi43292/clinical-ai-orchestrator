# 🏥 Clinical AI Triage Engine

> AI-powered clinical triage API that integrates **Computer Vision** and **Large Language Models** on Azure for intelligent medical image analysis and patient prioritization.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Deployment](#deployment)

---

## Overview

The **Clinical AI Triage Engine** solves the challenge of clinical workflow optimization by automating the analysis of medical images and clinical notes to provide faster, AI-assisted patient triage decisions.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| 🔬 **Computer Vision Analysis** | Automated detection of anomalies in medical images (X-Ray, CT, MRI) |
| 🧠 **LLM Clinical Reasoning** | LangChain-orchestrated analysis correlating imaging findings with clinical notes |
| 📊 **Priority Classification** | Automated triage with CRITICAL / HIGH / MEDIUM / LOW priority levels |
| 🔍 **Clinical Search** | Full-text search across patients, notes, and triage reports |
| 📋 **Structured Reports** | AI-generated triage assessments with findings, actions, and follow-up questions |

---

## Architecture

The application follows **Hexagonal Architecture** (Ports & Adapters), ensuring clean separation of concerns and testability.

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
│  FastAPI Endpoints · Middleware · Dependency Injection       │
│  /api/v1/patients · /medical-images · /triage · /search     │
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                        │
│  Use Cases · DTOs · Orchestration Logic                     │
│  RegisterPatient · AnalyzeImage · GenerateTriage            │
├─────────────────────────────────────────────────────────────┤
│                      Domain Layer                           │
│  Entities · Repository Interfaces · Service Interfaces      │
│  Patient · MedicalImage · TriageReport · ClinicalNote       │
├───────────────────────┬─────────────────────────────────────┤
│   Infrastructure      │        External Services            │
│                       │                                     │
│  PostgreSQL (SQLAlch) │  Azure Computer Vision              │
│  Cosmos DB (Motor)    │  Azure OpenAI + LangChain           │
│  Alembic Migrations   │  Azure Cognitive Search             │
│                       │  Azure ML                           │
└───────────────────────┴─────────────────────────────────────┘
```

### Data Flow

```
Patient Data ──► Register Patient ──► PostgreSQL
                                          │
Medical Image ──► Upload & Analyze ───► Azure CV ──► Analysis Results
                                          │
Clinical Note ──► Persist ──► PostgreSQL  │
                                          ▼
                              GenerateTriage Use Case
                                          │
                              ┌───────────┴───────────┐
                              │  LangChain Pipeline    │
                              │  (Prompt → LLM → JSON) │
                              └───────────┬───────────┘
                                          │
                                          ▼
                              TriageReport ──► Cosmos DB
                                           ──► Cognitive Search
```

---

## Technology Stack

| Category | Technology |
|----------|-----------|
| **Framework** | FastAPI 0.104+ (async ASGI) |
| **ORM** | SQLAlchemy 2.0 (async) |
| **Relational DB** | PostgreSQL 16 |
| **Document DB** | Cosmos DB (MongoDB API) / MongoDB 7 |
| **AI Orchestration** | LangChain 0.1+ |
| **LLM** | Azure OpenAI (GPT-4) |
| **Computer Vision** | Azure Cognitive Services (Image Analysis 4.0) |
| **Search** | Azure Cognitive Search |
| **ML Platform** | Azure Machine Learning |
| **Containerization** | Docker, Docker Compose |
| **Migration** | Alembic |
| **Testing** | pytest, pytest-asyncio, httpx |
| **Linting** | Ruff |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- (Optional) Azure subscription for production services

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/clinical-ai-orchestrator.git
cd clinical-ai-orchestrator

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings (mock mode works without Azure credentials)
```

Key settings for local development:
```env
APP_ENV=development
SERVICE_MODE=mock          # Uses mock services — no Azure needed
DATABASE_URL=postgresql+asyncpg://clinical_user:clinical_secret@localhost:5432/clinical_triage
COSMOS_DB_CONNECTION_STRING=mongodb://localhost:27017
```

### 3. Start Services

```bash
# Option A: Docker Compose (recommended)
docker-compose up -d

# Option B: Manual (requires local PostgreSQL + MongoDB)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Verify

```bash
# Health check
curl http://localhost:8000/health

# Swagger UI
open http://localhost:8000/docs

# ReDoc
open http://localhost:8000/redoc
```

---

## API Documentation

### Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Application health check |
| `POST` | `/api/v1/patients` | Register a new patient |
| `GET` | `/api/v1/patients` | List patients (paginated + search) |
| `GET` | `/api/v1/patients/{id}` | Get patient details |
| `POST` | `/api/v1/patients/{id}/medical-images` | Upload & analyze image |
| `GET` | `/api/v1/patients/{id}/medical-images` | List patient images |
| `POST` | `/api/v1/patients/{id}/clinical-notes` | Add clinical note |
| `GET` | `/api/v1/patients/{id}/clinical-notes` | List clinical notes |
| `POST` | `/api/v1/patients/{id}/triage` | Generate triage report |
| `GET` | `/api/v1/patients/{id}/triage/reports` | List triage reports |
| `GET` | `/api/v1/search/clinical` | Search clinical documents |

### Example: Full Triage Workflow

```bash
# 1. Register a patient
curl -X POST http://localhost:8000/api/v1/patients \
  -H "Content-Type: application/json" \
  -d '{
    "medical_record_number": "MRN-2024-001",
    "first_name": "María",
    "last_name": "García",
    "date_of_birth": "1985-03-15",
    "gender": "female",
    "allergies": ["Penicillin"],
    "pre_existing_conditions": ["Hypertension"]
  }'

# 2. Upload and analyze an X-Ray image
curl -X POST http://localhost:8000/api/v1/patients/{patient_id}/medical-images \
  -F "file=@chest_xray.png" \
  -F "modality=xray" \
  -F "body_region=chest"

# 3. Add clinical notes
curl -X POST http://localhost:8000/api/v1/patients/{patient_id}/clinical-notes \
  -H "Content-Type: application/json" \
  -d '{
    "author": "Dr. Martínez",
    "content": "Patient presents with persistent dry cough and fever (38.5°C).",
    "note_type": "admission"
  }'

# 4. Generate AI triage report
curl -X POST http://localhost:8000/api/v1/patients/{patient_id}/triage \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## Project Structure

```
clinical-ai-orchestrator/
├── app/
│   ├── api/                          # API Layer
│   │   ├── dependencies.py           # Dependency injection container
│   │   ├── middleware.py              # Logging, error handling
│   │   └── v1/                       # Versioned endpoints
│   │       ├── patients.py
│   │       ├── medical_images.py
│   │       ├── clinical_notes.py
│   │       ├── triage.py
│   │       ├── search.py
│   │       └── router.py
│   ├── application/                  # Application Layer
│   │   ├── dto/                      # Data Transfer Objects
│   │   │   ├── patient_dto.py
│   │   │   ├── image_dto.py
│   │   │   └── triage_dto.py
│   │   └── use_cases/                # Business logic orchestration
│   │       ├── register_patient.py
│   │       ├── analyze_image.py
│   │       └── generate_triage.py
│   ├── core/                         # Cross-cutting concerns
│   │   └── config.py                 # Pydantic Settings
│   ├── domain/                       # Domain Layer (pure business logic)
│   │   ├── entities/                 # Domain entities
│   │   │   ├── patient.py
│   │   │   ├── medical_image.py
│   │   │   ├── clinical_note.py
│   │   │   └── triage_report.py
│   │   ├── repositories/            # Abstract repository interfaces
│   │   │   ├── patient_repository.py
│   │   │   └── report_repository.py
│   │   ├── services/                 # Abstract service interfaces
│   │   │   ├── vision_service.py
│   │   │   ├── llm_service.py
│   │   │   └── search_service.py
│   │   └── exceptions.py            # Domain exceptions
│   └── infrastructure/               # Infrastructure Layer
│       ├── azure/                    # Azure service adapters
│       │   ├── vision_client.py
│       │   ├── openai_client.py
│       │   ├── search_client.py
│       │   ├── ml_client.py
│       │   └── mock_services.py      # Dev/test mocks
│       ├── database/
│       │   ├── postgresql/           # Relational data
│       │   │   ├── models.py
│       │   │   ├── session.py
│       │   │   └── repositories.py
│       │   └── cosmosdb/             # Document data
│       │       ├── client.py
│       │       └── repositories.py
│       └── langchain/                # AI orchestration
│           ├── prompts.py
│           ├── chains.py
│           └── triage_agent.py
├── alembic/                          # Database migrations
├── tests/                            # Test suite
├── Dockerfile                        # Production container
├── docker-compose.yml                # Local development
├── pyproject.toml                    # Project metadata
├── requirements.txt                  # Dependencies
└── .env.example                      # Environment template
```

---

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test suite
pytest tests/test_domain_entities.py
pytest tests/test_mock_services.py
pytest tests/test_api_endpoints.py

# Run with coverage
pytest --cov=app --cov-report=html
```

---

## Deployment

### Docker Production Build

```bash
docker build -t clinical-ai-orchestrator:latest .
docker run -p 8000:8000 --env-file .env clinical-ai-orchestrator:latest
```

### Azure Container Apps / Cloud Run

The application is containerized and ready for deployment to:
- **Azure Container Apps** (recommended)
- **Google Cloud Run**
- **AWS ECS/Fargate**

Set `SERVICE_MODE=azure` and configure all Azure credentials in the environment for production deployments.

---

## License

This project is licensed under the MIT License.

---

*Built with ❤️ using FastAPI, LangChain, and Azure AI Services*

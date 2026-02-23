# HazelJS ML Starter

A comprehensive, real-world example demonstrating **@hazeljs/ml** for machine learning in Node.js. This starter implements production-ready ML APIs with model training, prediction, batch processing, and metrics tracking.

## Features

- **SentimentClassifier** – Text sentiment (positive/negative/neutral) for reviews and feedback
- **EmbeddingSentimentClassifier** – Offline LLM embeddings (all-MiniLM-L6-v2) + centroid-based classification
- **SpamClassifier** – Binary spam/ham classification for emails, SMS, content moderation
- **IntentClassifier** – Multi-class intent routing for chatbots and support tickets
- **REST API** – Predict, batch predict, train, metrics, and model listing
- **Model Registry** – Versioned model registration and lookup
- **Training Pipeline** – Data preprocessing (normalize, filter) via `PipelineService`
- **Metrics Tracking** – Model evaluation for A/B testing and monitoring

## Quick Start

```bash
# Install dependencies (from hazeljs repo root)
cd hazeljs-ml-starter
npm install

# Build
npm run build

# Start the server
npm start

# Or run in dev mode with hot reload
npm run dev
```

The API runs at **http://localhost:3000**.

## Models

| Model | Use Case | Labels |
|-------|----------|--------|
| `sentiment-classifier` | Reviews, feedback (bag-of-words) | positive, negative, neutral |
| `embedding-sentiment-classifier` | Reviews, feedback (LLM embeddings) | positive, negative, neutral |
| `spam-classifier` | Email, SMS, moderation | spam, ham |
| `intent-classifier` | Chatbots, support routing | refund, bug_report, feature_request, greeting, farewell, general_inquiry, complaint, other |

Use the `model` parameter in request bodies to switch between models (default: `sentiment-classifier`).

## API Endpoints

### Single Prediction

```bash
# Sentiment (default)
curl -X POST http://localhost:3000/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing! I love it."}'

# Spam
curl -X POST http://localhost:3000/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Win a free iPhone now!", "model": "spam-classifier"}'

# Intent
curl -X POST http://localhost:3000/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I want a refund.", "model": "intent-classifier"}'

# Embedding-based sentiment (uses offline LLM embeddings; first call downloads model)
curl -X POST http://localhost:3000/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!", "model": "embedding-sentiment-classifier"}'
```

**Sentiment response:**
```json
{
  "result": {
    "sentiment": "positive",
    "confidence": 0.85,
    "scores": { "positive": 2.5, "negative": 0.2, "neutral": 0.3 }
  }
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:3000/ml/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible experience."], "model": "sentiment-classifier"}'
```

### Train Model

```bash
# Sentiment
curl -X POST http://localhost:3000/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentiment-classifier",
    "samples": [
      {"text": "Love this!", "label": "positive"},
      {"text": "Hate it.", "label": "negative"},
      {"text": "Its fine.", "label": "neutral"}
    ]
  }'

# Spam
curl -X POST http://localhost:3000/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spam-classifier",
    "samples": [{"text": "Win free money now!", "label": "spam"}, {"text": "Meeting at 3pm", "label": "ham"}]
  }'

# Intent (supports custom labels: alphanumeric + underscore)
curl -X POST http://localhost:3000/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "model": "intent-classifier",
    "samples": [{"text": "I want a refund", "label": "refund"}, {"text": "Hi there!", "label": "greeting"}]
  }'
```

### List Models

```bash
curl http://localhost:3000/ml/models
```

### Get Metrics

```bash
curl "http://localhost:3000/ml/metrics?model=sentiment-classifier"
curl "http://localhost:3000/ml/metrics?model=embedding-sentiment-classifier"
curl "http://localhost:3000/ml/metrics?model=spam-classifier"
curl "http://localhost:3000/ml/metrics?model=intent-classifier"
```

## Project Structure

```
hazeljs-ml-starter/
├── src/
│   ├── index.ts              # Bootstrap & server
│   ├── app.module.ts         # App module with MLModule
│   ├── models/
│   │   ├── sentiment.model.ts        # SentimentClassifier (bag-of-words)
│   │   ├── embedding-sentiment.model.ts  # EmbeddingSentimentClassifier (LLM embeddings)
│   │   ├── spam.classifier.ts        # SpamClassifier
│   │   └── intent.classifier.ts      # IntentClassifier
│   ├── controllers/
│   │   └── ml.controller.ts    # REST API
│   ├── ml/
│   │   └── ml.bootstrap.ts     # Training pipeline setup
│   ├── data/
│   │   ├── sample-training-data.json  # Sentiment samples
│   │   ├── sample-spam-data.json       # Spam/ham samples
│   │   └── sample-intent-data.json     # Intent samples
│   └── scripts/
│       ├── train-with-sample-data.ts   # CLI training (bag-of-words sentiment)
│       └── train-embedding-model.ts    # CLI training (embedding sentiment)
├── package.json
├── tsconfig.json
└── README.md
```

## Model Implementation

The `SentimentClassifier` uses `@hazeljs/ml` decorators:

```typescript
@Model({
  name: 'sentiment-classifier',
  version: '1.0.0',
  framework: 'custom',
})
@Injectable()
export class SentimentClassifier {
  @Train({ pipeline: 'sentiment-preprocessing', epochs: 1 })
  async train(data: SentimentTrainingData): Promise<TrainingResult> {
    // Build word frequency maps from labeled samples
    // ...
  }

  @Predict({ batch: true })
  async predict(input: unknown): Promise<SentimentPrediction> {
    // Score text against learned vocabularies
    // ...
  }
}
```

## Training Pipeline

The `PipelineService` registers preprocessing steps used before training:

1. **normalize** – Trim, lowercase text; normalize labels
2. **filter-invalid** – Remove samples with empty text or invalid labels

Register pipelines in `ml.bootstrap.ts`; they run automatically when `TrainerService.train()` is invoked.

## Programmatic Training

Train without the HTTP API:

```bash
# Bag-of-words sentiment model
npm run train:sample

# Embedding-based sentiment (downloads all-MiniLM-L6-v2 ~90MB on first run)
npm run train:embedding
```

These load `src/data/sample-training-data.json` and train the respective model. Useful for:

- Initial model setup
- CI/CD training jobs
- Batch retraining

## Extending to Production

1. **TensorFlow.js / ONNX** – Replace the bag-of-words logic with neural models; keep the same `@Model`, `@Train`, `@Predict` interface.

2. **Model Persistence** – Save/load trained weights to disk (e.g. `models/` directory) in `train()` and on model construction.

3. **MetricsService** – Call `metricsService.recordEvaluation()` after training/validation to support A/B tests and monitoring.

4. **PipelineService** – Add richer ETL (tokenization, feature extraction) or integrate with `@hazeljs/data`.

## Environment

Copy `.env.example` to `.env` and adjust:

```
PORT=3000
LOG_LEVEL=info
```

## License

Apache-2.0

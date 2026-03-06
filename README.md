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

## ML Decorators

This starter uses the three **@hazeljs/ml** decorators so the registry and services can discover models without manual wiring.

| Decorator | Where | Purpose |
|-----------|--------|---------|
| **`@Model`** | Class | Registers the model with `name`, `version`, `framework`. Required so `TrainerService` and `PredictorService` can find it. |
| **`@Train`** | One method | Marks the method that trains the model. `TrainerService.train(modelName, data)` calls it. Options: `pipeline`, `batchSize`, `epochs`. |
| **`@Predict`** | One method | Marks the method that runs inference. `PredictorService.predict(modelName, input)` calls it. Options: `batch`, `endpoint`. |

**Rules:** One `@Model` per class; exactly one `@Train()` and one `@Predict()` method per model. Use `@Injectable()` from `@hazeljs/core` so the app can create the model.

**In this repo:** See `src/models/sentiment.model.ts`, `src/models/spam.classifier.ts`, and `src/models/intent.classifier.ts` for full examples. Run the minimal decorator example:

```bash
npm run example:decorators
```

See [Decorator example](#decorator-example) below for a minimal code walkthrough.

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

Results are returned in the same order as inputs.

```bash
curl -X POST http://localhost:3000/ml/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible experience."], "model": "sentiment-classifier"}'
```

### Evaluate Model

Run evaluation on test data to compute accuracy, F1, precision, and recall:

```bash
curl -X POST http://localhost:3000/ml/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentiment-classifier",
    "testData": [
      {"text": "Love it!", "label": "positive"},
      {"text": "Terrible.", "label": "negative"}
    ]
  }'
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
│   ├── examples/
│   │   └── decorator-example.ts # Minimal @Model / @Train / @Predict (npm run example:decorators)
│   ├── data/
│   │   ├── sample-training-data.json  # Sentiment samples
│   │   ├── sample-spam-data.json       # Spam/ham samples
│   │   └── sample-intent-data.json     # Intent samples
│   └── scripts/
│       ├── train-with-sample-data.ts   # CLI training (inline pipeline)
│       ├── train-embedding-model.ts    # CLI training (embedding sentiment)
│       └── evaluate-with-sample-data.ts # CLI evaluation (MetricsService.evaluate)
├── package.json
├── tsconfig.json
└── README.md
```

## Model Implementation

Every model in this starter follows the same pattern. Example from **SentimentClassifier** (`src/models/sentiment.model.ts`):

```typescript
@Model({
  name: 'sentiment-classifier',
  version: '1.0.0',
  framework: 'custom',
  description: 'Text sentiment classification (positive/negative/neutral)',
  tags: ['nlp', 'sentiment', 'production'],
})
@Injectable()
export class SentimentClassifier {
  @Train({ pipeline: 'sentiment-preprocessing', epochs: 1, batchSize: 32 })
  async train(data: SentimentTrainingData): Promise<TrainingResult> {
    // Build word frequency maps from labeled samples
    // ...
  }

  @Predict({ batch: true, endpoint: '/predict' })
  async predict(input: unknown): Promise<SentimentPrediction> {
    // Score text against learned vocabularies
    // ...
  }
}
```

**SpamClassifier** and **IntentClassifier** use the same three decorators with different `name`, `pipeline`, and logic; see `src/models/spam.classifier.ts` and `src/models/intent.classifier.ts`.

## Decorator example

A minimal runnable example that uses only the decorators and the registry (no HTTP server) lives in `src/examples/decorator-example.ts`. Run it with:

```bash
npm run example:decorators
```

It defines a small classifier with `@Model`, `@Train`, and `@Predict`, registers it with `registerMLModel`, then calls `TrainerService.train()` and `PredictorService.predict()`. Use it as a template for adding a new model to this app.

## Training Pipeline

The `PipelineService` registers preprocessing steps used before training:

1. **normalize** – Trim, lowercase text; normalize labels
2. **filter-invalid** – Remove samples with empty text or invalid labels

Register pipelines in `ml.bootstrap.ts`; they run automatically when `TrainerService.train()` is invoked.

## Programmatic Training & Evaluation

Train and evaluate without the HTTP API:

```bash
# Bag-of-words sentiment (uses inline PipelineService.run(data, steps))
npm run train:sample

# Embedding-based sentiment (downloads all-MiniLM-L6-v2 ~90MB on first run)
npm run train:embedding

# Evaluate sentiment model on sample data (MetricsService.evaluate())
npm run evaluate:sample
```

These load `src/data/sample-training-data.json`. Useful for:

- Initial model setup
- CI/CD training jobs
- Batch retraining

## Extending to Production

1. **TensorFlow.js / ONNX** – Replace the bag-of-words logic with neural models; keep the same `@Model`, `@Train`, `@Predict` interface.

2. **Model Persistence** – Save/load trained weights to disk (e.g. `models/` directory) in `train()` and on model construction.

3. **MetricsService** – Use `metricsService.evaluate(modelName, testData)` to compute accuracy, F1, precision, recall on test data. Results are auto-recorded for A/B tests and monitoring.

4. **PipelineService** – Add richer ETL (tokenization, feature extraction) or integrate with `@hazeljs/data`.

## Environment

Copy `.env.example` to `.env` and adjust:

```
PORT=3000
LOG_LEVEL=info
```

## License

Apache-2.0

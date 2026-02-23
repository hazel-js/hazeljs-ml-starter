# Building Production ML APIs in Node.js with HazelJS

*A comprehensive guide to the HazelJS ML Starter—decorator-based machine learning, sentiment analysis, and scalable inference in TypeScript*

---

## Introduction

Machine learning in Node.js has matured significantly. With [TensorFlow.js](https://www.tensorflow.org/js), [ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-javascript.html), and a growing ecosystem, JavaScript/TypeScript developers can now build and serve ML models without leaving their preferred runtime.

[HazelJS](https://hazeljs.com) is a modern, decorator-first Node.js framework that brings this capability to production with **@hazeljs/ml**—a module for model registration, training pipelines, inference, and lifecycle management. In this post, we’ll walk through the [HazelJS ML Starter](https://github.com/hazel-js/hazeljs)—a real-world sentiment analysis API that demonstrates the full stack.

---

## Why ML in Node.js?

Traditionally, ML has lived in Python. For many teams, however, Node.js is the primary backend. Running inference in the same process as your API reduces latency, simplifies deployment, and keeps your stack consistent. Use cases like:

- **Sentiment analysis** for reviews and support tickets  
- **Fraud detection** on transactional data  
- **Recommendation models** in e-commerce  
- **Content moderation** for user-generated content  

can all be served directly from a Node.js API. [@hazeljs/ml](https://www.npmjs.com/package/@hazeljs/ml) is designed for exactly this pattern: register models, train (or load) them, and expose them via REST or any transport.

---

## What is HazelJS?

[HazelJS](https://hazeljs.com) is a TypeScript-first Node.js framework inspired by NestJS and Fastify. It provides:

- **Decorator-based APIs** for controllers, services, and modules  
- **Dependency injection** and modular architecture  
- **Integrated packages** for [AI](https://www.npmjs.com/package/@hazeljs/ai), [gRPC](https://www.npmjs.com/package/@hazeljs/grpc), [Kafka](https://www.npmjs.com/package/@hazeljs/kafka), [Prisma](https://www.npmjs.com/package/@hazeljs/prisma), and more  
- **Production-ready defaults** for health checks, shutdown, and observability  

The [@hazeljs/ml](https://www.npmjs.com/package/@hazeljs/ml) package adds machine learning primitives on top of this foundation. You can explore the full framework on the [HazelJS homepage](https://hazeljs.com) and [GitHub repository](https://github.com/hazel-js/hazeljs).

---

## Architecture of the HazelJS ML Starter

The [HazelJS ML Starter](https://github.com/hazel-js/hazeljs) implements **three widely-used models** plus shared infrastructure:

| Component | Responsibility |
|-----------|----------------|
| **SentimentClassifier** | Sentiment (positive/negative/neutral) for reviews and feedback |
| **SpamClassifier** | Binary spam/ham for email, SMS, content moderation |
| **IntentClassifier** | Multi-class intent routing for chatbots and support |
| **MLModule** | Registers models and ML services via `MLModule.forRoot()` |
| **MLController** | REST endpoints for predict, batch predict, train, metrics |
| **PipelineService** | Preprocessing pipeline (normalize, filter) per model |
| **ModelRegistry** | Versioned model storage and lookup |
| **MetricsService** | Evaluation metrics for A/B testing and monitoring |

The [starter repository](https://github.com/hazel-js/hazeljs) includes a working example you can clone and run locally.

---

## Getting Started

### Prerequisites

- Node.js 18+
- [HazelJS](https://hazeljs.com) and [@hazeljs/ml](https://www.npmjs.com/package/@hazeljs/ml) installed

### Installation

```bash
# Clone or navigate to the starter
cd hazeljs-ml-starter

# Install dependencies
npm install

# Build
npm run build

# Start the server
npm start
```

The API is available at `http://localhost:3000`. For development with hot reload, use `npm run dev`. Full setup instructions are in the [starter README](https://github.com/hazel-js/hazeljs/blob/main/README.md).

---

## The @Model, @Train, and @Predict Decorators

[@hazeljs/ml](https://www.npmjs.com/package/@hazeljs/ml) uses three core decorators to define ML models:

### @Model

Registers a class as an ML model with metadata for the [Model Registry](https://github.com/hazel-js/hazeljs/tree/main/packages/ml/src/registry):

```typescript
@Model({
  name: 'sentiment-classifier',
  version: '1.0.0',
  framework: 'custom',
  description: 'Text sentiment classification',
  tags: ['nlp', 'production'],
})
@Injectable()
export class SentimentClassifier {
  // ...
}
```

The [ML package documentation](https://github.com/hazel-js/hazeljs/tree/main/packages/ml) covers all options.

### @Train

Marks the training method. Options include pipeline name, batch size, and epochs:

```typescript
@Train({ pipeline: 'sentiment-preprocessing', epochs: 1, batchSize: 32 })
async train(data: SentimentTrainingData): Promise<TrainingResult> {
  // Build word weights from labeled samples
  return { accuracy: 0.95, loss: 0.05 };
}
```

The [Train decorator source](https://github.com/hazel-js/hazeljs/blob/main/packages/ml/src/decorators/train.decorator.ts) shows the full signature.

### @Predict

Marks the inference method for real-time and batch prediction:

```typescript
@Predict({ batch: true, endpoint: '/predict' })
async predict(input: unknown): Promise<SentimentPrediction> {
  // Score text, return sentiment + confidence
  return { sentiment: 'positive', confidence: 0.92, scores: {...} };
}
```

The [Predict decorator](https://github.com/hazel-js/hazeljs/blob/main/packages/ml/src/decorators/predict.decorator.ts) is used by both [PredictorService](https://github.com/hazel-js/hazeljs/blob/main/packages/ml/src/inference/predictor.service.ts) and [BatchService](https://github.com/hazel-js/hazeljs/blob/main/packages/ml/src/inference/batch.service.ts).

---

## REST API Walkthrough

### Single Prediction

Use the `model` parameter to switch between models (default: `sentiment-classifier`).

```bash
# Sentiment
curl -X POST http://localhost:3000/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing! I love it."}'

# Spam
curl -X POST http://localhost:3000/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Win a free iPhone!", "model": "spam-classifier"}'

# Intent
curl -X POST http://localhost:3000/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I want a refund.", "model": "intent-classifier"}'
```

Sentiment response:

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
  -d '{"texts": ["Great!", "Terrible.", "Its okay."], "batchSize": 32}'
```

The [BatchService](https://github.com/hazel-js/hazeljs/blob/main/packages/ml/src/inference/batch.service.ts) handles batching and concurrency.

### Training

```bash
curl -X POST http://localhost:3000/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"text": "Love this!", "label": "positive"},
      {"text": "Hate it.", "label": "negative"},
      {"text": "Its fine.", "label": "neutral"}
    ]
  }'
```

Training data flows through the [PipelineService](https://github.com/hazel-js/hazeljs/blob/main/packages/ml/src/training/pipeline.service.ts) for normalization and validation.

### Metrics and Model Listing

```bash
# Model evaluation metrics
curl "http://localhost:3000/ml/metrics?model=sentiment-classifier"

# List registered models
curl http://localhost:3000/ml/models

# Model versions
curl http://localhost:3000/ml/models/sentiment-classifier/versions
```

[MetricsService](https://github.com/hazel-js/hazeljs/blob/main/packages/ml/src/evaluation/metrics.service.ts) supports A/B testing and comparison of model versions.

---

## Training Pipeline and Data Preparation

Before training, data can be preprocessed via the [PipelineService](https://github.com/hazel-js/hazeljs/blob/main/packages/ml/src/training/pipeline.service.ts). In the starter, we register a `sentiment-preprocessing` pipeline in [ml.bootstrap.ts](https://github.com/hazel-js/hazeljs/blob/main/hazeljs-ml-starter/src/ml/ml.bootstrap.ts):

1. **normalize** – Trim and lowercase text; normalize labels  
2. **filter-invalid** – Remove samples with empty text or invalid labels  

This pattern extends to richer ETL. [@hazeljs/data](https://www.npmjs.com/package/@hazeljs/data) provides [@Pipeline](https://github.com/hazel-js/hazeljs/tree/main/packages/data), [@Transform](https://github.com/hazel-js/hazeljs/blob/main/packages/data/src/decorators/transform.decorator.ts), and [@Validate](https://github.com/hazel-js/hazeljs/blob/main/packages/data/src/decorators/validate.decorator.ts) for complex data pipelines. See the [Data package](https://github.com/hazel-js/hazeljs/tree/main/packages/data) for integration examples.

---

## Programmatic Training

You can train outside the HTTP API using the [train-with-sample-data script](https://github.com/hazel-js/hazeljs/blob/main/hazeljs-ml-starter/src/scripts/train-with-sample-data.ts):

```bash
npm run train:sample
```

This loads `src/data/sample-training-data.json`, trains the model, and runs a quick prediction. Useful for:

- Initial model setup  
- CI/CD training jobs  
- Batch retraining  

The [starter data file](https://github.com/hazel-js/hazeljs/blob/main/hazeljs-ml-starter/src/data/sample-training-data.json) includes 20 labeled samples that typically yield ~95% training accuracy.

---

## Extending to Production

### 1. Integrate TensorFlow.js or ONNX

The starter uses a bag-of-words model for clarity. For production NLP, you can:

- Load a [TensorFlow.js](https://www.tensorflow.org/js) or [ONNX](https://onnxruntime.ai/docs/get-started/with-javascript.html) model  
- Keep the same `@Model`, `@Train`, `@Predict` interface  
- Swap the internal logic without changing controllers or clients  

The [@hazeljs/ml types](https://github.com/hazel-js/hazeljs/blob/main/packages/ml/src/ml.types.ts) support `framework: 'tensorflow' | 'onnx' | 'custom'`.

### 2. Model Persistence

Implement save/load in `train()` and on construction:

- Save weights to `./models/` or object storage  
- Load on startup when the model is first resolved  
- Use [ModelRegistry](https://github.com/hazel-js/hazeljs/blob/main/packages/ml/src/registry/model.registry.ts) for versioning  

### 3. Metrics and A/B Testing

Use [MetricsService.recordEvaluation()](https://github.com/hazel-js/hazeljs/blob/main/packages/ml/src/evaluation/metrics.service.ts) after validation runs. [compareVersions()](https://github.com/hazel-js/hazeljs/blob/main/packages/ml/src/evaluation/metrics.service.ts) helps compare model versions for rollout decisions.

### 4. Data Pipelines

Combine [@hazeljs/ml](https://www.npmjs.com/package/@hazeljs/ml) with [@hazeljs/data](https://www.npmjs.com/package/@hazeljs/data) for:

- ETL with [@Pipeline](https://github.com/hazel-js/hazeljs/tree/main/packages/data), [@Transform](https://github.com/hazel-js/hazeljs/blob/main/packages/data/src/decorators/transform.decorator.ts), [@Validate](https://github.com/hazel-js/hazeljs/blob/main/packages/data/src/decorators/validate.decorator.ts)  
- Streaming with [@Stream](https://github.com/hazel-js/hazeljs/blob/main/packages/data/src/decorators/stream.decorator.ts) and Flink-style pipelines  
- Schema validation via [Schema](https://github.com/hazel-js/hazeljs/blob/main/packages/data/src/schema/schema.ts)  

---

## Summary

The [HazelJS ML Starter](https://github.com/hazel-js/hazeljs) shows how to build a production-oriented ML API in Node.js with:

- **Three widely-used models**: [SentimentClassifier](https://github.com/hazel-js/hazeljs) (reviews), [SpamClassifier](https://github.com/hazel-js/hazeljs) (moderation), [IntentClassifier](https://github.com/hazel-js/hazeljs) (chatbots)  
- **Decorator-based models** via [@hazeljs/ml](https://www.npmjs.com/package/@hazeljs/ml)  
- **REST API** for prediction, batch inference, and training  
- **Training pipeline** for data preprocessing  
- **Model registry and metrics** for versioning and monitoring  

You can use it as a template for sentiment analysis, spam detection, intent routing, or any ML workload that fits the same pattern.

---

## Links and Resources

| Resource | URL |
|----------|-----|
| **HazelJS** | [https://hazeljs.com](https://hazeljs.com) |
| **HazelJS GitHub** | [https://github.com/hazel-js/hazeljs](https://github.com/hazel-js/hazeljs) |
| **@hazeljs/ml on npm** | [https://www.npmjs.com/package/@hazeljs/ml](https://www.npmjs.com/package/@hazeljs/ml) |
| **@hazeljs/core on npm** | [https://www.npmjs.com/package/@hazeljs/core](https://www.npmjs.com/package/@hazeljs/core) |
| **@hazeljs/data on npm** | [https://www.npmjs.com/package/@hazeljs/data](https://www.npmjs.com/package/@hazeljs/data) |
| **@hazeljs/ai on npm** | [https://www.npmjs.com/package/@hazeljs/ai](https://www.npmjs.com/package/@hazeljs/ai) |
| **TensorFlow.js** | [https://www.tensorflow.org/js](https://www.tensorflow.org/js) |
| **ONNX Runtime JS** | [https://onnxruntime.ai/docs/get-started/with-javascript.html](https://onnxruntime.ai/docs/get-started/with-javascript.html) |
| **ML package source** | [packages/ml](https://github.com/hazel-js/hazeljs/tree/main/packages/ml) |
| **Data package source** | [packages/data](https://github.com/hazel-js/hazeljs/tree/main/packages/data) |
| **Open Collective** | [https://opencollective.com/hazeljs](https://opencollective.com/hazeljs) |

---

*This blog post was created for the HazelJS ML Starter. For questions and contributions, visit the [HazelJS GitHub repository](https://github.com/hazel-js/hazeljs) or [community](https://hazeljs.com).*

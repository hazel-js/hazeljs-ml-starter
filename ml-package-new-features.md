# @hazeljs/ml: Evaluate, Inline Pipelines, Guaranteed Order & Decorator Docs

**Machine learning in Node.js just got clearer and more capable.** The `@hazeljs/ml` package now includes programmatic evaluation, inline training pipelines, guaranteed batch result ordering, and full documentation for the three ML decorators—plus updated starters and examples so you can go from zero to a working model in minutes.

---

## What’s New

### 1. **MetricsService.evaluate()** — Evaluate Models on Test Data

You can now run **programmatic evaluation** on any registered model. Pass test samples and get accuracy, F1, precision, and recall without writing evaluation loops yourself.

```typescript
const evaluation = await metricsService.evaluate('sentiment-classifier', testData, {
  metrics: ['accuracy', 'f1', 'precision', 'recall'],
  labelKey: 'label',           // key in each sample for ground truth
  predictionKey: 'sentiment',  // key in model output (or auto-detect)
});
// evaluation.metrics → { accuracy, precision, recall, f1Score }
// Result is automatically recorded for A/B testing and history
```

`MetricsService` needs `PredictorService` and `ModelRegistry` (injected when you use `MLModule`). It runs predictions for each test sample, compares outputs to labels, and computes metrics. Custom label and prediction keys are supported so it works with sentiment, spam, intent, or any classifier that returns a single predicted class.

**In hazeljs-ml-starter:** A new **POST /ml/evaluate** endpoint accepts `model`, `testData`, and optional `metrics`. The starter also includes an **evaluate:sample** script that evaluates the sentiment model on sample data from the repo.

---

### 2. **PipelineService.run(data, steps)** — Inline Pipelines, No Registration

Training pipelines no longer have to be registered by name. You can run an **ad-hoc pipeline** by passing data and steps directly.

```typescript
const steps = [
  { name: 'normalize', transform: (d) => ({ ...d, text: d.text.toLowerCase() }) },
  { name: 'filter', transform: (d) => d.text.length > 0 ? d : null },
];
const processed = await pipelineService.run(data, steps);
await trainerService.train('my-model', processed);
```

Named pipelines still work: `pipelineService.run('pipeline-name', data)`. The new overload is for one-off or script-based workflows where registering a pipeline would be unnecessary.

**In hazeljs-ml-starter:** The **train:sample** script uses this pattern: it loads JSON, builds inline steps (normalize + filter), runs `pipeline.run(rawData, steps)`, then trains the sentiment model—no pipeline registration required.

---

### 3. **BatchService — Results Always Match Input Order**

Batch prediction now **guarantees** that the i-th result corresponds to the i-th input. Internally, each request is tracked by index and results are written into the correct position, so you can rely on order even with concurrency and batching.

```typescript
const results = await batchService.predictBatch('sentiment-classifier', items, {
  batchSize: 32,
  concurrency: 4,
});
// results[i] is the prediction for items[i]
```

No API changes—just a more reliable contract for consumers.

---

### 4. **ML Decorators — Documented and Exemplified**

The three decorators (**@Model**, **@Train**, **@Predict**) are now documented in one place with clear rules and options.

| Decorator   | Applied to   | Purpose |
|------------|--------------|---------|
| **@Model** | Class        | Registers the model with `name`, `version`, `framework`. Required so the registry, TrainerService, and PredictorService can find it. |
| **@Train** | One method   | Marks the method that trains the model. `TrainerService.train(modelName, data)` calls it. Options: `pipeline`, `batchSize`, `epochs`. |
| **@Predict** | One method | Marks the method that runs inference. `PredictorService.predict(modelName, input)` calls it. Options: `batch`, `endpoint`. |

**Rules:** One `@Model` per class; exactly one `@Train` and one `@Predict` method per model. Use `@Injectable()` so the app can construct the model. When you pass model classes to `MLModule.forRoot({ models: [...] })`, the bootstrap discovers the decorated methods and registers the model—no manual wiring.

Documentation and examples have been added in:

- **Package README** ([packages/ml/README.md](https://github.com/hazel-js/hazeljs/blob/main/packages/ml/README.md)) — “ML Decorators” section with option tables and code samples.
- **Docs site** ([hazeljs.com/docs/packages/ml](https://hazeljs.com/docs/packages/ml)) — Same section so the website is the single source of truth.
- **Example app** — `example/src/ml/` with a minimal runnable script (`npm run ml:decorators`) and a README that walks through each decorator.
- **hazeljs-ml-starter** — README section on decorators, plus `src/examples/decorator-example.ts` runnable with `npm run example:decorators`.

---

## Where to See It

- **Full app (REST, train, evaluate, batch):** [hazeljs-ml-starter](https://github.com/hazel-js/hazeljs/tree/main/hazeljs-ml-starter) — sentiment, spam, intent classifiers; POST /ml/evaluate; train and evaluate scripts; decorator example.
- **Minimal decorator-only example:** [example/src/ml](https://github.com/hazel-js/hazeljs/tree/main/example/src/ml) — `npm run ml:decorators` in the example repo.
- **Starter decorator example:** In hazeljs-ml-starter, `npm run example:decorators` runs `src/examples/decorator-example.ts`.

---

## Summary

- **Evaluate:** `metricsService.evaluate(modelName, testData, options)` for accuracy, F1, precision, recall with configurable label/prediction keys.
- **Inline pipelines:** `pipelineService.run(data, steps)` for one-off preprocessing without registering a named pipeline.
- **Batch order:** `batchService.predictBatch(...)` results are guaranteed to match input order.
- **Decorators:** @Model, @Train, @Predict documented in the package README and on the docs site, with runnable examples in the main repo and in hazeljs-ml-starter.

If you’re building ML into a Node or HazelJS app, the new evaluation and pipeline options should simplify your workflows, and the decorator docs and examples should make it easier to add and maintain models.

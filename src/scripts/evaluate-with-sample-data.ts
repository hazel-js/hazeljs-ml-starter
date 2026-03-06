#!/usr/bin/env ts-node
/**
 * Evaluate the sentiment model on test data
 *
 * Usage: npm run evaluate:sample
 *
 * Demonstrates MetricsService.evaluate() - runs predictions on test data
 * and computes accuracy, F1, precision, recall.
 */
import { Container } from '@hazeljs/core';
import {
  ModelRegistry,
  PredictorService,
  MetricsService,
} from '@hazeljs/ml';
import { SentimentClassifier } from '../models';
import * as fs from 'fs';
import * as path from 'path';

async function main(): Promise<void> {
  const container = Container.getInstance();

  const registry = new ModelRegistry();
  container.register(ModelRegistry, registry);

  const predictor = new PredictorService(registry);
  container.register(PredictorService, predictor);

  const metricsService = new MetricsService(registry, predictor);
  container.register(MetricsService, metricsService);

  const model = new SentimentClassifier();
  registry.register({
    metadata: {
      name: 'sentiment-classifier',
      version: '1.0.0',
      framework: 'custom',
    },
    instance: model,
    trainMethod: 'train',
    predictMethod: 'predict',
  });

  const dataPath = path.join(__dirname, '../data/sample-training-data.json');
  if (!fs.existsSync(dataPath)) {
    console.error('Sample data not found at', dataPath);
    process.exit(1);
  }

  const raw = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
  const testData = (raw.samples ?? []).map((s: { text?: string; label?: string }) => ({
    text: (s.text ?? '').toString().trim(),
    label: (s.label ?? 'neutral').toLowerCase(),
  }));

  if (testData.length === 0) {
    console.error('No test samples');
    process.exit(1);
  }

  console.log(`Evaluating sentiment-classifier on ${testData.length} samples...`);

  const evaluation = await metricsService.evaluate('sentiment-classifier', testData, {
    metrics: ['accuracy', 'f1', 'precision', 'recall'],
    labelKey: 'label',
    predictionKey: 'sentiment',
  });

  console.log('\nEvaluation complete!');
  console.log('  Accuracy:', (evaluation.metrics.accuracy ?? 0).toFixed(4));
  console.log('  Precision:', (evaluation.metrics.precision ?? 0).toFixed(4));
  console.log('  Recall:', (evaluation.metrics.recall ?? 0).toFixed(4));
  console.log('  F1 Score:', (evaluation.metrics.f1Score ?? 0).toFixed(4));
  console.log('  Model:', evaluation.modelName + '@' + evaluation.version);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

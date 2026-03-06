#!/usr/bin/env ts-node
/**
 * Train the sentiment model with sample data from JSON
 *
 * Usage: npm run train:sample
 *
 * This script demonstrates programmatic training outside the HTTP API.
 * Uses PipelineService.run(data, steps) for inline preprocessing (no registration).
 * Useful for batch jobs, CI/CD pipelines, or initial model setup.
 */

import { Container } from '@hazeljs/core';
import { TrainerService, ModelRegistry, PipelineService } from '@hazeljs/ml';
import type { PipelineStep } from '@hazeljs/ml';
import { SentimentClassifier } from '../models';
import * as fs from 'fs';
import * as path from 'path';

async function main(): Promise<void> {
  const container = Container.getInstance();

  // Register services and model
  const registry = new ModelRegistry();
  container.register(ModelRegistry, registry);

  const pipelineService = new PipelineService();
  container.register(PipelineService, pipelineService);

  const trainer = new TrainerService(registry);
  container.register(TrainerService, trainer);

  const model = new SentimentClassifier();
  container.register(SentimentClassifier, model);

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

  // Load sample data
  const dataPath = path.join(__dirname, '../data/sample-training-data.json');
  if (!fs.existsSync(dataPath)) {
    console.error('Sample data not found at', dataPath);
    process.exit(1);
  }

  const raw = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
  const rawData = { samples: raw.samples ?? [] };

  if (rawData.samples.length === 0) {
    console.error('No samples in training data');
    process.exit(1);
  }

  // Inline pipeline steps (no registration required)
  const steps: PipelineStep[] = [
    {
      name: 'normalize',
      transform: (data: unknown) => {
        const d = data as { samples?: Array<{ text?: string; label?: string }> };
        if (!d?.samples) return d;
        const normalized = d.samples.map((s) => ({
          text: (s.text ?? '').toString().trim().toLowerCase(),
          label: (s.label ?? 'neutral').toLowerCase(),
        }));
        return { ...d, samples: normalized };
      },
    },
    {
      name: 'filter-invalid',
      transform: (data: unknown) => {
        const d = data as { samples?: Array<{ text: string; label: string }> };
        if (!d?.samples) return d;
        const validLabels = ['positive', 'negative', 'neutral'];
        const filtered = d.samples.filter(
          (s) => s.text?.length > 0 && validLabels.includes(s.label)
        );
        return { ...d, samples: filtered };
      },
    },
  ];

  const trainingData = (await pipelineService.run(rawData, steps)) as { samples: Array<{ text: string; label: string }> };
  console.log(`Training with ${trainingData.samples.length} samples (preprocessed via inline pipeline)...`);

  const result = await trainer.train('sentiment-classifier', trainingData);

  console.log('\nTraining complete!');
  console.log('  Accuracy:', (result.accuracy ?? 0).toFixed(2));
  console.log('  Loss:', (result.loss ?? 0).toFixed(2));
  console.log('  Metrics:', result.metrics);

  // Quick test prediction
  const testText = 'This is an amazing product!';
  const pred = await model.predict({ text: testText });
  console.log('\nTest prediction for "' + testText + '":');
  console.log('  Sentiment:', pred.sentiment);
  console.log('  Confidence:', pred.confidence.toFixed(2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

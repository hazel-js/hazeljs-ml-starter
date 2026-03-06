#!/usr/bin/env ts-node
/**
 * Train the embedding-based sentiment model with sample data
 *
 * Usage: npm run train:embedding
 *
 * Downloads the all-MiniLM-L6-v2 model on first run (~90MB), then
 * embeds samples and computes centroids for centroid-based classification.
 */

import { Container } from '@hazeljs/core';
import { TrainerService, ModelRegistry, PipelineService } from '@hazeljs/ml';
import { EmbeddingSentimentClassifier } from '../models';
import * as fs from 'fs';
import * as path from 'path';

async function main(): Promise<void> {
  const container = Container.getInstance();

  const registry = new ModelRegistry();
  const pipelineService = new PipelineService();

  // Register sentiment preprocessing (reuse from ml.bootstrap logic)
  pipelineService.registerPipeline('sentiment-preprocessing', [
    {
      name: 'normalize',
      transform: (data: unknown): { samples: Array<{ text: string; label: string }> } => {
        const d = data as { samples?: Array<{ text?: string; label?: string }> };
        if (!d?.samples) return d as { samples: Array<{ text: string; label: string }> };
        const normalized = d.samples.map((s) => ({
          text: (s.text ?? '').toString().trim().toLowerCase(),
          label: (s.label ?? 'neutral').toLowerCase(),
        }));
        return { ...d, samples: normalized } as { samples: Array<{ text: string; label: string }> };
      },
    },
    {
      name: 'filter-invalid',
      transform: (data: unknown): { samples: Array<{ text: string; label: string }> } => {
        const d = data as { samples?: Array<{ text: string; label: string }> };
        if (!d?.samples) return d as { samples: Array<{ text: string; label: string }> };
        const validLabels = ['positive', 'negative', 'neutral'];
        const filtered = d.samples.filter(
          (s) => s.text?.length > 0 && validLabels.includes(s.label)
        );
        return { ...d, samples: filtered } as { samples: Array<{ text: string; label: string }> };
      },
    },
  ]);

  container.register(ModelRegistry, registry);
  container.register(PipelineService, pipelineService);

  const trainer = new TrainerService(registry);
  container.register(TrainerService, trainer);

  const model = new EmbeddingSentimentClassifier();
  container.register(EmbeddingSentimentClassifier, model);

  registry.register({
    metadata: {
      name: 'embedding-sentiment-classifier',
      version: '1.0.0',
      framework: 'onnx',
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
  const trainingData = { samples: raw.samples ?? [] };

  if (trainingData.samples.length === 0) {
    console.error('No samples in training data');
    process.exit(1);
  }

  console.log('Loading embedding model (first run downloads ~90MB)...');
  console.log(`Training embedding-sentiment-classifier with ${trainingData.samples.length} samples...`);

  const pipeline = pipelineService.getPipeline('sentiment-preprocessing');
  const preprocessed = pipeline
    ? await pipelineService.run('sentiment-preprocessing', trainingData)
    : trainingData;

  const result = await trainer.train('embedding-sentiment-classifier', preprocessed);

  console.log('\nTraining complete!');
  console.log('  Accuracy:', (result.accuracy ?? 0).toFixed(2));
  console.log('  Loss:', (result.loss ?? 0).toFixed(2));
  console.log('  Metrics:', result.metrics);

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

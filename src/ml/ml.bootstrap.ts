/**
 * ML Bootstrap - Initialize training data pipeline and record baseline metrics
 *
 * Runs on app startup to:
 * 1. Register the sentiment preprocessing pipeline for training data preparation
 * 2. Optionally record initial model evaluation
 */
import { Injectable } from '@hazeljs/core';
import { PipelineService } from '@hazeljs/ml';
import type { TrainingData } from '@hazeljs/ml';
import logger from '@hazeljs/core';

@Injectable()
export class MLBootstrap {
  constructor(private readonly pipelineService: PipelineService) {
    this.registerTrainingPipelines();
    logger.info('ML bootstrap: training pipelines registered');
  }

  private registerTrainingPipelines(): void {
    // Sentiment: positive | negative | neutral
    this.pipelineService.registerPipeline('sentiment-preprocessing', [
      {
        name: 'normalize',
        transform: (data: unknown): TrainingData => {
          const d = data as { samples?: Array<{ text?: string; label?: string }> };
          if (!d?.samples) return d as TrainingData;

          const normalized = d.samples.map((s) => ({
            text: (s.text ?? '').toString().trim().toLowerCase(),
            label: (s.label ?? 'neutral').toLowerCase(),
          }));

          return { ...d, samples: normalized };
        },
      },
      {
        name: 'filter-invalid',
        transform: (data: unknown): TrainingData => {
          const d = data as { samples?: Array<{ text: string; label: string }> };
          if (!d?.samples) return d as TrainingData;

          const validLabels = ['positive', 'negative', 'neutral'];
          const filtered = d.samples.filter(
            (s) =>
              s.text?.length > 0 &&
              validLabels.includes(s.label)
          );

          return { ...d, samples: filtered };
        },
      },
    ]);

    // Spam: spam | ham
    this.pipelineService.registerPipeline('spam-preprocessing', [
      {
        name: 'normalize',
        transform: (data: unknown): TrainingData => {
          const d = data as { samples?: Array<{ text?: string; label?: string }> };
          if (!d?.samples) return d as TrainingData;

          const normalized = d.samples.map((s) => ({
            text: (s.text ?? '').toString().trim().toLowerCase(),
            label: (s.label ?? 'ham').toLowerCase(),
          }));

          return { ...d, samples: normalized };
        },
      },
      {
        name: 'filter-invalid',
        transform: (data: unknown): TrainingData => {
          const d = data as { samples?: Array<{ text: string; label: string }> };
          if (!d?.samples) return d as TrainingData;

          const validLabels = ['spam', 'ham'];
          const filtered = d.samples.filter(
            (s) =>
              s.text?.length > 0 &&
              validLabels.includes(s.label)
          );

          return { ...d, samples: filtered };
        },
      },
    ]);

    // Intent: accepts alphanumeric + underscore labels (refund, bug_report, etc.)
    this.pipelineService.registerPipeline('intent-preprocessing', [
      {
        name: 'normalize',
        transform: (data: unknown): TrainingData => {
          const d = data as { samples?: Array<{ text?: string; label?: string }> };
          if (!d?.samples) return d as TrainingData;

          const normalized = d.samples.map((s) => ({
            text: (s.text ?? '').toString().trim().toLowerCase(),
            label: (s.label ?? 'other').toLowerCase().replace(/\s+/g, '_'),
          }));

          return { ...d, samples: normalized };
        },
      },
      {
        name: 'filter-invalid',
        transform: (data: unknown): TrainingData => {
          const d = data as { samples?: Array<{ text: string; label: string }> };
          if (!d?.samples) return d as TrainingData;

          const filtered = d.samples.filter(
            (s) =>
              s.text?.length > 0 &&
              s.label?.length > 0 &&
              /^[a-z0-9_]+$/.test(s.label)
          );

          return { ...d, samples: filtered };
        },
      },
    ]);
  }
}

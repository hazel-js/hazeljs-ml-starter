/**
 * HazelJS ML Starter - Bootstrap
 *
 * A comprehensive real-world example showcasing @hazeljs/ml:
 * - Sentiment classification model with @Model, @Train, @Predict
 * - REST API for predict, batch predict, train
 * - Model registry, metrics, training pipeline
 *
 * Run: npm run dev
 * API: http://localhost:3000
 */

import { HazelApp } from '@hazeljs/core';
import { AppModule } from './app.module';
import logger from '@hazeljs/core';

const PORT = parseInt(process.env.PORT ?? '3000', 10);

async function bootstrap(): Promise<void> {
  logger.info('Starting HazelJS ML Starter...');

  const app = new HazelApp(AppModule);

  await app.listen(PORT);

  logger.info('');
  logger.info('HazelJS ML Starter running at http://localhost:' + PORT);
  logger.info('');
  logger.info('API Endpoints:');
  logger.info('  POST /ml/predict       - Single sentiment prediction');
  logger.info('  POST /ml/predict/batch - Batch prediction (results in input order)');
  logger.info('  POST /ml/train         - Train model with labeled samples');
  logger.info('  POST /ml/evaluate      - Evaluate model on test data');
  logger.info('  GET  /ml/metrics       - Model metrics');
  logger.info('  GET  /ml/models        - List registered models');
  logger.info('  GET  /health           - Health check');
  logger.info('');
}

bootstrap().catch((err) => {
  logger.error('Failed to start:', err);
  process.exit(1);
});

/**
 * ML Controller - REST API for model operations
 *
 * Endpoints:
 * - POST /ml/predict - Single prediction (model in body)
 * - POST /ml/predict/batch - Batch prediction (results in input order)
 * - POST /ml/train - Train model (model in body)
 * - POST /ml/evaluate - Evaluate model on test data (accuracy, f1, precision, recall)
 * - GET /ml/metrics - Model evaluation metrics
 * - GET /ml/models - List registered models
 *
 * Supported models: sentiment-classifier | embedding-sentiment-classifier | spam-classifier | intent-classifier
 */
import {
  Controller,
  Post,
  Get,
  Body,
  Param,
  Query,
  BadRequestError,
  InternalServerError,
  logger,
} from '@hazeljs/core';
import {
  PredictorService,
  TrainerService,
  MetricsService,
  ModelRegistry,
  BatchService,
  PipelineService,
} from '@hazeljs/ml';

const DEFAULT_MODEL = 'sentiment-classifier';
const MODEL_CONFIG: Record<
  string,
  { pipeline: string; validLabels: string[]; customLabels?: boolean; predictionKey: string }
> = {
  'sentiment-classifier': {
    pipeline: 'sentiment-preprocessing',
    validLabels: ['positive', 'negative', 'neutral'],
    predictionKey: 'sentiment',
  },
  'embedding-sentiment-classifier': {
    pipeline: 'sentiment-preprocessing',
    validLabels: ['positive', 'negative', 'neutral'],
    predictionKey: 'sentiment',
  },
  'spam-classifier': {
    pipeline: 'spam-preprocessing',
    validLabels: ['spam', 'ham'],
    predictionKey: 'label',
  },
  'intent-classifier': {
    pipeline: 'intent-preprocessing',
    validLabels: ['refund', 'bug_report', 'feature_request', 'general_inquiry', 'complaint', 'greeting', 'farewell', 'other'],
    customLabels: true,
    predictionKey: 'intent',
  },
};

@Controller('/ml')
export class MLController {
  constructor(
    private readonly predictorService: PredictorService,
    private readonly trainerService: TrainerService,
    private readonly batchService: BatchService,
    private readonly metricsService: MetricsService,
    private readonly modelRegistry: ModelRegistry,
    private readonly pipelineService: PipelineService
  ) {}

  /**
   * Single prediction - POST /ml/predict
   * Body: { text: string, model?: string, version?: string }
   * Also accepts: message, content, input (aliases for text)
   */
  @Post('/predict')
  async predict(
    @Body() body: { text?: string; message?: string; content?: string; input?: string; model?: string; version?: string }
  ): Promise<{ result: unknown }> {
    const text = (body?.text ?? body?.message ?? body?.content ?? body?.input ?? '').toString();
    if (!text.trim()) {
      throw new BadRequestError(
        'text is required. Send JSON body: { "text": "your input" } or { "message": "..." }'
      );
    }

    const modelName = body?.model ?? DEFAULT_MODEL;
    if (!MODEL_CONFIG[modelName]) {
      throw new BadRequestError(`Unknown model: ${modelName}. Use: ${Object.keys(MODEL_CONFIG).join(', ')}`);
    }

    try {
      const result = await this.predictorService.predict(
        modelName,
        { text },
        body?.version
      );
      return { result };
    } catch (error) {
      logger.error('Prediction failed:', error);
      throw new InternalServerError(
        error instanceof Error ? error.message : 'Prediction failed'
      );
    }
  }

  /**
   * Batch prediction - POST /ml/predict/batch
   * Body: { texts: string[], model?: string, batchSize?: number }
   */
  @Post('/predict/batch')
  async predictBatch(
    @Body() body: { texts?: string[]; model?: string; batchSize?: number }
  ): Promise<{ results: unknown[]; count: number }> {
    const texts = body?.texts ?? [];
    if (!Array.isArray(texts) || texts.length === 0) {
      throw new BadRequestError('texts array is required');
    }

    const modelName = body?.model ?? DEFAULT_MODEL;
    if (!MODEL_CONFIG[modelName]) {
      throw new BadRequestError(`Unknown model: ${modelName}. Use: ${Object.keys(MODEL_CONFIG).join(', ')}`);
    }

    try {
      const inputs = texts.map((text) => ({ text }));
      const results = await this.batchService.predictBatch(
        modelName,
        inputs,
        { batchSize: body?.batchSize ?? 32 }
      );
      return { results, count: results.length };
    } catch (error) {
      logger.error('Batch prediction failed:', error);
      throw new InternalServerError(
        error instanceof Error ? error.message : 'Batch prediction failed'
      );
    }
  }

  /**
   * Train model - POST /ml/train
   * Body: { model?: string, samples: [{ text: string, label: string }] }
   */
  @Post('/train')
  async train(
    @Body() body: { model?: string; samples?: Array<{ text: string; label: string }> }
  ): Promise<{ result: unknown; samplesUsed: number }> {
    const samples = body?.samples ?? [];
    if (!Array.isArray(samples) || samples.length === 0) {
      throw new BadRequestError('samples array is required');
    }

    const modelName = body?.model ?? DEFAULT_MODEL;
    const config = MODEL_CONFIG[modelName];
    if (!config) {
      throw new BadRequestError(`Unknown model: ${modelName}. Use: ${Object.keys(MODEL_CONFIG).join(', ')}`);
    }

    const validated = samples.filter((s) => {
      if (typeof s?.text !== 'string' || typeof s?.label !== 'string') return false;
      if (config.customLabels) {
        return /^[a-z0-9_]+$/.test(s.label.toLowerCase().replace(/\s+/g, '_'));
      }
      return config.validLabels.includes(s.label);
    });

    if (validated.length === 0) {
      throw new BadRequestError(
        `No valid samples (need text and label: ${config.validLabels.join('|')})`
      );
    }

    try {
      const pipeline = this.pipelineService.getPipeline(config.pipeline);
      const trainingData = pipeline
        ? await this.pipelineService.run(config.pipeline, { samples: validated })
        : { samples: validated };

      const result = await this.trainerService.train(modelName, trainingData);
      return { result, samplesUsed: validated.length };
    } catch (error) {
      logger.error('Training failed:', error);
      throw new InternalServerError(
        error instanceof Error ? error.message : 'Training failed'
      );
    }
  }

  /**
   * Evaluate model on test data - POST /ml/evaluate
   * Body: { model?: string, testData: [{ text: string, label: string }], metrics?: string[] }
   */
  @Post('/evaluate')
  async evaluate(
    @Body() body: { model?: string; testData?: Array<{ text: string; label: string }>; metrics?: string[] }
  ): Promise<{ evaluation: unknown }> {
    const testData = body?.testData ?? [];
    if (!Array.isArray(testData) || testData.length === 0) {
      throw new BadRequestError('testData array is required with at least one { text, label } sample');
    }

    const modelName = body?.model ?? DEFAULT_MODEL;
    if (!MODEL_CONFIG[modelName]) {
      throw new BadRequestError(`Unknown model: ${modelName}. Use: ${Object.keys(MODEL_CONFIG).join(', ')}`);
    }

    const config = MODEL_CONFIG[modelName];
    const validated = testData.filter(
      (s) =>
        typeof s?.text === 'string' &&
        typeof s?.label === 'string' &&
        (config.customLabels ? /^[a-z0-9_]+$/.test(s.label.toLowerCase().replace(/\s+/g, '_')) : config.validLabels.includes(s.label))
    );

    if (validated.length === 0) {
      throw new BadRequestError(
        `No valid test samples (need text and label: ${config.validLabels.join('|')})`
      );
    }

    try {
      const evaluation = await this.metricsService.evaluate(modelName, validated, {
        metrics: (body?.metrics as ('accuracy' | 'f1' | 'precision' | 'recall')[]) ?? [
          'accuracy',
          'f1',
          'precision',
          'recall',
        ],
        labelKey: 'label',
        predictionKey: config.predictionKey,
      });
      return { evaluation };
    } catch (error) {
      logger.error('Evaluation failed:', error);
      throw new InternalServerError(
        error instanceof Error ? error.message : 'Evaluation failed'
      );
    }
  }

  /**
   * Get model metrics - GET /ml/metrics?model=sentiment-classifier&version=1.0.0
   */
  @Get('/metrics')
  async getMetrics(
    @Query('model') modelName?: string,
    @Query('version') version?: string
  ) {
    const model = modelName ?? DEFAULT_MODEL;
    const metrics = this.metricsService.getMetrics(model, version);
    const history = this.metricsService.getHistory(model);

    return {
      model,
      current: metrics ?? null,
      history: history.map((h) => ({
        version: h.version,
        accuracy: h.metrics.accuracy,
        evaluatedAt: h.evaluatedAt,
      })),
    };
  }

  /**
   * List registered models - GET /ml/models
   */
  @Get('/models')
  async listModels() {
    const models = this.modelRegistry.list();
    return {
      models: models.map((m) => ({
        name: m.name,
        version: m.version,
        framework: m.framework,
        description: m.description,
      })),
    };
  }

  /**
   * Model versions - GET /ml/models/:name/versions
   */
  @Get('/models/:name/versions')
  async getVersions(@Param('name') name: string) {
    const versions = this.modelRegistry.getVersions(name);
    return { name, versions };
  }
}

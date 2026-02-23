/**
 * EmbeddingSentimentClassifier - Sentiment analysis using offline LLM embeddings
 *
 * Uses @huggingface/transformers with all-MiniLM-L6-v2 for 384-dim embeddings.
 * - Training: Embeds labeled samples, computes centroid per class (positive/negative/neutral)
 * - Prediction: Embeds input, nearest-centroid classification via cosine similarity
 *
 * Falls back to the sentiment-analysis pipeline when no training data is provided yet.
 */
import {
  Model,
  Train,
  Predict,
  Injectable,
  type TrainingData,
  type TrainingResult,
} from '@hazeljs/ml';

export type SentimentLabel = 'positive' | 'negative' | 'neutral';

export interface SentimentSample {
  text: string;
  label: SentimentLabel;
}

export interface SentimentPrediction {
  sentiment: SentimentLabel;
  confidence: number;
  scores: Record<SentimentLabel, number>;
}

export interface SentimentTrainingData extends TrainingData {
  samples: SentimentSample[];
}

type Pipeline = Awaited<ReturnType<typeof import('@huggingface/transformers').pipeline>>;

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum; // Already normalized -> dot product = cosine sim
}

function tensorToArray(tensor: unknown): number[] {
  if (Array.isArray(tensor)) return tensor.flat(Infinity) as number[];
  const t = tensor as {
    data?: Float32Array | number[];
    tolist?: () => unknown;
    size?: number[];
  };
  if (typeof t?.tolist === 'function') {
    const list = t.tolist();
    return (Array.isArray(list) ? list.flat(Infinity) : []) as number[];
  }
  if (t?.data) return Array.from(t.data);
  return [];
}

@Model({
  name: 'embedding-sentiment-classifier',
  version: '1.0.0',
  framework: 'onnx',
  description: 'Sentiment analysis using offline LLM embeddings (all-MiniLM-L6-v2)',
  tags: ['nlp', 'sentiment', 'embeddings', 'production'],
})
@Injectable()
export class EmbeddingSentimentClassifier {
  private extractor: Pipeline | null = null;
  private sentimentPipeline: Pipeline | null = null;
  private centroids: Map<SentimentLabel, number[]> = new Map();
  private isTrained = false;

  private async getExtractor(): Promise<Pipeline> {
    if (this.extractor) return this.extractor;
    const { pipeline } = await import('@huggingface/transformers');
    this.extractor = await pipeline(
      'feature-extraction',
      'Xenova/all-MiniLM-L6-v2'
    );
    return this.extractor;
  }

  private async getSentimentPipeline(): Promise<Pipeline> {
    if (this.sentimentPipeline) return this.sentimentPipeline;
    const { pipeline } = await import('@huggingface/transformers');
    this.sentimentPipeline = await pipeline(
      'sentiment-analysis',
      'Xenova/distilbert-base-uncased-finetuned-sst-2-english'
    );
    return this.sentimentPipeline;
  }

  private async embed(text: string): Promise<number[]> {
    const ext = await this.getExtractor();
    const out = await ext(text, { pooling: 'mean', normalize: true });
    return this.parseEmbeddingOutput(out, 1)[0] ?? [];
  }

  private async embedBatch(texts: string[]): Promise<number[][]> {
    const ext = await this.getExtractor();
    const out = await ext(texts, { pooling: 'mean', normalize: true });
    return this.parseEmbeddingOutput(out, texts.length);
  }

  private parseEmbeddingOutput(out: unknown, expectedCount: number): number[][] {
    const raw = tensorToArray(out);
    const dim = 384;
    if (raw.length === 0) return Array(expectedCount)
      .fill(null)
      .map(() => new Array(dim).fill(0));

    const result: number[][] = [];
    const totalFloats = raw.length;
    const floatsPerItem = totalFloats / expectedCount;
    for (let i = 0; i < expectedCount; i++) {
      const start = Math.floor(i * floatsPerItem);
      const end = Math.floor((i + 1) * floatsPerItem);
      result.push(raw.slice(start, end));
    }
    return result;
  }

  private meanVectors(vectors: number[][]): number[] {
    if (vectors.length === 0) return [];
    const dim = vectors[0].length;
    const mean = new Array(dim).fill(0);
    for (const v of vectors) {
      for (let i = 0; i < dim; i++) mean[i] += v[i];
    }
    const n = vectors.length;
    for (let i = 0; i < dim; i++) mean[i] /= n;
    // Normalize
    let mag = 0;
    for (let i = 0; i < dim; i++) mag += mean[i] * mean[i];
    mag = Math.sqrt(mag) || 1;
    for (let i = 0; i < dim; i++) mean[i] /= mag;
    return mean;
  }

  @Train({ pipeline: 'sentiment-preprocessing', epochs: 1, batchSize: 32 })
  async train(data: SentimentTrainingData): Promise<TrainingResult> {
    const { samples } = data;
    if (!samples?.length) {
      return { accuracy: 0, loss: 1, metrics: { error: 1 } };
    }

    const byLabel: Record<SentimentLabel, number[][]> = {
      positive: [],
      negative: [],
      neutral: [],
    };

    const texts = samples.map((s) => s.text);
    const embeddings = await this.embedBatch(texts);

    for (let i = 0; i < samples.length; i++) {
      const label = samples[i].label as SentimentLabel;
      if (byLabel[label]) byLabel[label].push(embeddings[i]);
    }

    this.centroids = new Map();
    for (const label of ['positive', 'negative', 'neutral'] as SentimentLabel[]) {
      const vecs = byLabel[label];
      if (vecs.length > 0) {
        this.centroids.set(label, this.meanVectors(vecs));
      } else {
        this.centroids.set(label, new Array(384).fill(0));
      }
    }

    this.isTrained = true;

    let correct = 0;
    for (let i = 0; i < samples.length; i++) {
      const pred = await this.predictInternal(samples[i].text);
      if (pred.sentiment === samples[i].label) correct++;
    }

    const accuracy = samples.length > 0 ? correct / samples.length : 0;
    return {
      accuracy,
      loss: 1 - accuracy,
      metrics: {
        samplesTrained: samples.length,
        vocabularySize: this.centroids.size * 384,
      },
    };
  }

  @Predict({ batch: true, endpoint: '/predict' })
  async predict(input: unknown): Promise<SentimentPrediction> {
    const text = typeof input === 'string' ? input : (input as { text?: string })?.text ?? '';
    return this.predictInternal(text);
  }

  private async predictInternal(text: string): Promise<SentimentPrediction> {
    if (this.isTrained && this.centroids.size > 0) {
      const emb = await this.embed(text);
      const scores: Record<SentimentLabel, number> = {
        positive: 0,
        negative: 0,
        neutral: 0,
      };

      for (const [label, centroid] of this.centroids) {
        scores[label] = Math.max(0, cosineSimilarity(emb, centroid));
      }

      const total = scores.positive + scores.negative + scores.neutral;
      const entries = (Object.entries(scores) as [SentimentLabel, number][]).sort(
        (a, b) => b[1] - a[1]
      );
      const [sentiment, score] = entries[0];
      const confidence = total > 0 ? score / total : 1 / 3;

      return {
        sentiment,
        confidence: Math.min(1, Math.max(0, confidence)),
        scores,
      };
    }

    return this.predictWithFallback(text);
  }

  private async predictWithFallback(text: string): Promise<SentimentPrediction> {
    const pipe = await this.getSentimentPipeline();
    const out = await pipe(text, { top_k: null });
    const results = Array.isArray(out) ? out : [out];
    const item = results[0] as { label: string; score: number } | undefined;

    // Binary model returns one label; derive the other as 1 - score
    let posScore: number;
    let negScore: number;
    if (item?.label === 'POSITIVE') {
      posScore = item.score;
      negScore = 1 - item.score;
    } else if (item?.label === 'NEGATIVE') {
      negScore = item.score;
      posScore = 1 - item.score;
    } else {
      posScore = 0.5;
      negScore = 0.5;
    }

    let sentiment: SentimentLabel;
    let confidence: number;
    if (Math.abs(posScore - negScore) < 0.25) {
      sentiment = 'neutral';
      confidence = 0.5;
    } else if (posScore > negScore) {
      sentiment = 'positive';
      confidence = posScore;
    } else {
      sentiment = 'negative';
      confidence = negScore;
    }

    return {
      sentiment,
      confidence,
      scores: {
        positive: posScore,
        negative: negScore,
        neutral: Math.max(0, 1 - posScore - negScore),
      },
    };
  }

  getStats(): { isTrained: boolean; hasCentroids: boolean } {
    return {
      isTrained: this.isTrained,
      hasCentroids: this.centroids.size > 0,
    };
  }
}

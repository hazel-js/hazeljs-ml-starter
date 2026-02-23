/**
 * SentimentClassifier - Real-world sentiment analysis model
 *
 * Uses a bag-of-words approach with trainable word weights:
 * - Training: Builds word frequency maps per sentiment from labeled data
 * - Prediction: Scores text against learned vocabularies, returns sentiment + confidence
 *
 * In production, you would integrate TensorFlow.js or ONNX Runtime for neural models.
 * This implementation demonstrates the @hazeljs/ml decorator pattern with a working algorithm.
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

// Default sentiment lexicons (can be overwritten by training)
const DEFAULT_POSITIVE = new Set([
  'good', 'great', 'love', 'excellent', 'amazing', 'wonderful', 'happy', 'best',
  'fantastic', 'perfect', 'awesome', 'brilliant', 'beautiful', 'nice',
]);

const DEFAULT_NEGATIVE = new Set([
  'bad', 'terrible', 'hate', 'awful', 'poor', 'worst', 'horrible', 'disappointing',
  'sad', 'angry', 'ugly', 'stupid', 'useless', 'failed',
  // Profanity and offensive language
  'fuck', 'damn', 'shit', 'crap', 'hell', 'ass', 'idiot', 'dumb', 'moron',
  'bastard', 'bullshit', 'suck', 'sucks', 'ridiculous', 'pathetic',
]);

const NEGATION_WORDS = new Set([
  'not', 'no', 'never', 'neither', 'nor', 'nothing', 'nobody', 'nowhere',
  'hardly', 'barely', 'scarcely', 'isnt', 'arent', 'wasnt', 'werent',
  'dont', 'doesnt', 'didnt', 'wont', 'cant', 'cannot', 'without',
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 1);
}

@Model({
  name: 'sentiment-classifier',
  version: '1.0.0',
  framework: 'custom',
  description: 'Text sentiment classification (positive/negative/neutral)',
  tags: ['nlp', 'sentiment', 'production'],
})
@Injectable()
export class SentimentClassifier {
  private positiveWords: Map<string, number> = new Map();
  private negativeWords: Map<string, number> = new Map();
  private isTrained = false;

  constructor() {
    this.initializeDefaults();
  }

  private initializeDefaults(): void {
    DEFAULT_POSITIVE.forEach((w) => this.positiveWords.set(w, 1));
    DEFAULT_NEGATIVE.forEach((w) => this.negativeWords.set(w, 1));
  }

  @Train({ pipeline: 'sentiment-preprocessing', epochs: 1, batchSize: 32 })
  async train(data: SentimentTrainingData): Promise<TrainingResult> {
    const { samples } = data;
    if (!samples?.length) {
      return { accuracy: 0, loss: 1, metrics: { error: 1 } };
    }

    const posMap = new Map<string, number>();
    const negMap = new Map<string, number>();
    const neuMap = new Map<string, number>();

    for (const { text, label } of samples) {
      const tokens = tokenize(text);
      for (const token of tokens) {
        if (label === 'positive') {
          posMap.set(token, (posMap.get(token) ?? 0) + 1);
        } else if (label === 'negative') {
          negMap.set(token, (negMap.get(token) ?? 0) + 1);
        } else {
          neuMap.set(token, (neuMap.get(token) ?? 0) + 1);
        }
      }
    }

    // Merge with defaults, add learned weights
    const merge = (
      base: Map<string, number>,
      learned: Map<string, number>,
      weight: number
    ): void => {
      learned.forEach((count, word) => {
        base.set(word, (base.get(word) ?? 0) + count * weight);
      });
    };

    merge(this.positiveWords, posMap, 2);
    merge(this.negativeWords, negMap, 2);
    // Neutral words dampen extremity - we use inverse of pos/neg
    neuMap.forEach((count, word) => {
      this.positiveWords.set(word, (this.positiveWords.get(word) ?? 0) - count * 0.5);
      this.negativeWords.set(word, (this.negativeWords.get(word) ?? 0) - count * 0.5);
    });

    this.isTrained = true;

    // Compute training accuracy on same data (demo)
    let correct = 0;
    for (const sample of samples) {
      const pred = this.predictInternal(sample.text);
      if (pred.sentiment === sample.label) correct++;
    }

    const accuracy = samples.length > 0 ? correct / samples.length : 0;
    return {
      accuracy,
      loss: 1 - accuracy,
      metrics: {
        samplesTrained: samples.length,
        vocabularySize: this.positiveWords.size + this.negativeWords.size,
      },
    };
  }

  @Predict({ batch: true, endpoint: '/predict' })
  async predict(input: unknown): Promise<SentimentPrediction> {
    const text = typeof input === 'string' ? input : (input as { text?: string })?.text ?? '';
    return this.predictInternal(text);
  }

  private predictInternal(text: string): SentimentPrediction {
    const tokens = tokenize(text);
    let posScore = 0;
    let negScore = 0;
    let negateNext = false;

    for (const token of tokens) {
      if (NEGATION_WORDS.has(token)) {
        negateNext = true;
        continue;
      }

      let p = this.positiveWords.get(token) ?? 0;
      let n = this.negativeWords.get(token) ?? 0;
      if (negateNext && (p > 0 || n > 0)) {
        [p, n] = [n, p]; // flip polarity: "not good" -> negative
        negateNext = false;
      }

      posScore += p;
      negScore += n;
    }

    const neutralScore = Math.max(0, 1 - Math.abs(posScore - negScore) / (tokens.length || 1));
    const scores: Record<SentimentLabel, number> = {
      positive: Math.max(0, posScore),
      negative: Math.max(0, negScore),
      neutral: neutralScore,
    };

    const total = scores.positive + scores.negative + scores.neutral;
    const maxLabel = (Object.entries(scores) as [SentimentLabel, number][]).reduce((a, b) =>
      a[1] >= b[1] ? a : b
    );
    const confidence = total > 0 ? maxLabel[1] / total : 1 / 3;

    return {
      sentiment: maxLabel[0],
      confidence: Math.min(1, Math.max(0, confidence)),
      scores,
    };
  }

  getStats(): { isTrained: boolean; vocabSize: number } {
    return {
      isTrained: this.isTrained,
      vocabSize: this.positiveWords.size + this.negativeWords.size,
    };
  }
}

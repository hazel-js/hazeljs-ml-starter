/**
 * SpamClassifier - Binary spam/ham classification
 *
 * Widely used for email, SMS, and content moderation.
 * Uses trainable bag-of-words with spam and ham vocabularies.
 */
import {
  Model,
  Train,
  Predict,
  Injectable,
  type TrainingData,
  type TrainingResult,
} from '@hazeljs/ml';

export type SpamLabel = 'spam' | 'ham';

export interface SpamSample {
  text: string;
  label: SpamLabel;
}

export interface SpamPrediction {
  label: SpamLabel;
  confidence: number;
  scores: Record<SpamLabel, number>;
  isSpam: boolean;
}

export interface SpamTrainingData extends TrainingData {
  samples: SpamSample[];
}

const DEFAULT_SPAM_INDICATORS = new Set([
  'free', 'win', 'prize', 'click', 'urgent', 'winner', 'congratulations',
  'limited', 'offer', 'act now', 'cash', 'money', 'guarantee', 'viagra',
  'lottery', 'inheritance', 'nigeria', 'unsubscribe', 'opt out',
]);

const DEFAULT_HAM_INDICATORS = new Set([
  'meeting', 'tomorrow', 'schedule', 'project', 'thanks', 'regards',
  'invoice', 'confirm', 'question', 'please', 'dear', 'hello',
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 1);
}

@Model({
  name: 'spam-classifier',
  version: '1.0.0',
  framework: 'custom',
  description: 'Binary spam/ham classification for emails, SMS, and content moderation',
  tags: ['nlp', 'spam', 'moderation', 'production'],
})
@Injectable()
export class SpamClassifier {
  private spamWords: Map<string, number> = new Map();
  private hamWords: Map<string, number> = new Map();
  private isTrained = false;

  constructor() {
    DEFAULT_SPAM_INDICATORS.forEach((w) => this.spamWords.set(w, 1));
    DEFAULT_HAM_INDICATORS.forEach((w) => this.hamWords.set(w, 1));
  }

  @Train({ pipeline: 'spam-preprocessing', epochs: 1, batchSize: 32 })
  async train(data: SpamTrainingData): Promise<TrainingResult> {
    const { samples } = data;
    if (!samples?.length) {
      return { accuracy: 0, loss: 1, metrics: { error: 1 } };
    }

    const spamMap = new Map<string, number>();
    const hamMap = new Map<string, number>();

    for (const { text, label } of samples) {
      const tokens = tokenize(text);
      for (const token of tokens) {
        if (label === 'spam') {
          spamMap.set(token, (spamMap.get(token) ?? 0) + 1);
        } else {
          hamMap.set(token, (hamMap.get(token) ?? 0) + 1);
        }
      }
    }

    const merge = (base: Map<string, number>, learned: Map<string, number>, weight: number): void => {
      learned.forEach((count, word) => {
        base.set(word, (base.get(word) ?? 0) + count * weight);
      });
    };

    merge(this.spamWords, spamMap, 2);
    merge(this.hamWords, hamMap, 2);

    this.isTrained = true;

    let correct = 0;
    for (const sample of samples) {
      const pred = this.predictInternal(sample.text);
      if (pred.label === sample.label) correct++;
    }

    const accuracy = samples.length > 0 ? correct / samples.length : 0;
    return {
      accuracy,
      loss: 1 - accuracy,
      metrics: {
        samplesTrained: samples.length,
        vocabularySize: this.spamWords.size + this.hamWords.size,
      },
    };
  }

  @Predict({ batch: true, endpoint: '/predict' })
  async predict(input: unknown): Promise<SpamPrediction> {
    const text = typeof input === 'string' ? input : (input as { text?: string })?.text ?? '';
    return this.predictInternal(text);
  }

  private predictInternal(text: string): SpamPrediction {
    const tokens = tokenize(text);
    let spamScore = 0;
    let hamScore = 0;

    for (const token of tokens) {
      spamScore += this.spamWords.get(token) ?? 0;
      hamScore += this.hamWords.get(token) ?? 0;
    }

    const scores: Record<SpamLabel, number> = {
      spam: Math.max(0, spamScore),
      ham: Math.max(0, hamScore),
    };

    const total = scores.spam + scores.ham;
    const isSpam = scores.spam >= scores.ham;
    const label: SpamLabel = isSpam ? 'spam' : 'ham';
    const confidence = total > 0 ? (isSpam ? scores.spam : scores.ham) / total : 0.5;

    return {
      label,
      confidence: Math.min(1, Math.max(0, confidence)),
      scores,
      isSpam,
    };
  }

  getStats(): { isTrained: boolean; vocabSize: number } {
    return {
      isTrained: this.isTrained,
      vocabSize: this.spamWords.size + this.hamWords.size,
    };
  }
}

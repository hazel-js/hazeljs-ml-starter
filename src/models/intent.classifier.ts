/**
 * IntentClassifier - Multi-class intent classification
 *
 * Widely used for chatbots, virtual assistants, and support ticket routing.
 * Maps user input to predefined intents (e.g. refund, bug_report, feature_request).
 */
import {
  Model,
  Train,
  Predict,
  Injectable,
  type TrainingData,
  type TrainingResult,
} from '@hazeljs/ml';

export type IntentLabel =
  | 'refund'
  | 'bug_report'
  | 'feature_request'
  | 'general_inquiry'
  | 'complaint'
  | 'greeting'
  | 'farewell'
  | 'other';

export interface IntentSample {
  text: string;
  label: IntentLabel;
}

export interface IntentPrediction {
  intent: IntentLabel;
  confidence: number;
  scores: Record<IntentLabel, number>;
  topIntents: Array<{ intent: IntentLabel; score: number }>;
}

export interface IntentTrainingData extends TrainingData {
  samples: IntentSample[];
}

const DEFAULT_INTENTS: IntentLabel[] = [
  'refund',
  'bug_report',
  'feature_request',
  'general_inquiry',
  'complaint',
  'greeting',
  'farewell',
  'other',
];

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 1);
}

@Model({
  name: 'intent-classifier',
  version: '1.0.0',
  framework: 'custom',
  description: 'Multi-class intent classification for chatbots and support routing',
  tags: ['nlp', 'intent', 'chatbot', 'support', 'production'],
})
@Injectable()
export class IntentClassifier {
  private intentVocab: Map<IntentLabel, Map<string, number>> = new Map();
  private knownIntents: Set<IntentLabel> = new Set(DEFAULT_INTENTS);
  private isTrained = false;

  constructor() {
    DEFAULT_INTENTS.forEach((intent) => this.intentVocab.set(intent, new Map()));
  }

  @Train({ pipeline: 'intent-preprocessing', epochs: 1, batchSize: 32 })
  async train(data: IntentTrainingData): Promise<TrainingResult> {
    const { samples } = data;
    if (!samples?.length) {
      return { accuracy: 0, loss: 1, metrics: { error: 1 } };
    }

    // Reinitialize vocab maps
    this.knownIntents = new Set();
    this.intentVocab = new Map();

    for (const { label } of samples) {
      if (!this.knownIntents.has(label as IntentLabel)) {
        this.knownIntents.add(label as IntentLabel);
        this.intentVocab.set(label as IntentLabel, new Map());
      }
    }

    for (const { text, label } of samples) {
      const vocab = this.intentVocab.get(label as IntentLabel);
      if (!vocab) continue;

      const tokens = tokenize(text);
      for (const token of tokens) {
        vocab.set(token, (vocab.get(token) ?? 0) + 1);
      }
    }

    // Merge with weight for learned terms
    const intents = Array.from(this.intentVocab.keys());
    for (const intent of intents) {
      const vocab = this.intentVocab.get(intent)!;
      vocab.forEach((count, word) => vocab.set(word, 1 + count * 1.5));
    }

    this.isTrained = true;

    let correct = 0;
    for (const sample of samples) {
      const pred = this.predictInternal(sample.text);
      if (pred.intent === sample.label) correct++;
    }

    const accuracy = samples.length > 0 ? correct / samples.length : 0;
    return {
      accuracy,
      loss: 1 - accuracy,
      metrics: {
        samplesTrained: samples.length,
        intentCount: this.knownIntents.size,
        vocabularySize: Array.from(this.intentVocab.values()).reduce(
          (sum, m) => sum + m.size,
          0
        ),
      },
    };
  }

  @Predict({ batch: true, endpoint: '/predict' })
  async predict(input: unknown): Promise<IntentPrediction> {
    const text = typeof input === 'string' ? input : (input as { text?: string })?.text ?? '';
    return this.predictInternal(text);
  }

  private predictInternal(text: string): IntentPrediction {
    const tokens = tokenize(text);
    const scores: Partial<Record<IntentLabel, number>> = {};

    for (const intent of this.knownIntents) {
      const vocab = this.intentVocab.get(intent)!;
      let score = 0;
      for (const token of tokens) {
        score += vocab.get(token) ?? 0;
      }
      scores[intent] = Math.max(0, score);
    }

    const total = Object.values(scores).reduce((a, b) => a + b, 0);
    const entries = (Object.entries(scores) as [IntentLabel, number][]).sort((a, b) => b[1] - a[1]);
    const topIntent = entries[0];
    const intent: IntentLabel = topIntent?.[0] ?? 'other';
    const confidence = total > 0 ? (topIntent?.[1] ?? 0) / total : 1 / this.knownIntents.size;

    return {
      intent,
      confidence: Math.min(1, Math.max(0, confidence)),
      scores: scores as Record<IntentLabel, number>,
      topIntents: entries.slice(0, 3).map(([i, s]) => ({ intent: i, score: s })),
    };
  }

  getStats(): { isTrained: boolean; intentCount: number; vocabSize: number } {
    return {
      isTrained: this.isTrained,
      intentCount: this.knownIntents.size,
      vocabSize: Array.from(this.intentVocab.values()).reduce((sum, m) => sum + m.size, 0),
    };
  }
}

/**
 * App Module - ML Starter Application
 *
 * Configures MLModule with SentimentClassifier and ML controller.
 * The MLBootstrap registers training pipelines and runs on init.
 */
import { HazelModule } from '@hazeljs/core';
import { MLModule } from '@hazeljs/ml';
import { SentimentClassifier, EmbeddingSentimentClassifier, SpamClassifier, IntentClassifier } from './models';
import { MLController } from './controllers/ml.controller';
import { MLBootstrap } from './ml/ml.bootstrap';

@HazelModule({
  imports: [
    MLModule.forRoot({
      models: [SentimentClassifier, EmbeddingSentimentClassifier, SpamClassifier, IntentClassifier],
    }),
  ],
  controllers: [MLController],
  providers: [MLBootstrap],
})
export class AppModule {}

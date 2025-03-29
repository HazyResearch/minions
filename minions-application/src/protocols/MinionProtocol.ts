import { Config } from '../components/Sidebar';
import { BaseProtocol, ProtocolCallback, ProtocolResult, ProtocolOptions, Message, Usage } from '../types/protocol';
import { ClientFactory } from '../clients/ClientFactory';

const SUPERVISOR_PROMPT = `You are a supervisor AI that helps answer questions.
Your job is to:
1. Understand the question and context
2. Provide clear and accurate answers
3. Include relevant details and explanations

You should:
- Be clear and concise
- Focus on the most important aspects
- Provide context when needed
- Ensure your answer is complete`;

const WORKER_PROMPT_TEMPLATE = `You are a worker AI that helps answer specific questions.
Your job is to:
1. Understand the question and its context
2. Provide a clear and focused answer
3. Include relevant details and explanations

Context: {context}
Context Description: {contextDescription}

Question: {question}

Please provide your answer:`;

export class MinionProtocol implements BaseProtocol {
  private localClient: any;
  private remoteClient: any;
  private callback?: ProtocolCallback;

  constructor(config: Config, callback?: ProtocolCallback) {
    console.log('Initializing MinionProtocol with config:', config);
    this.localClient = ClientFactory.createClient(config, true);
    this.remoteClient = ClientFactory.createClient(config, false);
    this.callback = callback;
  }

  async run(query: string, context: string[], options: ProtocolOptions): Promise<ProtocolResult> {
    console.log('MinionProtocol.run called with:', { query, context, options });
    const { contextDescription, fileMetadata } = options;
    let totalUsage: Usage = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };

    // Get supervisor's advice
    console.log('Getting supervisor advice...');
    const supervisorPrompt = this.createSupervisorPrompt(query, context, contextDescription);
    console.log('Supervisor prompt:', supervisorPrompt);
    
    if (this.callback) {
      this.callback('supervisor', 'Working...', false);
    }

    const [advice, adviceUsage] = await this.remoteClient.chat([
      { role: 'system', content: SUPERVISOR_PROMPT },
      { role: 'user', content: supervisorPrompt }
    ]);
    console.log('Supervisor advice:', advice);

    if (this.callback) {
      this.callback('supervisor', advice[0], true);
    }

    totalUsage = this.mergeUsage(totalUsage, adviceUsage);

    // Get worker's answer
    console.log('Getting worker answer...');
    const workerPrompt = this.createWorkerPrompt(query, context, contextDescription);
    console.log('Worker prompt:', workerPrompt);
    
    if (this.callback) {
      this.callback('worker', 'Working...', false);
    }

    const [answer, answerUsage] = await this.localClient.chat([
      { role: 'system', content: WORKER_PROMPT_TEMPLATE },
      { role: 'user', content: workerPrompt }
    ]);
    console.log('Worker answer:', answer);

    if (this.callback) {
      this.callback('worker', answer[0], true);
    }

    totalUsage = this.mergeUsage(totalUsage, answerUsage);

    // Get supervisor's final answer
    console.log('Getting supervisor final answer...');
    const finalPrompt = this.createFinalPrompt(query, context, answer);
    console.log('Final prompt:', finalPrompt);
    
    if (this.callback) {
      this.callback('supervisor', 'Working on final answer...', false);
    }

    const [finalAnswer, finalUsage] = await this.remoteClient.chat([
      { role: 'system', content: SUPERVISOR_PROMPT },
      { role: 'user', content: finalPrompt }
    ]);
    console.log('Final answer:', finalAnswer);

    if (this.callback) {
      this.callback('supervisor', finalAnswer[0], true);
    }

    totalUsage = this.mergeUsage(totalUsage, finalUsage);

    return {
      finalAnswer: finalAnswer[0],
      usage: totalUsage
    };
  }

  private createSupervisorPrompt(query: string, context: string[], contextDescription: string): string {
    return `Question: ${query}

Context:
${context.join('\n')}

Context Description: ${contextDescription}

Please provide advice on how to answer this question.`;
  }

  private createWorkerPrompt(query: string, context: string[], contextDescription: string): string {
    return WORKER_PROMPT_TEMPLATE
      .replace('{context}', context.join('\n'))
      .replace('{contextDescription}', contextDescription)
      .replace('{question}', query);
  }

  private createFinalPrompt(query: string, context: string[], answer: string[]): string {
    return `Question: ${query}

Context:
${context.join('\n')}

Worker's Answer:
${answer.join('\n')}

Please provide a final answer based on the worker's response.`;
  }

  private mergeUsage(a: Usage, b: Usage): Usage {
    return {
      prompt_tokens: a.prompt_tokens + b.prompt_tokens,
      completion_tokens: a.completion_tokens + b.completion_tokens,
      total_tokens: a.total_tokens + b.total_tokens
    };
  }
} 
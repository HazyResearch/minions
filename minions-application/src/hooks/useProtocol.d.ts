import { Config } from '../components/Sidebar';

interface ProtocolInput {
  query: string;
  context: string[];
  config: Config;
  contextDescription: string;
  fileMetadata: string;
}

interface ProtocolOutput {
  output: string;
  loading: boolean;
  runProtocol: (input: ProtocolInput) => Promise<void>;
}

declare const useProtocol: () => ProtocolOutput;

export default useProtocol; 
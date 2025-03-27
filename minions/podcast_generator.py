from csm_mlx import CSM, csm_1b, generate, Segment
import mlx.core as mx
import audiofile
import numpy as np
import torch
import torchaudio
import re
import os


# Function to load and preprocess audio (as before)
def read_audio(audio_path, sampling_rate=24000) -> mx.array:
    import audresample
    signal, original_sampling_rate = audiofile.read(audio_path, always_2d=True)
    signal = audresample.resample(signal, original_sampling_rate, sampling_rate)
    signal = mx.array(signal)
    if signal.shape[0] > 1:
        signal = signal.mean(axis=0)
    else:
        signal = signal.squeeze(0)
    return signal

class PodcastGenerator:
    def __init__(self, vector_speaker_id=3, gru_speaker_id=1):
        """Initialize the PodcastGenerator with the CSM model and speaker IDs."""
        # Initialize the CSM model
        self.csm = CSM(csm_1b())
        from huggingface_hub import hf_hub_download
        weight = hf_hub_download(repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors")
        self.csm.load_weights(weight)
        
        # Define speaker IDs
        self.VECTOR_SPEAKER_ID = vector_speaker_id  # Host 1 (using 3 instead of 0)
        self.GRU_SPEAKER_ID = gru_speaker_id        # Host 2
    
    def parse_transcript(self, transcript):
        """Parse a podcast transcript into speaker/text tuples using regex."""
        # Remove any tags or unwanted prefix (like <PODCAST TRANSCRIPT>)
        transcript = transcript.replace("<PODCAST TRANSCRIPT>", "").strip()

        # Regex pattern to capture each utterance
        pattern = re.compile(r'(HOST\s+(\d+):)\s*(.*?)(?=HOST\s+\d+:|$)', re.DOTALL)

        parsed = []
        for match in pattern.finditer(transcript):
            host_number = match.group(2)  # "1" or "2"
            utterance = match.group(3).strip().replace("\n", " ")
            
            # Map host numbers to speaker IDs
            if host_number == "1":
                speaker_id = self.VECTOR_SPEAKER_ID
            elif host_number == "2":
                speaker_id = self.GRU_SPEAKER_ID
            else:
                speaker_id = 0  # Default speaker if unknown
            
            parsed.append((speaker_id, utterance))
        
        return parsed
    
    def create_conversation_audio(self, conversation_plan, output_filename="full_conversation.wav", sampling_rate=24000):
        """Generate audio for a conversation plan with micro-chunking for maximum stability."""
        # Ensure output directory exists
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        all_audio_segments = []
        vector_segments = []
        gru_segments = []
        
        # Import the proper sampler
        try:
            from mlx_lm.sample_utils import make_sampler
        except ImportError:
            from csm_mlx.sampling import make_sampler
        
        # Create stable sampler configuration
        stable_sampler = make_sampler(temp=0.7, min_p=0.05)
        
        # Process each segment
        for i, (speaker_id, text) in enumerate(conversation_plan):
            if not text or text.strip() == "":
                print(f"Skipping empty text segment at position {i}")
                continue
            
            # Use micro-chunking (5 words max)
            text_chunks = self.chunk_text(text, max_words=5)
            print(f"Split into {len(text_chunks)} micro-chunks")
            
            for chunk in text_chunks:
                print(f"Generating audio for: {chunk}")
                
                # Generate audio with empty context for stability
                response_audio = generate(
                    self.csm,
                    text=chunk,
                    speaker=speaker_id,
                    context=[],  # Empty context like voice_generator
                    max_audio_length_ms=3000,  # Shorter segments for very small chunks
                    sampler=stable_sampler
                )
                
                # Maintain context (though not using it)
                new_segment = Segment(speaker=speaker_id, text=chunk, audio=mx.array(response_audio))
                
                if speaker_id == self.VECTOR_SPEAKER_ID:
                    vector_segments.append(new_segment)
                    if len(vector_segments) > 2:
                        vector_segments = vector_segments[-2:]
                else:
                    gru_segments.append(new_segment)
                    if len(gru_segments) > 2:
                        gru_segments = gru_segments[-2:]
                
                # Add to audio collection
                audio_tensor = torch.tensor(np.asarray(response_audio)).unsqueeze(0)
                all_audio_segments.append(audio_tensor)
                
                # Very short pause between micro-chunks (0.03s)
                silence_duration = int(0.03 * sampling_rate)
                silence = torch.zeros(1, silence_duration)
                all_audio_segments.append(silence)
            
            # Add slightly longer pause between different speakers
            silence_duration = int(0.5 * sampling_rate)
            silence = torch.zeros(1, silence_duration)
            all_audio_segments.append(silence)
        
        # Combine all segments
        if not all_audio_segments:
            print("No audio was generated!")
            return False
        
        full_conversation = torch.cat(all_audio_segments, dim=1)
        
        try:
            torchaudio.save(output_filename, full_conversation, sampling_rate)
            print(f"Saved full conversation to {output_filename}")
            return True
        except Exception as e:
            print(f"Error saving combined audio: {e}")
            return False
    
    def generate_from_transcript(self, transcript, output_filename="podcast_audio.wav"):
        """Parse a transcript and generate audio from it."""
        conversation_plan = self.parse_transcript(transcript)
        return self.create_conversation_audio(conversation_plan, output_filename)

    def chunk_text(self, text, max_words=5):
        """Split text into very small chunks of approximately 5 words each."""
        if not text:
            return []
        
        # First split by sentence boundaries
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        
        for sentence in sentences:
            # Split sentences into words
            words = sentence.split()
            
            # Group words into micro-chunks of max_words
            for i in range(0, len(words), max_words):
                chunk = ' '.join(words[i:i+max_words])
                if chunk:  # Skip empty chunks
                    chunks.append(chunk)
        
        # Ensure we have at least one chunk
        if not chunks and text.strip():
            chunks = [text.strip()]
        
        return chunks


# Example usage:
if __name__ == "__main__":
    # Example podcast transcript
    transcript = """
   <PODCAST TRANSCRIPT>\n\nHOST 1 (Alex): Welcome to Research Roundup! I'm Alex.\n\nHOST 2 (Jamie): And I'm Jamie. Today we're diving into a fascinating topic in the world of machine learning—the Transformer model. It's been a game-changer in how we process sequential data.\n\nHOST 1: Let's start with the basics. The Transformer model was introduced as a new sequence transduction model that replaces traditional recurrent neural networks, or RNNs, with self-attention mechanisms. Jamie, why is this shift significant?\n\nHOST 2: Well, Alex, RNNs have been the standard for sequence tasks, but they have limitations, especially in parallelizing computation. The Transformer model's use of self-attention allows for more efficient processing, which is crucial for tasks like language translation.\n\nHOST 1: Exactly. The paper's introduction highlights these advantages. Moving on to the background, it discusses sequence transduction models, including RNNs, CNNs, and attention mechanisms. What stands out to you here?\n\nHOST 2: The key takeaway is the limitations of RNNs and the benefits of self-attention. Self-attention mechanisms allow for more efficient processing by focusing on different parts of the input data simultaneously.\n\nHOST 1: That brings us to the model architecture. The Transformer model consists of encoder and decoder stacks, using multi-head attention and position-wise feed-forward networks. Can you break that down for us?\n\nHOST 2: Sure! Multi-head attention allows the model to attend to information from different representation subspaces at different positions. Position-wise feed-forward networks transform the output of each position separately, enhancing the model's ability to process data.\n\nHOST 1: And the attention mechanisms are crucial here. The scaled dot-product attention and multi-head attention work together to efficiently compute attention weights and process data in parallel.\n\nHOST 2: Right. The training section outlines the use of batched data, the Adam optimizer, and regularization techniques like dropout and label smoothing. These methods optimize the model's performance.\n\nHOST 1: The results speak for themselves. The Transformer model achieved state-of-the-art performance on several machine translation tasks, including English-to-German and English-to-French translations.\n\nHOST 2: It's impressive how the model handles long-range dependencies and generalizes to other tasks. The conclusion suggests future research directions, like exploring more attention-based models and refining the architecture.\n\nHOST 1: Exciting times ahead for machine learning! That's all for today's episode.\n\nHOST 2: Thanks for joining us on Research Roundup. Until next time, stay curious and keep exploring the world of tech!"""
    
    # Create generator and produce audio
    generator = PodcastGenerator()
    generator.generate_from_transcript(transcript, "podcast_example.wav")





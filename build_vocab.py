# Build vocabulary from training data - FIXED VERSION
# This version directly accesses the dataset samples to avoid negative stride errors

from data_loader import CuriosityDataset, get_primary_question

# Build vocabulary directly from dataset samples (no image loading needed)
vocab = Vocabulary()

print("Building vocabulary from training data...")

# Access the underlying dataset samples directly
train_dataset = train_loader.dataset

# Get questions directly from samples without loading images
for sample in train_dataset.samples:
    # Get primary question using the same logic as the dataset
    questions = sample.get('questions', [])
    question_types = sample.get('question_types', [])
    
    # Use the same logic as get_primary_question
    if not questions:
        question = "what is this?"
    else:
        # Return first non-empty question
        question = "what is this?"
        for q in questions:
            if q and q.strip():
                question = q.strip()
                break
    
    vocab.add_sentence(question)

print(f"Vocabulary size: {len(vocab)}")
print(f"Sample words: {list(vocab.word2idx.keys())[:20]}")

# Test tokenization
test_question = "why is the man sitting like that?"
tokens = vocab.sentence_to_indices(test_question)
print(f"\nTest question: '{test_question}'")
print(f"Tokens: {tokens}")
print(f"Reconstructed: '{vocab.indices_to_sentence(tokens)}'")


reference_text = "Once upon a time, there was a little girl who lived in a village near the forest."
generated_text = "Once upon a time, the world was a place of great beauty and great danger. The world of the gods was the place where the great gods were born, and where they were to live."

# Initialize BLEU and ROUGE scorers
bleu = BLEUScore()
rouge = ROUGEScore()

# Calculate the BLEU and ROUGE scores
bleu_score = bleu([generated_text], [[reference_text]])
rouge_score = rouge([generated_text], [[reference_text]])

# Print the BLEU and ROUGE scores
print("BLEU Score:", bleu_score.item())
print("ROUGE Score:", rouge_score)

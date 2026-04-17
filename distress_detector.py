from transformers import pipeline

# Load the AI model (downloads once, then cached)
print("Loading AI model...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
print("AI model ready!")

# Labels we want to detect
LABELS = ["distress", "emergency", "safe", "normal"]

def analyze_text(text):
    """
    Takes any text and returns:
    - Whether it's a distress signal
    - Confidence score
    - All label probabilities
    """
    result = classifier(text, LABELS)
    
    # Get the top label and its score
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    
    # Build a clean result
    scores = dict(zip(result["labels"], result["scores"]))
    
    is_distress = (
        top_label in ["distress", "emergency"] 
        and top_score > 0.5
    )
    
    return {
        "text": text,
        "is_distress": is_distress,
        "top_label": top_label,
        "confidence": round(top_score * 100, 1),
        "scores": scores
    }

# ─────────────────────────────────────────
# TEST IT — run this file directly to test
# ─────────────────────────────────────────
if __name__ == "__main__":
    test_messages = [
        "Help! Earthquake destroyed my house, people are trapped!",
        "The weather is nice today, going for a walk",
        "SOS! We are stuck under rubble, need rescue immediately!",
        "Just had lunch, everything is fine here",
        "Flooding in our area, roads are blocked, we need help!",
        "Beautiful sunset today, feeling peaceful"
    ]
    
    print("\n" + "="*60)
    print("DISTRESS DETECTION TEST")
    print("="*60)
    
    for msg in test_messages:
        result = analyze_text(msg)
        status = "🚨 DISTRESS" if result["is_distress"] else "✅ SAFE"
        print(f"\n{status}")
        print(f"Text: {msg[:50]}...")
        print(f"Label: {result['top_label']} ({result['confidence']}% confidence)")
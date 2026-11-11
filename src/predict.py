from transformers import pipeline
import sys

RESET = "\033[0m"

def predict_sentiment(text):
    # Carrega o modelo direto do Hugging Face
    classifier = pipeline("sentiment-analysis", model="aluc0de/bertimbau-olist-sentiment")
    
    result = classifier(text)[0]
    
    # Mapeia as labels numéricas para texto
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    sentiment = label_map.get(result['label'], result['label'])
    
    return sentiment, result['score']

def show_batcave():
    yellow = "\033[1;33m"
    white = "\033[1;37m"
    grey = "\033[90m"
    print(f"\n{grey}==============================================={RESET}")
    print(f"{yellow}        I AM VENGEANCE. I AM THE NIGHT.        {RESET}")
    print(f"{white}         I AM ALUC0DE THE DEVELOPER.          {RESET}")
    print(f"{grey}==============================================={RESET}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "batcave":
            show_batcave()
        else:
            print(f"Unknown command: '{command}'. Try running without arguments.")
    
    else:
        example_text = "O produto chegou antes do prazo e é excelente!"
        
        sentiment, confidence = predict_sentiment(example_text)
        
        print(f"Text: {example_text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2%}")
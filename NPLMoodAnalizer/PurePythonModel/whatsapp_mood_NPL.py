import re
import matplotlib.pyplot as plt
import csv
from deep_translator import GoogleTranslator  # Importa Deep Translate per la traduzione
from textblob import TextBlob  # Per l'analisi del sentimento
import emoji

def extract_messages(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    messages = []
    participants = {}
    pattern = re.compile(r'\[\d{2}/\d{2}/\d{2}, \d{2}:\d{2}:\d{2}\] (.*?): (.*)')
    
    for line in lines:
        match = pattern.match(line)
        if match:
            participant = match.group(1)
            msg = match.group(2)
            msg = emoji.demojize(msg)  # Converte le emoji in testo
            if participant not in participants:
                participants[participant] = []
            participants[participant].append(msg)
    
    return participants

def translate_and_analyze(text):
    try:
        # Usa Deep Translate per tradurre il testo
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        return TextBlob(str(translated_text)).sentiment.polarity  # Usa TextBlob per analizzare il sentimento
    except Exception as e:
        print(f"Errore nella traduzione: {e}")  # Debug per eventuali errori
        return 0  # Se la traduzione fallisce, considera il messaggio neutro

def analyze_sentiment(participants, output_csv):
    mood_trends = {participant: [] for participant in participants}
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Participant", "Message", "Sentiment", "Mood Score"])
        
        for participant, messages in participants.items():
            mood_score = 0
            for msg in messages:
                if msg.strip():  # Evita messaggi vuoti
                    sentiment = translate_and_analyze(msg)  # Traduci e analizza
                    mood_score += sentiment  # Aggiungi il sentiment al punteggio accumulato
                    mood_trends[participant].append(mood_score)  # Salva il punteggio accumulato
                    writer.writerow([participant, msg, sentiment, mood_score])
    
    return mood_trends

def plot_mood(mood_trends):
    plt.figure(figsize=(10, 5))
    
    # Per ogni interlocutore, traccia il grafico
    for participant, mood_trend in mood_trends.items():
        plt.plot(mood_trend, marker='o', linestyle='-', label=participant)
    
    plt.xlabel('Numero di messaggi')
    plt.ylabel('Mood Score')
    plt.title('Andamento del Mood nella chat per ogni interlocutore')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    file_path = "/Users/emmanuelepiola/documents/_chat.txt"  # Cambia con il percorso corretto del file WhatsApp
    output_csv = "/Users/emmanuelepiola/documents/mood_scores.csv"  # File CSV per salvare i risultati
    participants = extract_messages(file_path)  # Estrai i messaggi dal file
    mood_trends = analyze_sentiment(participants, output_csv)  # Analizza i sentimenti e salva nel CSV
    plot_mood(mood_trends)  # Visualizza il grafico dell'andamento dell'umore

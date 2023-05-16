import numpy as np
from datasets import load_dataset
from spacy import tokenizer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, GPTJForCausalLM, AutoModel
import sys
import utils
import rg
import openai
import logging

#Replace with real key
API_KEY = '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'
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

openai.api_key = API_KEY


sum_prompt = """create a short summary of the following email body:
"Vi ønsker å gratulere deg med å bli akseptert til stillingen i vårt firma. Vi tror at din erfaring og kompetanse vil være en flott tilføyelse til teamet vårt. Vennligst se vedlagte dokument for detaljer om stillingen og forberedelse til din første arbeidsdag. Vi ser frem til å jobbe sammen med deg!" in from the perspective of the sender in norwegian"""


sum_prompt2 = """create a short summary of the following email body:
"Kjære ansatte, vi har oppdatert våre opplæringsprogrammer. Vennligst se gjennom programoversikten og meld deg på de kursene som passer for deg på HR-portalen. Gi beskjed til HR-lederen hvis du har spørsmål eller bekymringer. Takk skal du ha!" in from the perspective of the sender in norwegian"""


sum_1_shot = '''Hei Team, Bare en rask påminnelse om at vi har et møte planlagt for i morgen kl. 10 i konferanserommet. Vennligst vær forberedt med eventuelle oppdateringer eller spørsmål du måtte ha. Ser frem til å se dere alle der!->->Påminnelse om møte i konferanserommet kl. 10 imorgen.###
'''

sum_3_shot = '''Vi ønsker å informere deg om et fremtidig prosjekt i Tokyo som vil fokusere på utvikling av nye teknologier og tjenester innen vår bransje. Ta kontakt med oss hvis du er interessert i å diskutere mulighetene nærmere. QUALCOMM innovasjon.->->Informasjon om fremtidig prosjekt i Tokyo.###
Hei alle sammen, vi ønsker å informere dere om en endring i arbeidstiden fra mandag denne uken. Vi vil nå begynne en time tidligere om morgenen, kl. 8.00. Dette vil gi oss mer tid til å fullføre prosjektene våre innen fristen. Takk for forståelsen!->->Endring i arbeidstiden fra mandag denne uken til å begynne kl. 8.00.###
Hei Team, Bare en rask påminnelse om at vi har et møte planlagt for i morgen kl. 10 i konferanserommet. Vennligst vær forberedt med eventuelle oppdateringer eller spørsmål du måtte ha. Ser frem til å se dere alle der!->->Påminnelse om møte i konferanserommet kl. 10 imorgen.###
'''


sum_5_shot = '''Vi ønsker å informere deg om et fremtidig prosjekt i Tokyo som vil fokusere på utvikling av nye teknologier og tjenester innen vår bransje. Ta kontakt med oss hvis du er interessert i å diskutere mulighetene nærmere. QUALCOMM innovasjon.->->Informasjon om fremtidig prosjekt i Tokyo.###
Hei alle sammen, vi ønsker å informere dere om en endring i arbeidstiden fra mandag denne uken. Vi vil nå begynne en time tidligere om morgenen, kl. 8.00. Dette vil gi oss mer tid til å fullføre prosjektene våre innen fristen. Takk for forståelsen!->->Endring i arbeidstiden fra mandag denne uken til å begynne kl. 8.00.###
Hei Team, Bare en rask påminnelse om at vi har et møte planlagt for i morgen kl. 10 i konferanserommet. Vennligst vær forberedt med eventuelle oppdateringer eller spørsmål du måtte ha. Ser frem til å se dere alle der!->->Påminnelse om møte i konferanserommet kl. 10 imorgen.###
Hei alle unge skrivetalenter, vi ønsker å invitere dere til en skrivegruppe som vil finne sted hver torsdag kl. 02:15 på natten på Bryggen 12, Bergen. Dette er en mulighet for deg som liker å skrive, enten det er poesi, skjønnlitteratur eller annet, til å dele dine tekster med andre ungdommer og få feedback. Det er ingen påmelding eller krav til ferdighetsnivå, alle er velkomne. Vennlig hilsen, Anne Andersen->->Invitasjon til skrivegruppe som vil finne sted hver torsdag kl. 02:15 på Bryggen 12, Bergen. Det er ingen påmelding.###
Hei alle sammen, Jeg vil informere deg om noen av aktivitetene som vil være tilgjengelige på bedriftsfesten på 26. mai. Aktivitetene inkluderer minigolf, bordtennis og biljard. Dette vil være en flott mulighet til å slappe av og ha det gøy med kollegene dine. Vennligst la meg få vite hvis du har noen spørsmål eller bekymringer angående festen eller aktivitetene. Takk på forhånd. Vennlig hilsen, Anders.->->Informasjon om aktivitetene som vil være tilgjengelige på bedriftsfesten 26. mai.###
'''
FINE_TUNED = 1
TEXT_COMPLETION = 2
CHAT_COMPLETION = 3


ZERO_SHOT = 1
ONE_SHOT = 2
THREE_SHOT = 3



#fine-tuned davinci
#mode = FINE_TUNED
#context = ZERO_SHOT
#include_subjects = False
#model_name = "davinci:ft-edialog24-as-2023-05-07-19-34-37"

#Three shot gpt 3.5 turbo
#mode = CHAT_COMPLETION
#context = THREE_SHOT
#include_subjects = False
#model_name = "gpt-3.5-turbo"


#Fine tuned babbage with subjects
mode = FINE_TUNED
context = THREE_SHOT
include_subjects = True
model_name = "babbage:ft-edialog24-as-2023-05-07-20-12-57"

no_summary_dataset = load_dataset("BiasedLiar/nor_email_sum")

gold_labels = []
labels = []
for row in no_summary_dataset["test"]:
    gold_labels.append(row["goldlabel"])
    labels.append(row["label"])


start = 0
print("API loaded...")
# Create summaries of dataset with model
texts = [row["text"] for row in no_summary_dataset["test"]]
subjects = [row["subject"] for row in no_summary_dataset["test"]]
pred = []
i = 0
failures = 0
max_failures = 10
while i < len(texts) and failures < max_failures:
    
    logging.info(str(i) + " of " + str(len(texts)))
    try:
        if mode == CHAT_COMPLETION:
            if context == ZERO_SHOT:
                message_list = [{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "create a short summary of the following email body:\n\"" + texts[i] + "\" from the perspective of the sender in norwegian.  Only include the important information"}]
            if context == ONE_SHOT:
                message_list = [{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "create a short summary of the following email body:\n\"Hei Team, Bare en rask påminnelse om at vi har et møte planlagt for i morgen kl. 10 i konferanserommet. Vennligst vær forberedt med eventuelle oppdateringer eller spørsmål du måtte ha. Ser frem til å se dere alle der!\" from the perspective of the sender in norwegian.  Only include the important information"},
                        {"role": "assistant", "content": "Påminnelse om møte i konferanserommet kl. 10 imorgen."},
                        {"role": "user", "content": "create a short summary of the following email body:\n\"" + texts[i] + "\" from the perspective of the sender in norwegian.  Only include the important information"}]
            if context == THREE_SHOT:
                message_list = [{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "create a short summary of the following email body:\n\"Vi ønsker å informere deg om et fremtidig prosjekt i Tokyo som vil fokusere på utvikling av nye teknologier og tjenester innen vår bransje. Ta kontakt med oss hvis du er interessert i å diskutere mulighetene nærmere. QUALCOMM innovasjon.\" from the perspective of the sender in norwegian.  Only include the important information"},
                        {"role": "assistant", "content": "Informasjon om fremtidig prosjekt i Tokyo."},
                        {"role": "user", "content": "create a short summary of the following email body:\n\"Hei alle sammen, vi ønsker å informere dere om en endring i arbeidstiden fra mandag denne uken. Vi vil nå begynne en time tidligere om morgenen, kl. 8.00. Dette vil gi oss mer tid til å fullføre prosjektene våre innen fristen. Takk for forståelsen!\" from the perspective of the sender in norwegian.  Only include the important information"},
                        {"role": "assistant", "content": "Endring i arbeidstiden fra mandag denne uken til å begynne kl. 8.00."},
                        {"role": "user", "content": "create a short summary of the following email body:\n\"Hei Team, Bare en rask påminnelse om at vi har et møte planlagt for i morgen kl. 10 i konferanserommet. Vennligst vær forberedt med eventuelle oppdateringer eller spørsmål du måtte ha. Ser frem til å se dere alle der!\" from the perspective of the sender in norwegian.  Only include the important information"},
                        {"role": "assistant", "content": "Påminnelse om møte i konferanserommet kl. 10 imorgen."},
                        {"role": "user", "content": "create a short summary of the following email body:\n\"" + texts[i] + "\" from the perspective of the sender in norwegian.  Only include the important information"}]
                
                response = openai.ChatCompletion.create(model=model_name, messages=message_list, temperature=0.3, max_tokens=500, stop="###")
                summary = response["choices"][0]["message"]["content"]
        else:
            if mode == TEXT_COMPLETION:
                if context == ZERO_SHOT:
                    sum_prompt = "Skriv et sammendrag av eposten: \"" + texts[i] + "\" på norsk"
                if context == ONE_SHOT:
                    sum_prompt = sum_1_shot + texts[i] + "->->"
                if context == THREE_SHOT:
                    sum_prompt = sum_3_shot + texts[i] + "->->"
            else: #Mode is fine tuned
                
                if include_subjects:
                    sum_prompt = subjects[i] + " $$$ " + texts[i] + "->->"
                else: 
                    sum_prompt = texts[i] + "->->"
            response = openai.Completion.create(model="davinci:ft-edialog24-as-2023-05-07-19-34-37", prompt=sum_prompt, temperature=0.3, max_tokens=500, stop="###")
            summary = response["choices"][0]["text"]
            if include_subjects:
                summary = summary.replace("$$$", "")
        pred.append(summary)
        print(summary)
        i += 1
    except:
        failures += 1
        print("Failed")


        
print("Score vs human generated gold labels:")
print(rg.compareRouge(gold_labels, pred))
print()
print("Score vs gpt-generated labels:")
print(rg.compareRouge(labels, pred))




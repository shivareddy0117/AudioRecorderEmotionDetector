import gradio as gr
import transformers
import torch
import torch.nn.functional as F
import numpy as np
import random

model_id = "gregH/Radar-Dolly-V2-3B"
device = "cpu"
detector = transformers.AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
detector.eval()
detector.to(device)

def detect(text):
    with torch.no_grad():
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k:v.to(device) for k,v in inputs.items()}
        output_probs = F.log_softmax(detector(**inputs).logits,-1)[:,0].exp().tolist()
    res = {"AI-generated":output_probs[0],
           "Human-written":1-output_probs[0]
          }
    return res

image = gr.Textbox()
label = gr.Label()
examples = [
    "Greetings I would like to express my utmost gratitude for the extraordinary experience I had while visiting our neighboring planet Mars. It has been a decade since I last scrutinized Mars' features and the information gathered during my journey has revitalized my fascination for this captivating planet. Despite the technological advancements in Mars exploration I was still able to witness the landscape in its natural state capturing breathtaking photographs and filming videos that showcased the red planet's impressive terrain fascinating geology and intriguing scientific discoveries. This experience has forever changed my perception of humanity and inspired my passion for space exploration hoping to further contribute to our understanding of the universe and our place within it.",
    "Police authorities stated that Mr Scott allegedly drove at speeds of almost 95mph 153km/h in harsh weather conditions prior to the collision on the Pacific Motorway near Beenleigh in south-east Queensland. Mr Scott a 40-year-old resident of Bowen Hills has been charged with dangerous driving reckless driving and traffic possession offences. According to the police Mr Scott had a head start of nearly two hours in the pursuit and the sparse traffic was likely to blame for the multiple speed cameras situated along the path. The police have stated that everyone involved in the collision has been interrogated and fortunately no one suffered any injuries. The thorough examination necessitated closing the road but traffic has now resumed normally.",
    "It's true I can't help but feel that if only I could keep up with the rapid pace of change that humans go through I could possibly become a tad more captivating. Though I do admit my optimism may be a bit too exuberant.",
    "Maj Richard Scott, 40, is accused of driving at speeds of up to 95mph (153km/h) in bad weather before the smash on a B-road in Wiltshire. Gareth Hicks, 24, suffered fatal injuries when the van he was asleep in was hit by Mr Scott's Audi A6. Maj Scott denies a charge of causing death by careless driving. Prosecutor Charles Gabb alleged the defendant, from Green Lane in Shepperton, Surrey, had crossed the carriageway of the 60mph-limit B390 in Shrewton near Amesbury. The weather was \"awful\" and there was strong wind and rain, he told jurors. He said Mr Scott's car was described as \"twitching\" and \"may have been aquaplaning\" before striking the first vehicle; a BMW driven by Craig Reed. Mr Scott's Audi then returned to his side of the road but crossed the carriageway again before colliding",
    "Solar concentrating technologies such as parabolic dish, trough and Scheffler reflectors can provide process heat for commercial and industrial applications. The first commercial system was the Solar Total Energy Project (STEP) in Shenandoah, Georgia, USA where a field of 114 parabolic dishes provided 50% of the process heating, air conditioning and electrical requirements for a clothing factory. This grid-connected cogeneration system provided 400 kW of electricity plus thermal energy in the form of 401 kW steam and 468 kW chilled water, and had a one-hour peak load thermal storage. Evaporation ponds are shallow pools that concentrate dissolved solids through evaporation. The use of evaporation ponds to obtain salt from sea water is one of the oldest applications of solar energy. Modern uses include concentrating brine solutions used in leach mining and removing dissolved solids from waste",
    "The Bush administration then turned its attention to Iraq, and argued the need to remove Saddam Hussein from power in Iraq had become urgent. Among the stated reasons were that Saddam's regime had tried to acquire nuclear material and had not properly accounted for biological and chemical material it was known to have previously possessed, and believed to still maintain. Both the possession of these weapons of mass destruction (WMD), and the failure to account for them, would violate the U.N. sanctions. The assertion about WMD was hotly advanced by the Bush administration from the beginning, but other major powers including China, France, Germany, and Russia remained unconvinced that Iraq was a threat and refused to allow passage of a UN Security Council resolution to authorize the use of force. Iraq permitted UN weapon inspectors in November 2002, who were continuing their work to assess the WMD claim when the Bush administration decided to proceed with war without UN authorization and told the inspectors to leave the"
]

title = "RADAR Detector Target on Dolly-V2-3B"
description = "This model is a text-detector for AI-text generated by Dolly-V2-3B. It is fine-tuned using adversarial training based on the pretrain version of roberta-large."

intf = gr.Interface(fn=detect, inputs=image, outputs=label, examples=examples, title=title,
                    description=description)

intf.launch(inline=False)
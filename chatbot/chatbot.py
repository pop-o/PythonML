import spacy

#loading the medium english language model
nlp=spacy.load("en_core_web_md")

def process_input(user_input):
    #process the input using spacy
    doc=nlp(user_input)

    #Extract named entities 
    entities={ent.label_:ent.text for ent in doc.ents}
    return doc,entities

def generate_response(doc,entities):
    if "PERSON" in entities:
        return f"Hello {entities['PERSON']}! How can I assist you today?"
    elif "GPE" in entities:  # GPE stands for Geopolitical Entity (like a city, country)
        return f"{entities['GPE']} is a great place! What would you like to know about it?"
    elif "DATE" in entities:
        return f"Ah, {entities['DATE']} is an interesting date. Do you have any plans?"
    elif "ORG" in entities:
        return f"{entities['ORG']} is a well-known organization. What would you like to discuss about it?"
    elif "when" in doc.text.lower():
        return "Genshin impact was launched in September of 2020"
    elif "download" in doc.text.lower():
        return f"You can download Genshin impact from the link {'https://genshin.hoyoverse.com/en/'}"
    elif "size" in doc.text.lower():
        return "Genshin Impact has a size of about 90 gigabytes"
    elif "large" in doc.text.lower():
        return "Genshin Impact has a size of about 90 gigabytes"
    elif "how fun" in doc.text.lower():
        return "Genshin impact is a very fun game"
    elif "type" in doc.text.lower():
        return "Genshin impact is an open-world Adventure RPG game."
    elif "multiplayer" in doc.text.lower():
        return "Genshin impact is a single player game but you can play with your friends using the co-op feature"
    elif "developed" in doc.text.lower():
        return "Genshin impact was developed by MIHOYO"
    else:
        return "Sorry I can't process what you are saying right now"
    
def chatbot_response(user_input):
    doc,entities=process_input(user_input)
    response=generate_response(doc,entities)
    return response

if __name__=="__main__":
    print("This is a Genshin Impact Chatbot. Type 'exit' to exit.")
    while(True):
        user_input=input("Question: ")
        if user_input.lower()=="exit":
            print("Thank you for your time.")
            break
        
        response=chatbot_response(user_input)
        print("Answer: ",response)

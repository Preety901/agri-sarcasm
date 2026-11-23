"""Language detection (English / Hindi / Hinglish) and reply texts in the user's language."""
import re

HINGLISH_WORDS = set("""
hai hain tha thi hoga hogi hua hui kya kaise kaun kaunsa kab kahan kitna kitni kitne kyun
nahi nahin toh mein se par pe aur ya ka ki ke ko ne wah wahi yeh ye woh wo mera meri mere
aapka aapki aapke mujhe humein hum tum aap main kar karo karein karun kare karna raha rahi rahe
gaya gayi gaye aaya aayi aaye bahut bohot badhiya accha acha achi achhi aadhi pura poora sab
sarkar fasal khaad khad beej dawai dawa keede mitti paani pani bijli nahar bhaav
bhav gehun gehu dhaan dhan sarson aalu pyaaz ganna makka tamatar baarish baadh sookh sookha
sookh ole khet khate paisa rupaye kist bima jaanch sinchai upjaau tarika samay bataiye
batao bataye dijiye chahiye milta mila mili mile lagat wada saal mahine ghante baje kaam
madad dukaan dukaandaar keede daalein rakhein roke sahi kaise liye jaldi kitni bhi tak
abhi phir tab jab itne itna itni kuch koi sirf bilkul kismat meherbani kamaal asardaar
""".split())

def detect(text: str) -> str:
    """Returns 'hi' (Devanagari), 'hn' (Hinglish, Roman script) or 'en'."""
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    toks = re.findall(r"[a-z]+", text.lower())
    if not toks:
        return "en"
    hits = sum(t in HINGLISH_WORDS for t in toks)
    return "hn" if hits >= 2 or hits / len(toks) >= 0.25 else "en"

LANG_NAME = {"en": "English", "hi": "हिन्दी", "hn": "Hinglish"}
LLM_LANG = {"en": "English", "hi": "Hindi (Devanagari script)",
            "hn": "Hinglish (Hindi written in Roman/English letters)"}

UI = {
 "en": dict(prob="sarcasm probability", text_only="Text-only score", photo="Photo damage score",
            photo_note="Crop damage is separate from sarcasm; the photo adjusts this score only when the text has positive-sentiment cues.",
            incon="positive words + damaged crop = incongruity", answer="Advisory answer",
            sarc=("🚩 Sarcastic grievance", "Routed to the priority expert desk",
                  "I can sense frustration in your message. Your grievance has been sent to an agricultural expert, who will contact you soon."),
            lit=("✅ Literal query", "Sent to the advisory system",
                 "Thank you for your question. It has been sent to the advisory system and you will get guidance shortly.")),
 "hi": dict(prob="व्यंग्य की संभावना", text_only="केवल टेक्स्ट स्कोर", photo="फोटो में फसल खराबी स्कोर",
            photo_note="फसल की खराबी का स्कोर अलग है; फोटो व्यंग्य स्कोर को तभी बदलती है जब टेक्स्ट में सकारात्मक भाव के संकेत हों।",
            incon="सकारात्मक शब्द + खराब फसल = असंगति", answer="सलाह",
            sarc=("🚩 व्यंग्यात्मक शिकायत", "प्राथमिकता वाले विशेषज्ञ डेस्क को भेजा गया",
                  "आपकी बात में नाराज़गी झलक रही है। आपकी शिकायत कृषि विशेषज्ञ को भेज दी गई है, वे जल्द ही आपसे संपर्क करेंगे।"),
            lit=("✅ सीधा सवाल", "सलाह प्रणाली को भेजा गया",
                 "आपके प्रश्न के लिए धन्यवाद। इसे सलाह प्रणाली को भेज दिया गया है, जल्द ही आपको मार्गदर्शन मिलेगा।")),
 "hn": dict(prob="vyangya ki sambhavna", text_only="Sirf text ka score", photo="Photo mein fasal kharabi score",
            photo_note="Fasal kharabi ka score alag hai; photo vyangya score ko tabhi badalti hai jab text mein positive-bhav ke cues hon.",
            incon="positive shabd + kharab fasal = incongruity", answer="Salah",
            sarc=("🚩 Vyangya wali shikayat", "Priority expert desk ko bheja gaya",
                  "Aapki baat mein naraazgi dikh rahi hai. Aapki shikayat krishi visheshagya ko bhej di gayi hai, woh jaldi aapse sampark karenge."),
            lit=("✅ Seedha sawal", "Salah pranali ko bheja gaya",
                 "Aapke sawal ke liye dhanyavaad. Ise salah pranali ko bhej diya gaya hai, jaldi margdarshan milega.")),
}
# What the Mac 'say' command reads out (Roman Hinglish is read badly, so speak the Hindi text).
SPOKEN = {"sarc": UI["hi"]["sarc"][2], "lit": UI["hi"]["lit"][2]}
VOICE = {"en": "Samantha", "hi": "Lekha", "hn": "Lekha"}
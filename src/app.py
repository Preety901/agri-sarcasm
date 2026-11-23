"""Streamlit demo:  streamlit run src/app.py"""
import base64, hashlib, io, os, sys, subprocess, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd, streamlit as st
from src.embed import embed
from src.models import EmbLR
from src.lang import detect, LANG_NAME, LLM_LANG, UI, SPOKEN, VOICE

st.set_page_config(page_title="Agri Sarcasm Detector", page_icon="🌾",
                   layout="wide", initial_sidebar_state="collapsed")

def b64svg(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

# Framed field illustration for the hero.
ART_HERO = b64svg("""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 380">
<rect width="300" height="380" fill="#E9E4DC"/>
<g transform="rotate(-7 150 190)">
<rect x="36" y="56" width="218" height="262" fill="#FFFFFF" stroke="#EFEAE2"/>
<rect x="54" y="74" width="182" height="226" fill="#F4DCCB"/>
<circle cx="146" cy="150" r="36" fill="#B8654A"/>
<circle cx="190" cy="118" r="9" fill="#EBC9A8"/>
<path d="M54 252 Q110 202 150 238 T236 226 V300 H54Z" fill="#9DA78C"/>
<path d="M54 280 Q120 244 236 272 V300 H54Z" fill="#6F7F66"/>
<g stroke="#D9A441" stroke-width="2.2" stroke-linecap="round"><line x1="92" y1="296" x2="92" y2="262"/><line x1="110" y1="296" x2="110" y2="256"/><line x1="128" y1="296" x2="128" y2="264"/></g>
<g fill="#D9A441"><ellipse cx="92" cy="258" rx="4" ry="9"/><ellipse cx="110" cy="252" rx="4" ry="9"/><ellipse cx="128" cy="260" rx="4" ry="9"/></g>
</g>
<rect x="168" y="292" width="108" height="66" rx="14" fill="#2B2724"/>
<polygon points="246,356 246,376 224,356" fill="#2B2724"/>
<circle cx="200" cy="325" r="6" fill="#EFE9DD"/><circle cx="222" cy="325" r="6" fill="#EFE9DD"/><circle cx="244" cy="325" r="6" fill="#EFE9DD"/>
</svg>""")

# Field illustration for the How It Works section.
ART_ABOUT = b64svg("""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 340">
<rect width="300" height="340" fill="#EDE8E1"/>
<rect x="30" y="28" width="240" height="190" fill="#FFFFFF"/>
<rect x="38" y="36" width="224" height="174" fill="#DCE5E6"/>
<line x1="150" y1="36" x2="150" y2="210" stroke="#FFFFFF" stroke-width="5"/><line x1="38" y1="122" x2="262" y2="122" stroke="#FFFFFF" stroke-width="5"/>
<circle cx="210" cy="78" r="20" fill="#E9B98F"/>
<path d="M38 170 Q100 136 160 164 T262 156 V210 H38Z" fill="#A9B39A"/>
<path d="M38 192 Q120 170 262 190 V210 H38Z" fill="#7B8A70"/>
<rect x="60" y="236" width="180" height="70" rx="6" fill="#E3DACC"/>
<g stroke="#B79B54" stroke-width="2.6" stroke-linecap="round"><line x1="104" y1="300" x2="104" y2="250"/><line x1="130" y1="300" x2="130" y2="242"/><line x1="156" y1="300" x2="156" y2="250"/><line x1="182" y1="300" x2="182" y2="244"/></g>
<g fill="#D2A94A"><ellipse cx="104" cy="244" rx="5" ry="11"/><ellipse cx="130" cy="236" rx="5" ry="11"/><ellipse cx="156" cy="244" rx="5" ry="11"/><ellipse cx="182" cy="238" rx="5" ry="11"/></g>
</svg>""")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+SC:wght@500;600;700&family=Caveat:wght@500;600&family=Lato:wght@400;500;700&display=swap');
#MainMenu, footer, .stAppDeployButton, [data-testid="stToolbar"] {display:none !important;}
header[data-testid="stHeader"] {background: transparent; height:0;}
.stApp {background:#FFFFFF;}
[data-testid="stAppViewContainer"], .stApp {overflow-x:hidden;}
.block-container {max-width: 1040px !important; padding: 0 24px 3rem 24px !important;}
.band {width:100vw; position:relative; left:50%; margin-left:-50vw; background:#EDEAE5; padding:0 0 24px 0;
       font-family:'Lato',sans-serif;}
.band .in {max-width:1040px; margin:0 auto; padding:0 24px;}
.nav {display:flex; align-items:center; justify-content:space-between; gap:20px; padding:18px 0 6px 0;}
.nav .logo {font-family:'Caveat',cursive; font-size:2rem; font-weight:600; color:#2B2724; line-height:1;}
.nav .links a {font-size:.78rem; font-weight:500; color:#2B2724 !important; text-decoration:none !important; margin-left:30px;}
.nav .links a:hover {text-decoration:underline !important;}
.hero {display:flex; align-items:center; gap:70px; flex-wrap:wrap; padding-top:42px;}
.hero .txt {flex:1 1 360px; padding-bottom:22px;}
.hero h1 {font-family:'Cormorant SC',Georgia,serif; font-size:2.25rem; font-weight:700; line-height:1.2;
          letter-spacing:.015em; color:#2B2724; margin:0 0 16px 0; text-shadow:0 1px 2px rgba(0,0,0,.12);}
.hero p {color:#514B45; font-size:.88rem; line-height:1.8; max-width:430px; margin:0 0 26px 0;}
.btn {display:inline-block; background:#222222; color:#FFFFFF !important; padding:13px 40px; font-size:.78rem; font-weight:500;
      text-decoration:none !important; border-radius:1px; box-shadow:0 2px 6px rgba(0,0,0,.16); transition:background .2s ease;}
.btn:hover {background:#3A3632;}
.hero img.art {flex:0 0 300px; width:300px; margin-bottom:-84px; position:relative; z-index:2; display:block;}
.after {height:96px;}
.sec {font-family:'Cormorant SC',Georgia,serif; font-size:2rem; font-weight:700; color:#2B2724; letter-spacing:.02em; margin:0 0 18px 0;}
.about {display:flex; gap:80px; align-items:center; flex-wrap:wrap; margin:46px 0 10px 0; font-family:'Lato',sans-serif;}
.about img {flex:0 0 300px; width:300px;}
.about .t {flex:1 1 360px;}
.about p {font-size:.86rem; line-height:1.85; color:#514B45; margin:0 0 12px 0;}
.about .line {width:110px; border-top:1px solid #2B2724; margin-top:18px;}
.lbl, .res, .small, .ph, .hist, .reply, .ttl {font-family:'Lato',sans-serif;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#FFFFFF; border:1px solid #E6E1D8 !important;
       border-radius:2px !important; box-shadow:0 4px 18px rgba(43,39,36,.045); padding:.8rem 1rem;}
.lbl {font-size:.66rem; letter-spacing:.18em; color:#8F897F; font-weight:700; text-transform:uppercase; margin-bottom:2px;}
.ttl {font-family:'Cormorant SC',Georgia,serif; font-weight:700; color:#2B2724; font-size:1.5rem; margin-bottom:10px; letter-spacing:.02em;}
.small {color:#8F897F; font-size:.78rem;}
.res {background:#FFFFFF; border:1px solid #E6E1D8; border-left:4px solid; padding:20px 22px;}
.res.sarc {border-left-color:#A9714A;} .res.lit {border-left-color:#7C7C78;}
.badge {font-size:.7rem; letter-spacing:.16em; text-transform:uppercase; font-weight:700;}
.sarc .badge {color:#8A5530;} .lit .badge {color:#55554F;}
.big {font-family:'Cormorant SC',Georgia,serif; font-size:3.6rem; font-weight:700; margin:2px 0 0 0; color:#2B2724; line-height:1.05;}
.bar {height:3px; background:#ECE7DE; overflow:hidden; margin:10px 0 14px 0;}
.sarc .bar > div {height:100%; background:#A9714A;} .lit .bar > div {height:100%; background:#7C7C78;}
.route {font-weight:700; color:#2B2724; font-size:.9rem;}
.reply {background:#F6F3EE; padding:12px 14px; margin-top:10px; color:#2B2724; font-size:.86rem; line-height:1.7;}
.ph {text-align:center; padding:84px 10px; color:#8F897F; font-size:.88rem; line-height:1.8;}
.photo {width:100%; max-height:190px; object-fit:cover; margin-top:12px; border:1px solid #E6E1D8;}
.hist {width:100%; border-collapse:collapse; font-size:.82rem;}
.hist th {text-align:left; color:#8F897F; padding:8px 12px; font-weight:700; letter-spacing:.1em; font-size:.68rem; text-transform:uppercase;}
.hist td {padding:8px 12px; border-top:1px solid #EFEBE2; color:#2B2724;}
.tag {font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; font-weight:700;}
.tag.s {color:#8A5530;} .tag.l {color:#55554F;}
.stButton > button {border-radius:1px; font-weight:500; border:1px solid #D9D3C6; background:#FFFFFF; color:#2B2724; font-size:.8rem; transition:all .15s ease;}
.stButton > button:hover {border-color:#222222; color:#222222;}
.stButton > button[kind="primary"] {background:#222222; color:#FFFFFF; border:none; height:2.9rem; letter-spacing:.06em;
        box-shadow:0 2px 6px rgba(0,0,0,.16);}
.stButton > button[kind="primary"]:hover {background:#3A3632; color:#FFFFFF;}
div[data-testid="stTextArea"] textarea {border-radius:1px; background:#FAF9F6; border:1px solid #E6E1D8; font-size:.98rem;}
[data-testid="stFileUploaderDropzone"] {background:#FAF9F6; border:1px dashed #D5CFC2; border-radius:1px;}
[data-testid="stExpander"] {background:#FFFFFF; border:1px solid #E6E1D8; border-radius:2px;}
@media (max-width: 700px) {
  .block-container {padding:0 18px 2rem 18px !important;}
  .band .in {padding:0 18px;}
  .nav {align-items:flex-start; flex-direction:column; gap:12px;}
  .nav .links {display:flex; flex-wrap:wrap; gap:10px 18px;}
  .nav .links a {margin-left:0;}
  .hero {gap:24px; padding-top:30px;}
  .hero .txt {flex-basis:100%; padding-bottom:0;}
  .hero h1 {font-size:1.9rem;}
  .hero img.art {flex-basis:min(300px, 100%); width:min(300px, 100%); margin:0 auto -72px;}
  .after {height:84px;}
  .about {gap:28px; margin-top:34px;}
  .about img {flex-basis:min(300px, 100%); width:min(300px, 100%); margin:0 auto;}
}
@media (max-width: 420px) {
  .hero h1 {font-size:1.65rem;}
  .sec {font-size:1.75rem;}
  [data-testid="stVerticalBlockBorderWrapper"] {padding:.6rem .65rem;}
}
html {scroll-behavior:smooth;}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="band" id="top"><div class="in">
  <div class="nav">
    <span class="logo">Agri Sarcasm</span>
    <span class="links"><a href="#top">Home</a><a href="#try">Analyse</a><a href="#how">How it works</a><a href="#history">History</a></span>
  </div>
  <div class="hero">
    <div class="txt">
      <h1>When words hide a story, we listen.</h1>
      <p>Sarcasm can hide real frustration. Share a farming query in English, हिन्दी or Hinglish to uncover its meaning
      and find the right next step.</p>
      <a class="btn" href="#try">Try it now</a>
    </div>
    <img class="art" src="{ART_HERO}">
  </div>
</div></div>
<div class="after"></div>
<div id="try" class="sec">Share what happened</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load(k):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "agri_sarcasm_seed.csv"))
    tr = pd.concat([df[df.label == c].sample(k, random_state=0) for c in (0, 1)])
    return EmbLR(embed, True).fit(tr.text.tolist(), tr.label.tolist())

WHISPER = {"Base (fast, ~290 MB)": "openai/whisper-base", "Tiny (fastest, ~150 MB)": "openai/whisper-tiny",
           "Small (best, ~970 MB)": "openai/whisper-small"}
SPEECH_LANGS = {"Auto-detect": None, "Hindi": "hindi", "English": "english"}
with st.sidebar:
    st.subheader("Settings")
    k = st.slider("Few-shot examples per class (k)", 1, 20, 8)
    speak = st.checkbox("Speak the reply", True)
    use_llm = st.checkbox("Advisory answer with local LLM (needs Ollama)", False)
    wname = st.selectbox("Speech model", list(WHISPER), 1)
    slang = st.selectbox("Speech language", list(SPEECH_LANGS), 1)
model = load(k)

EXAMPLES = {"Hinglish": "Wah re sarkar, khaad toh aadhi mitti nikli!",
            "हिन्दी": "बीमा का पैसा सिर्फ दो साल में मिल गया, कितनी जल्दी!",
            "English": "Great, the helpline answers only after my crop has died.",
            "Normal question": "Gehun ke liye sabse achi khaad kaun si hai?"}
if "query" not in st.session_state:
    st.session_state.query = EXAMPLES["Hinglish"]
if "history" not in st.session_state:
    st.session_state.history = []
def set_q(t): st.session_state.query = t

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="ttl">Your query</div>', unsafe_allow_html=True)
        mode = st.radio("Input", ["Type", "Speak"], horizontal=True, label_visibility="collapsed")
        q = ""
        if mode == "Type":
            q = st.text_area("Query", key="query", height=120, label_visibility="collapsed",
                             placeholder="English / हिन्दी / Hinglish")
            with st.expander("Try an example"):
                examples = list(EXAMPLES.items())
                for start in range(0, len(examples), 2):
                    cols = st.columns(2)
                    for c, (name, txt) in zip(cols, examples[start:start + 2]):
                        c.button(name, on_click=set_q, args=(txt,), use_container_width=True)
        else:
            audio = st.audio_input("Record", sample_rate=16000, label_visibility="collapsed")
            if audio is not None:
                data = audio.getvalue()
                key = hashlib.md5(data).hexdigest() + WHISPER[wname] + str(SPEECH_LANGS[slang])
                if st.session_state.get("tr_key") != key:   # transcribe once per recording
                    st.session_state["tr_key"] = key
                    try:
                        from src.speech import transcribe
                        with st.spinner("Transcribing… (first time downloads the speech model)"):
                            st.session_state["transcript"] = transcribe(data, WHISPER[wname], SPEECH_LANGS[slang])
                    except Exception as e:
                        st.session_state["transcript"] = ""
                        st.error(f"Transcription failed: {e}")
                q = st.text_area("Transcript (edit if wrong)", key="transcript", height=110)
        img = st.file_uploader("Crop photo (optional)", type=["jpg", "jpeg", "png"])
        go = st.button("Analyse", type="primary", use_container_width=True)

with right:
    with st.container(border=True):
        st.markdown('<div class="ttl">A closer look</div>', unsafe_allow_html=True)
        if go and q.strip():
            lg = detect(q); T = UI[lg]
            p = float(model.predict_proba([q])[0]); p_text = p; extra = ""; photo_html = ""; photo_note = ""
            if img is not None:
                from PIL import Image
                from src.image_incongruity import damage_score, cross_modal_incongruity
                im = Image.open(img).convert("RGB")
                with st.spinner("Analysing photo… (first time downloads CLIP)"):
                    dmg = damage_score(im); inc = cross_modal_incongruity(q, im)
                p = min(1.0, p + 0.3 * inc)
                extra = f" · {T['photo']} {dmg:.2f}" + (f" · {T['incon']}" if inc > 0 else "")
                photo_note = f'<div class="small" style="margin-top:8px;">{T["photo_note"]}</div>'
                small = im.copy(); small.thumbnail((700, 700)); buf = io.BytesIO(); small.save(buf, "JPEG", quality=85)
                photo_html = f'<img class="photo" src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}">'
            sarc = p >= 0.5
            label, route, msg = T["sarc" if sarc else "lit"]
            label = label.replace("🚩 ", "").replace("✅ ", "")
            answer_html = ""
            if use_llm and not sarc:
                try:
                    from src.llm_fewshot import advise
                    with st.spinner("Writing advisory answer…"):
                        ans = advise(q, LLM_LANG[lg])
                    answer_html = f'<div class="reply"><b>{T["answer"]}:</b> {ans}</div>'
                except Exception as e:
                    answer_html = f'<div class="reply small">LLM not available ({e.__class__.__name__}). Start Ollama to enable.</div>'
            st.markdown(f"""
            <div class="res {'sarc' if sarc else 'lit'}">
              <span class="badge">{label}</span>
              <div class="big">{p*100:.0f}%</div><div class="small">{T['prob']}</div>
              <div class="bar"><div style="width:{p*100:.0f}%"></div></div>
              <div class="route">{route}</div>
              <div class="reply">{msg}</div>{answer_html}
              <div class="small" style="margin-top:10px;">{T['text_only']}: {p_text*100:.0f}%{extra}</div>{photo_note}
              {photo_html}
            </div>""", unsafe_allow_html=True)
            st.session_state.history.insert(0, dict(t=time.strftime("%H:%M:%S"), q=q[:90], lg=LANG_NAME[lg],
                                                    sarc=sarc, p=p))
            if speak:
                spoken_text = msg if lg == "en" else SPOKEN["sarc" if sarc else "lit"]
                try:
                    subprocess.Popen(["say", "-v", VOICE[lg], spoken_text])
                except OSError:
                    pass  # 'say' exists only on macOS
                try:
                    from gtts import gTTS
                    tts_lang = "en" if lg == "en" else "hi"
                    mp3_buf = io.BytesIO()
                    gTTS(text=spoken_text, lang=tts_lang).write_to_fp(mp3_buf)
                    mp3_buf.seek(0)
                    st.audio(mp3_buf.getvalue(), format="audio/mp3", autoplay=True)
                except Exception as e:
                    st.caption(f"Voice playback unavailable: {e}")
        else:
            st.markdown('<div class="ph">Your result will appear here.<br>Enter a query on the left and press <b>Analyse</b>.</div>',
                        unsafe_allow_html=True)

st.markdown(f"""
<div id="how" class="about">
  <img src="{ART_ABOUT}">
  <div class="t">
    <div class="sec">How it works</div>
    <p>Type a query, record your voice, or add a crop photo. The model reads the words in English, हिन्दी or Hinglish
    and checks for the gap between what is said and what is happening.</p>
    <p>It identifies likely sarcasm and routes a complaint to the priority desk. For a regular question, it can offer
    an advisory response in the same language.</p>
    <div class="line"></div>
  </div>
</div>
<div id="history" style="height:10px"></div>
""", unsafe_allow_html=True)

with st.expander("Recent analyses"):
    H = st.session_state.history[:8]
    if H:
        rows = "".join(
            f"<tr><td>{h['t']}</td><td>{h['q']}</td><td>{h['lg']}</td>"
            f"<td><span class='tag {'s' if h['sarc'] else 'l'}'>{'Sarcastic' if h['sarc'] else 'Normal'}</span></td>"
            f"<td>{h['p']*100:.0f}%</td></tr>" for h in H)
        st.markdown(f"<table class='hist'><tr><th>Time</th><th>Query</th><th>Language</th><th>Result</th><th>Score</th></tr>{rows}</table>",
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="small">No analyses yet.</div>', unsafe_allow_html=True)
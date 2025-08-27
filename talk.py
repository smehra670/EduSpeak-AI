

from flask import Flask, request, render_template_string, send_file, after_this_request
from openai import OpenAI
import os, re, uuid, tempfile
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq



app = Flask(__name__)
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


enhance_agent = Agent(
    name="Language Enhancement Agent",
    role="Text Refiner and Grammar Corrector",
    model=Groq(id=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")),
    instructions=[
        "Fix grammar/tense, punctuation, and clarity. Keep meaning.",
        "Return ONLY the improved text (no explanations)."
    ],
    show_tool_calls=False, markdown=False, stream=False
)

translate_agent = Agent(
    name="Translator",
    role="Translate English to requested language",
    model=Groq(id=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")),
    instructions=[
        "Translate English to the requested language.",
        "If none specified, default to Spanish.",
        "Return ONLY the translation."
    ],
    show_tool_calls=False, markdown=False, stream=False
)

def clean_output(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def h(text: str) -> str:
    """Basic HTML escape for safe rendering inside <div>."""
    text = text or ""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
    )


def render_page(content, title, active_page, **kwargs):
    template = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>{{ title }} | EduSpeak AI</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">
<style>
:root{
  --bg:#f6f8ff;              /* light background */
  --ink:#0f172a;             /* dark text */
  --muted:#4b5563;           /* secondary text */
  --card:#ffffff;            /* card surface */
  --line:#e5e7eb;            /* hairline */
  --primary:#6366f1;         /* indigo */
  --accent:#10b981;          /* green */
  --grad:linear-gradient(135deg, #7c8cfb, #6ee7b7);
  --shadow:0 8px 30px rgba(17,24,39,.08);
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0; color:var(--ink); background:
    radial-gradient(1200px 600px at 10% 10%,rgba(124,140,251,.15),transparent 60%),
    radial-gradient(800px 500px at 90% 0%,rgba(110,231,183,.10),transparent 60%),
    var(--bg);
  font-family:'Plus Jakarta Sans', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}
/* NAV */
.nav{position:fixed; inset:0 0 auto 0; z-index:50; background:rgba(255,255,255,.85); backdrop-filter:blur(12px); border-bottom:1px solid var(--line)}
.wrap{max-width:1100px; margin:0 auto; display:flex; align-items:center; justify-content:space-between; padding:12px 18px}
.brand{display:flex; gap:.7rem; align-items:center; text-decoration:none; color:var(--ink); font-weight:800}
.badge{width:38px; height:38px; border-radius:12px; background:var(--grad); display:grid; place-items:center; color:white; box-shadow:var(--shadow)}
.links a{color:var(--muted); text-decoration:none; padding:.55rem .9rem; border-radius:12px}
.links a:hover, .links a.active{background:#eef2ff; color:#3730a3}
/* MAIN */
main{padding:92px 16px 40px}
.container{max-width:1100px; margin:0 auto}
.card{background:var(--card); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow); padding:22px}
.btn{border:0; border-radius:12px; padding:.9rem 1.2rem; font-weight:700; cursor:pointer; display:inline-flex; gap:.6rem; align-items:center; transition:.2s}
.btn-primary{background:var(--grad); color:#fff; box-shadow:0 8px 24px rgba(99,102,241,.25)}
.btn-outline{background:#fff; color:#3730a3; border:2px solid #e0e7ff}
.btn-success{background:linear-gradient(135deg,#22c55e,#60a5fa); color:#fff}
.btn:active{transform:translateY(1px)}
/* HERO (simplified, no code window) */
.hero{display:grid; grid-template-columns:1.2fr .8fr; gap:22px; align-items:center}
.hero .left{padding:8px}
.kicker{display:inline-flex; gap:.5rem; align-items:center; color:#2563eb; background:#e0f2fe; border:1px solid #bfdbfe; padding:.35rem .65rem; border-radius:999px; font-weight:700; font-size:.85rem}
.title{font-size:clamp(34px,6vw,56px); line-height:1.05; margin:.3rem 0; color:#0f172a}
.subtitle{color:var(--muted); font-size:1.05rem; max-width:55ch}
.hero .right{display:grid; gap:14px}
.snapshot{background:var(--card); border:1px solid var(--line); border-radius:16px; padding:18px; box-shadow:var(--shadow)}
.snapshot h4{margin:0 0 8px 0}
.statgrid{display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:18px 0}
.stat{background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px; text-align:center; box-shadow:var(--shadow)}
.stat .num{font-weight:800; font-size:1.6rem}
/* Features */
.features{display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px}
.feature{background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px; box-shadow:var(--shadow)}
.iconbox{width:44px; height:44px; border-radius:10px; background:var(--grad); display:grid; place-items:center; color:#fff}
/* Forms */
.input, textarea, select{width:100%; background:#fff; border:1px solid var(--line); color:var(--ink); border-radius:12px; padding:12px; font-family:inherit}
textarea{min-height:110px}
.textbox{white-space:pre-wrap; background:#fff; border:1px solid var(--line); border-radius:12px; padding:12px; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, "JetBrains Mono", monospace; max-height:380px; overflow:auto}
.textbox.ok{border-color:#86efac; background:#f0fdf4}
.grid{display:grid; gap:14px}
.grid-2{grid-template-columns:repeat(auto-fit,minmax(360px,1fr))}
.err{background:#fef2f2; border:1px solid #fecaca; padding:10px 12px; border-radius:12px; color:#991b1b}
.footer{margin-top:30px; color:var(--muted); text-align:center}
@media (max-width:980px){ .hero{grid-template-columns:1fr} .statgrid{grid-template-columns:repeat(2,1fr)} }
</style>
</head>
<body>
  <nav class="nav">
    <div class="wrap">
      <a class="brand" href="/">
        <div class="badge"><i class="fas fa-graduation-cap"></i></div>
        <span>EduSpeak AI</span>
      </a>
      <div class="links">
        <a href="/" class="{{ 'active' if active_page=='home' else '' }}">Home</a>
        <a href="/transcribe" class="{{ 'active' if active_page=='transcribe' else '' }}">Transcribe</a>
        <a href="/enhance" class="{{ 'active' if active_page=='enhance' else '' }}">Enhance</a>
        <a href="/translate" class="{{ 'active' if active_page=='translate' else '' }}">Translate</a>
        <a href="/speak" class="{{ 'active' if active_page=='speak' else '' }}">Speech</a>
      </div>
    </div>
  </nav>
  <main>
    <div class="container">{{ content|safe }}</div>
  </main>
</body>
</html>
"""
    return render_template_string(
        template,
        title=title,
        active_page=active_page,
        content=content,
        **kwargs
    )

# ---------- Home (clean + minimal, no code window) ----------
@app.route("/")
def home():
    content = """
<section class="hero">
  <div class="left">
    <span class="kicker"><i class="fas fa-sparkles"></i> New • Classroom-friendly</span>
    <h1 class="title">Turn speech into learning—fast.</h1>
    <p class="subtitle">EduSpeak AI helps teachers and students convert audio into clear notes, improve writing, translate into multiple languages, and listen back with natural speech.</p>
    <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:14px">
      <a href="/transcribe" class="btn btn-primary"><i class="fas fa-microphone"></i> Try Transcription</a>
      <a href="/translate" class="btn btn-outline"><i class="fas fa-language"></i> Translate a sample</a>
    </div>
    <div class="statgrid">
      <div class="stat"><div class="num">20+ languages</div><small>Accurate translation</small></div>
      <div class="stat"><div class="num">Clarity first</div><small>Polished writing</small></div>
      <div class="stat"><div class="num">Accessible</div><small>Slow, natural TTS</small></div>
      <div class="stat"><div class="num">Simple</div><small>1–2 click workflow</small></div>
    </div>
  </div>
  <div class="right">
    <div class="snapshot">
      <h4><i class="fas fa-graduation-cap"></i> Designed for education</h4>
      <ul style="margin:8px 0 0 18px; line-height:1.8">
        <li>Works with phone recordings and lectures</li>
        <li>Improves grammar and tone—keeps meaning</li>
        <li>Easy translation for multilingual classes</li>
        <li>Text-to-speech for accessibility & revision</li>
      </ul>
    </div>
    <div class="snapshot">
      <h4><i class="fas fa-route"></i> Quick flow</h4>
      <ol style="margin:8px 0 0 18px; line-height:1.8">
        <li>Upload audio (mp3/m4a/wav)</li>
        <li>Enhance clarity (optional)</li>
        <li>Translate to target language</li>
        <li>Listen or download as audio</li>
      </ol>
    </div>
  </div>
</section>

<section class="card" style="margin-top:18px">
  <div class="features">
    <div class="feature">
      <div class="iconbox"><i class="fas fa-microphone"></i></div>
      <h3 style="margin:.6rem 0 0 0">Smart Transcription</h3>
      <p>Whisper-based speech-to-text that handles accents and background noise.</p>
    </div>
    <div class="feature">
      <div class="iconbox"><i class="fas fa-wand-magic-sparkles"></i></div>
      <h3 style="margin:.6rem 0 0 0">Language Enhancement</h3>
      <p>Fix grammar, improve clarity, and keep your original meaning.</p>
    </div>
    <div class="feature">
      <div class="iconbox"><i class="fas fa-language"></i></div>
      <h3 style="margin:.6rem 0 0 0">Quick Translation</h3>
      <p>Translate to 20+ languages with one click for multilingual classrooms.</p>
    </div>
    <div class="feature">
      <div class="iconbox"><i class="fas fa-volume-up"></i></div>
      <h3 style="margin:.6rem 0 0 0">Text-to-Speech</h3>
      <p>Natural, slower voice for better comprehension and accessibility.</p>
    </div>
  </div>
</section>

<p class="footer">Built with love for teachers & learners · OpenAI Whisper · Groq LLM · Flask</p>
"""
    return render_page(content, "Home", "home")

# ---------- Transcribe ----------
@app.route("/transcribe", methods=["GET","POST"])
def transcribe():
    transcript = enhanced = error = None
    if request.method == "POST":
        file = request.files.get("audio")
        if not file or file.filename.strip() == "":
            error = "No audio file uploaded. Ensure input name='audio' and multipart/form-data."
        else:
            tmpname = f"{uuid.uuid4()}_{file.filename}"
            try:
                file.save(tmpname)
                with open(tmpname, "rb") as af:
                    whisper_out = openai_client.audio.transcriptions.create(
                        model="whisper-1", file=af, response_format="text", temperature=0
                    )
                transcript = str(whisper_out)
                raw = enhance_agent.run(transcript)
                enhanced = clean_output(getattr(raw, "content", None) or getattr(raw, "text", None) or str(raw))
            except Exception as e:
                error = f"Processing error: {e}"
            finally:
                try:
                    os.remove(tmpname)
                except:
                    pass

    base = """
<div class="card" style="margin-top:6px">
  <h2 style="margin:0 0 8px 0"><i class="fas fa-microphone"></i> Transcribe Audio</h2>
  <form method="POST" enctype="multipart/form-data" id="up">
    <label class="textbox" style="display:block; cursor:pointer;">
      <div style="text-align:center; color:#3730a3"><i class="fas fa-cloud-upload-alt"></i></div>
      <div id="pick" style="text-align:center; margin-top:6px">Click to choose a file (WAV / MP3 / M4A / FLAC)</div>
      <input type="file" name="audio" id="audio" accept=".wav,.mp3,.m4a,.flac" style="display:none"/>
    </label>
    <div style="text-align:center; margin-top:10px">
      <button class="btn btn-primary" type="submit"><i class="fas fa-bolt"></i> Upload & Transcribe</button>
    </div>
  </form>
</div>
<script>
document.getElementById('audio').addEventListener('change', function(e){
  var f = e.target.files[0]; if(!f) return;
  document.getElementById('pick').textContent = 'Ready: ' + f.name + ' (' + (f.size/1024/1024).toFixed(2) + ' MB)';
});
</script>
"""
    content = base
    if transcript or enhanced or error:
        content += '<div class="grid grid-2" style="margin-top:12px">'
        if transcript:
            content += f"""
  <div class="card"><h3 style="margin:0 0 6px 0">Raw Transcript</h3>
  <div class="textbox">{h(transcript)}</div></div>"""
        if enhanced:
            content += f"""
  <div class="card"><h3 style="margin:0 0 6px 0">Enhanced</h3>
  <div class="textbox ok">{h(enhanced)}</div>
  <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:10px">
    <a class="btn btn-outline" href="/translate"><i class="fas fa-language"></i> Translate</a>
    <a class="btn btn-success" href="/speak"><i class="fas fa-volume-up"></i> Speak</a>
  </div></div>"""
        content += "</div>"
    if error:
        content += f'<div class="card" style="margin-top:12px"><div class="err"><i class="fas fa-triangle-exclamation"></i> {h(error)}</div></div>'
    return render_page(content, "Transcribe", "transcribe")


@app.route("/enhance", methods=["GET","POST"])
def enhance():
    original = enhanced_text = error = None
    if request.method == "POST":
        text = (request.form.get("text") or "").strip()
        if not text:
            error = "No text provided."
        else:
            try:
                raw = enhance_agent.run(text)
                enhanced_text = clean_output(getattr(raw, "content", None) or getattr(raw, "text", None) or str(raw))
                original = text
            except Exception as e:
                error = f"Enhancement error: {e}"

    content = """
<div class="card">
  <h2 style="margin:0 0 8px 0"><i class="fas fa-wand-magic-sparkles"></i> Enhance Text</h2>
  <form method="POST">
    <textarea name="text" class="input" placeholder="Paste text to polish...">{}</textarea>
    <div style="text-align:center; margin-top:10px">
      <button class="btn btn-primary"><i class="fas fa-magic"></i> Enhance</button>
    </div>
  </form>
</div>
""".format(h(original or ""))

    if enhanced_text:
        content += """
<div class="grid grid-2" style="margin-top:12px">
  <div class="card"><h3>Original</h3><div class="textbox">{}</div></div>
  <div class="card"><h3>Enhanced</h3><div class="textbox ok">{}</div>
    <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:10px">
      <a href="/translate" class="btn btn-outline"><i class="fas fa-language"></i> Translate</a>
      <a href="/speak" class="btn btn-success"><i class="fas fa-volume-up"></i> Speak</a>
    </div>
  </div>
</div>""".format(h(original), h(enhanced_text))
    if error:
        content += '<div class="card" style="margin-top:12px"><div class="err">{}</div></div>'.format(h(error))
    return render_page(content, "Enhance", "enhance")


@app.route("/translate", methods=["GET","POST"])
def translate_page():
    if request.method == "POST":
        text = (request.form.get("text") or "").strip()
        lang = (request.form.get("lang") or "").strip() or "Spanish"
        if not text:
            return "No text provided", 400
        try:
            raw = translate_agent.run(f"Translate the following English text to {lang}:\n\n{text}")
            translated = clean_output(getattr(raw, "content", None) or getattr(raw, "text", None) or str(raw))
            return """<!DOCTYPE html><html><head><meta charset='utf-8'><title>Translated</title>
<style>body{background:#f6f8ff;color:#0f172a;font-family:'Plus Jakarta Sans',sans-serif;padding:24px} .box{white-space:pre-wrap;background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:14px;box-shadow:0 8px 30px rgba(17,24,39,.08)}</style>
</head><body><h2 style="margin:0 0 10px 0">Translated to %s</h2><div class="box">%s</div><p style="margin-top:12px"><a href="/translate" style="color:#3730a3;text-decoration:none">New translation</a></p></body></html>""" % (h(lang), h(translated))
        except Exception as e:
            return f"Translation error: {h(str(e))}", 500

    content = """
<div class="card">
  <h2 style="margin:0 0 8px 0"><i class="fas fa-language"></i> Translate</h2>
  <form method="POST">
    <div class="grid grid-2">
      <div>
        <label>Target language</label>
        <select name="lang" class="input" required>
          <option value="" disabled selected>Choose</option>
          <option>Spanish</option><option>French</option><option>German</option>
          <option>Hindi</option><option>Urdu</option><option>Bengali</option>
          <option>Punjabi</option><option>Arabic</option><option>Turkish</option>
          <option>Portuguese</option><option>Chinese (Simplified)</option>
          <option>Tamil</option><option>Gujarati</option><option>Polish</option>
          <option>Ukrainian</option><option>Swahili</option>
        </select>
      </div>
      <div style="display:flex; align-items:end; justify-content:flex-end">
        <button class="btn btn-success" type="submit"><i class="fas fa-globe"></i> Translate</button>
      </div>
    </div>
    <label>Text</label>
    <textarea name="text" class="input" placeholder="Enter English text..."></textarea>
  </form>
</div>
"""
    return render_page(content, "Translate", "translate")


@app.route("/speak", methods=["GET","POST"])
def speak_page():
    if request.method == "POST":
        return generate_speech()
    content = """
<div class="card">
  <h2 style="margin:0 0 8px 0"><i class="fas fa-volume-up"></i> Text → Speech</h2>
  <form method="POST">
    <textarea name="text" class="input" placeholder="Enter text…" required></textarea>
    <div style="display:flex; gap:10px; flex-wrap:wrap; justify-content:center; margin-top:10px">
      <button class="btn btn-success" type="submit"><i class="fas fa-download"></i> Download WAV</button>
      <button class="btn btn-outline" type="button" onclick="previewSpeech(event)"><i class="fas fa-play"></i> Preview in Browser</button>
    </div>
  </form>
</div>
<script>
function previewSpeech(ev){
  const area = document.querySelector('textarea[name="text"]');
  const text = (area.value || '').trim();
  if(!text){alert('Please enter text.');return;}
  if(!('speechSynthesis' in window)){alert('Browser TTS not supported.');return;}
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang='en-GB'; u.rate=0.6; u.pitch=1.0;
  const male = speechSynthesis.getVoices().find(v => /male|david|mark|daniel|alex/i.test(v.name));
  if(male) u.voice = male;
  const btn = ev.target; const orig = btn.innerHTML; btn.innerHTML='Speaking...'; btn.disabled=true;
  u.onend=()=>{btn.innerHTML=orig;btn.disabled=false;}; u.onerror=()=>{btn.innerHTML=orig;btn.disabled=false;alert('Speech error.');};
  speechSynthesis.speak(u);
}
</script>
"""
    return render_page(content, "Speak", "speak")

def generate_speech():
    import pyttsx3
    text = (request.form.get("text") or "").strip()
    if not text:
        return "No text provided", 400
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    path = tmp.name
    tmp.close()
    try:
        eng = pyttsx3.init()
        chosen = None
        for v in eng.getProperty("voices"):
            name = (getattr(v, "name", "") or "").lower()
            if any(k in name for k in ["male", "david", "mark", "daniel", "alex", "george", "tom"]):
                chosen = v.id
                break
        if chosen:
            eng.setProperty("voice", chosen)
        base = eng.getProperty("rate") or 200
        eng.setProperty("rate", max(80, int(base * 0.6)))
        eng.setProperty("volume", 1.0)
        eng.save_to_file(text, path)
        eng.runAndWait()

        @after_this_request
        def cleanup(resp):
            try:
                os.remove(path)
            except:
                pass
            return resp

        return send_file(path, mimetype="audio/wav", as_attachment=True, download_name="eduspeak_audio.wav")
    except Exception as e:
        return f"Speech generation error: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)

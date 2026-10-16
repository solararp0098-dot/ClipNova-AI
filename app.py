
import os, tempfile, time
import gradio as gr
import torch
import spaces
from diffusers import WanPipeline
from diffusers.utils import export_to_video

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

pipe = WanPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.to("cuda")

@spaces.GPU(duration=180)
def generate_video(prompt, style, seed, steps):
    prompt = (prompt or "").strip()
    if len(prompt) < 8:
        raise gr.Error("Décris la scène avec un peu plus de détails.")
    if len(prompt) > 700:
        raise gr.Error("700 caractères maximum.")
    suffixes = {
        "Cinématique": "cinematic composition, smooth camera movement, dramatic natural lighting, detailed textures",
        "Réaliste": "photorealistic, natural lighting, realistic motion, detailed textures",
        "Anime": "high-quality anime style, expressive colors, fluid animation",
        "Fantaisie": "fantasy art direction, magical atmosphere, cinematic lighting",
        "Nature": "nature documentary, realistic wildlife and landscapes, natural light",
        "Aucun": ""
    }
    suffix = suffixes.get(style, "")
    full_prompt = f"{prompt}. {suffix}" if suffix else prompt
    chosen_seed = int(seed) if seed is not None and int(seed) >= 0 else int(time.time()) % 2000000000
    generator = torch.Generator(device="cuda").manual_seed(chosen_seed)
    result = pipe(prompt=full_prompt, height=480, width=832, num_frames=17,
                  num_inference_steps=int(steps), guidance_scale=5.0, generator=generator)
    frames = result.frames[0]
    path = os.path.join(tempfile.gettempdir(), f"clipnova_{chosen_seed}.mp4")
    export_to_video(frames, path, fps=8)
    return path, f"Vidéo créée avec Wan 2.1 · seed {chosen_seed}"

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto !important; }
body, .gradio-container { background: #080a12 !important; color: #f4f5ff !important; }
#hero { border: 1px solid #2a2e49; border-radius: 24px; padding: 28px; background: radial-gradient(ellipse at top left, #29245b 0%, #12162b 42%, #0c0e18 100%); margin-bottom: 18px; }
#brand { font-size: 38px; font-weight: 850; letter-spacing: -1.5px; margin: 0; }
#brand span { color: #a78bfa; }
#subtitle { color: #c6c8df; font-size: 15px; margin-top: 8px; }
#tag { display: inline-block; background: #202445; color: #c4b5fd; border: 1px solid #484277; padding: 6px 11px; border-radius: 99px; font-size: 12px; margin-bottom: 12px; }
.panel { border: 1px solid #292d44 !important; border-radius: 18px !important; background: #101321 !important; }
button.primary { background: linear-gradient(90deg, #7c3aed, #5b5bd6) !important; border: none !important; }
"""

with gr.Blocks(theme=gr.themes.Base(primary_hue="violet", secondary_hue="indigo", neutral_hue="slate"), css=CSS, title="ClipNova AI") as demo:
    gr.HTML("<div id='hero'><div id='tag'>✦ TEXT-TO-VIDEO · OPEN MODEL</div><h1 id='brand'>Clip<span>Nova</span> AI</h1><div id='subtitle'>Transforme une idée en courte vidéo : décris la scène, choisis une ambiance, puis lance la génération.</div></div>")
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["panel"]):
            prompt = gr.Textbox(label="Décris ta vidéo", placeholder="Ex. : Un petit robot traverse une ville futuriste sous la pluie, reflets néon, travelling cinématique...", lines=5, max_lines=8)
            style = gr.Dropdown(choices=["Cinématique", "Réaliste", "Anime", "Fantaisie", "Nature", "Aucun"], value="Cinématique", label="Style visuel")
            with gr.Row():
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = aléatoire)")
                steps = gr.Dropdown(choices=[12, 16, 20], value=12, label="Qualité / étapes")
            generate = gr.Button("✦ Générer ma vidéo", variant="primary", size="lg")
            status = gr.Markdown("**Modèle :** Wan 2.1 T2V 1.3B · Quotas gratuits limités.")
        with gr.Column(scale=5, elem_classes=["panel"]):
            video = gr.Video(label="Aperçu généré", format="mp4", autoplay=False)
            gr.Markdown("La génération peut prendre plusieurs minutes. Disponibilité selon le GPU partagé et les quotas gratuits.")
    generate.click(generate_video, inputs=[prompt, style, seed, steps], outputs=[video, status], show_progress="full")
    gr.Markdown("---\n**À savoir :** prototype communautaire non affilié à Remakeit. Les résultats IA peuvent contenir des erreurs. Utilise seulement des contenus que tu as le droit d'utiliser.")

if __name__ == "__main__":
    demo.queue(max_size=2).launch()

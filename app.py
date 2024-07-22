from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# Define language codes supported by the model
lang_code_to_id = {
    "English": "eng_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Russian": "rus_Cyrl"
}

def translate(text, src_lang, tgt_lang):
    tgt_lang_code = lang_code_to_id[tgt_lang]
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Generate translation
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang_code)
    )
    
    # Decode the tokens to get the translated text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Define the Gradio interface
interface = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter text to translate"),
        gr.Dropdown(choices=["English", "French", "German", "Spanish","Russian"], label="Source Language"),
        gr.Dropdown(choices=["English", "French", "German", "Spanish","Russian"], label="Target Language")
    ],
    outputs="text",
    title="Multilingual Translator",
    description="Translate text between various languages using the facebook/nllb-200-distilled-600M model."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()

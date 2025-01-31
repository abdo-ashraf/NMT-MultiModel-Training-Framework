import torch
from Models.ModelArgs import ModelArgs
from Models.AutoModel import get_model
from gradio_utils import Callable_tokenizer, greedy_decode
import gradio as gr

def en_translate_ar(text, model, tokenizer, max_tries=50):
    source_tensor = torch.tensor(tokenizer(text)).unsqueeze(0)
    target_tokens = greedy_decode(model, source_tensor,
                                  tokenizer.get_tokenId('<s>'),
                                  tokenizer.get_tokenId('</s>'),
                                  tokenizer.get_tokenId('<pad>'), max_tries)
    
    return tokenizer.decode(target_tokens)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = Callable_tokenizer('./assets/tokenizers/en-ar_tokenizer.model')

model_state_dict = torch.load("./assets/models/en-ar_s2sAttention.pth", map_location=device, weights_only=True)['model_state_dict']
model_args = ModelArgs('s2sattention', "./Configurations/s2sattention_model_config.json")
s2sattention = get_model(model_args, len(tokenizer))
s2sattention.load_state_dict(model_state_dict)
s2sattention.to(device)

model_state_dict = torch.load("./assets/models/en-ar_s2s.pth", map_location=device, weights_only=True)['model_state_dict']
model_args = ModelArgs('s2s', "./Configurations/s2s_model_config.json")
s2s = get_model(model_args, len(tokenizer))
s2s.load_state_dict(model_state_dict)
s2s.to(device)

model_state_dict = torch.load("./assets/models/en-ar_transformer.pth", map_location=device, weights_only=True)['model_state_dict']
model_args = ModelArgs('transformer', "./Configurations/transformer_model_config.json")
transformer = get_model(model_args, len(tokenizer))
transformer.load_state_dict(model_state_dict)
transformer.to(device)


def lunch(raw_input, maxtries=50):
    transformer_out = en_translate_ar(raw_input, transformer, tokenizer, maxtries)
    s2s_out = en_translate_ar(raw_input, s2s, tokenizer, maxtries)
    s2sattention_out = en_translate_ar(raw_input, s2sattention, tokenizer, maxtries)
    return transformer_out, s2s_out, s2sattention_out


custom_css ='.gr-button {background-color: #bf4b04; color: white;}'
with gr.Blocks(css=custom_css) as demo:
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label='English Sentence')
            gr.Examples(['How are you?',
                         'She is a good girl.',
                         'Who is better than me?!'],
                        inputs=input_text, label="Examples: ")
        with gr.Column():
            output1 = gr.Textbox(label="Arabic Transformer Translation")
            output2 = gr.Textbox(label="Arabic seq2seq with Attention Translation")
            output3 = gr.Textbox(label="Arabic seq2seq No Attention Translation")
            
            start_btn = gr.Button(value='Arabic Translation', elem_classes=["gr-button"])
    start_btn.click(fn=lunch, inputs=input_text, outputs=[output1, output2, output3])

demo.launch()

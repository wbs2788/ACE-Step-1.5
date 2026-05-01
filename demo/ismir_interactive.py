#!/usr/bin/env python3
"""
ACE-Step Real-time Interactive Music Generation System (ISMIR Demo)
Features:
- Dear PyGui UI for adjusting parameters in real-time
- Streaming audio playback via sounddevice
- Seamless chunked inference with prompt interpolation and EMA smoothing
- Optional MIDI CC mapping via mido
"""

import argparse
import os
import queue
import threading
import time
from typing import Optional

import dearpygui.dearpygui as dpg
import numpy as np
import sounddevice as sd
import torch

try:
    import mido
    HAVE_MIDO = True
except ImportError:
    HAVE_MIDO = False

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler

# --- State Management ---

class ControlState:
    def __init__(self, initial_blend=0.5, initial_temp=1.0, alpha=0.2):
        self.lock = threading.Lock()
        
        # Target values set by UI/MIDI
        self.target_blend = initial_blend
        self.target_temp = initial_temp
        
        # Smoothed values for inference
        self.smoothed_blend = initial_blend
        self.smoothed_temp = initial_temp
        
        # Smoothing factor
        self.alpha = alpha

    def update_targets(self, blend: Optional[float] = None, temp: Optional[float] = None):
        with self.lock:
            if blend is not None:
                self.target_blend = max(0.0, min(1.0, blend))
            if temp is not None:
                self.target_temp = max(0.5, min(1.5, temp))

    def get_smoothed_params(self) -> tuple[float, float]:
        with self.lock:
            # Apply Exponential Moving Average
            self.smoothed_blend = self.alpha * self.target_blend + (1 - self.alpha) * self.smoothed_blend
            self.smoothed_temp = self.alpha * self.target_temp + (1 - self.alpha) * self.smoothed_temp
            return self.smoothed_blend, self.smoothed_temp


# --- Helper Functions ---

def _build_text_payload_batch(
    dit_handler: AceStepHandler,
    captions: list[str],
    lyrics: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build text and lyric embeddings for a batch to ensure identical padding."""
    batch_size = len(captions)
    (
        _text_inputs,
        text_token_ids,
        text_attention_masks,
        lyric_token_ids,
        lyric_attention_masks,
        _non_cover_text_input_ids,
        _non_cover_text_attention_masks,
    ) = dit_handler._prepare_text_conditioning_inputs(
        batch_size=batch_size,
        instructions=["Fill the audio semantic mask based on the given conditions:"] * batch_size,
        captions=captions,
        lyrics=lyrics,
        parsed_metas=[""] * batch_size,
        vocal_languages=["unknown"] * batch_size,
        audio_cover_strength=1.0,
    )

    text_token_ids = text_token_ids.to(dit_handler.device)
    text_attention_masks = text_attention_masks.to(dit_handler.device)
    lyric_token_ids = lyric_token_ids.to(dit_handler.device)
    lyric_attention_masks = lyric_attention_masks.to(dit_handler.device)

    with dit_handler._load_model_context("text_encoder"):
        text_hidden_states = dit_handler.infer_text_embeddings(text_token_ids).to(dit_handler.dtype)
        lyric_hidden_states = dit_handler.infer_lyric_embeddings(lyric_token_ids).to(dit_handler.dtype)

    return (
        text_hidden_states,
        text_attention_masks.bool(),
        lyric_hidden_states,
        lyric_attention_masks.bool(),
    )


def _prepare_condition_tensors_batch(
    dit_handler: AceStepHandler,
    text_hidden_states: torch.Tensor,
    text_attention_mask: torch.Tensor,
    lyric_hidden_states: torch.Tensor,
    lyric_attention_mask: torch.Tensor,
    total_frames: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare encoder/context latents."""
    dit_handler._ensure_silence_latent_on_device()
    silence_slice = dit_handler._get_silence_latent_slice(total_frames).unsqueeze(0).expand(batch_size, -1, -1).to(
        device=dit_handler.device, dtype=dit_handler.dtype
    )
    full_mask = torch.ones(batch_size, total_frames, device=dit_handler.device, dtype=dit_handler.dtype)
    chunk_masks = torch.ones(batch_size, total_frames, 64, device=dit_handler.device, dtype=torch.bool)
    is_covers = torch.zeros(batch_size, device=dit_handler.device, dtype=torch.bool)

    refer_audio_latents, refer_audio_order_mask = dit_handler.infer_refer_latent(
        [[torch.zeros(2, 96000, device=dit_handler.device, dtype=dit_handler.dtype)]] * batch_size
    )
    refer_audio_latents = refer_audio_latents.to(device=dit_handler.device, dtype=dit_handler.dtype)

    with torch.inference_mode():
        encoder_hidden_states, encoder_attention_mask, context_latents = dit_handler.model.prepare_condition(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_latents,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=silence_slice,
            attention_mask=full_mask,
            silence_latent=dit_handler.silence_latent,
            src_latents=silence_slice,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            precomputed_lm_hints_25Hz=None,
        )
    return encoder_hidden_states, encoder_attention_mask, context_latents


def truncate_kv_cache(pkv, max_frames: int):
    """Sliding window truncation for KV cache to prevent infinite memory growth."""
    if pkv is None:
        return None
    
    self_cache = pkv.self_attention_cache
    if not hasattr(self_cache, 'key_cache'):
        return pkv
        
    for i in range(len(self_cache.key_cache)):
        # Shape is (batch, num_heads, seq_len, head_dim)
        if self_cache.key_cache[i].size(2) > max_frames:
            self_cache.key_cache[i] = self_cache.key_cache[i][:, :, -max_frames:, :]
            self_cache.value_cache[i] = self_cache.value_cache[i][:, :, -max_frames:, :]
            
    if hasattr(self_cache, '_seen_tokens'):
        self_cache._seen_tokens = min(self_cache._seen_tokens, max_frames)
        
    return pkv


# --- Audio & Inference Engine ---

def inference_worker(
    state: ControlState,
    audio_queue: queue.Queue,
    args,
):
    print("[Inference] Initializing models...")
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(os.path.abspath(args.checkpoint_dir))
    
    dit_handler = AceStepHandler()
    _llm_handler = LLMHandler()

    dit_handler.initialize_service(
        project_root=".",
        config_path=args.config_path,
        device=args.device,
        use_flash_attention=None,
        compile_model=False,
        offload_to_cpu=args.offload_to_cpu,
        offload_dit_to_cpu=False,
    )
    if args.lora_path:
        dit_handler.load_lora(args.lora_path)
        dit_handler.set_lora_scale(1.0)
        dit_handler.set_use_lora(True)

    print("[Inference] Extracting prompt embeddings...")
    # Get max possible frames for condition tensors to avoid OOM or out of bounds.
    # Since it's streaming, we can generate a large dummy context and reuse it.
    MAX_CONTEXT_FRAMES = 5000  # 100 seconds
    
    text_hs, text_mask, lyric_hs, lyric_mask = _build_text_payload_batch(
        dit_handler, [args.prompt_a, args.prompt_b], [args.lyrics, args.lyrics]
    )
    enc_hs, enc_mask, ctx = _prepare_condition_tensors_batch(
        dit_handler, text_hs, text_mask, lyric_hs, lyric_mask, MAX_CONTEXT_FRAMES, batch_size=2
    )
    
    enc_hs_A, enc_hs_B = enc_hs[0:1], enc_hs[1:2]
    ctx_A, ctx_B = ctx[0:1], ctx[1:2]
    enc_mask_A = enc_mask[0:1]

    fps = 50
    # ~0.96s per chunk = 48 frames, or ~1.0s = 50 frames.
    chunk_frames = 50
    MAX_KV_FRAMES = fps * 10  # 10 seconds history
    
    pkv = None
    decoder = dit_handler.model.decoder
    dtype = dit_handler.dtype
    device = dit_handler.device

    print("[Inference] Starting generation loop...")
    chunk_idx = 0
    
    while True:
        # Get smoothed controls
        blend, temp = state.get_smoothed_params()
        
        # Interpolate embeddings
        enc_hs_curr = (1 - blend) * enc_hs_A + blend * enc_hs_B
        ctx_curr = (1 - blend) * ctx_A + blend * ctx_B
        
        start_frame = chunk_idx * chunk_frames
        end_frame = start_frame + chunk_frames
        
        # Wrap context if we exceed MAX_CONTEXT_FRAMES
        if end_frame > MAX_CONTEXT_FRAMES:
            chunk_idx = 0
            start_frame = 0
            end_frame = chunk_frames
            # NOTE: pkv should probably be reset here if we wrap context, 
            # or we generate a massive context. 5000 frames is 100s.
            pkv = None
            
        ctx_slice = ctx_curr[:, start_frame:end_frame, :]
        
        # Temperature applies to initial noise scale
        xt = temp * torch.randn(1, chunk_frames, 64, device=device, dtype=dtype)
        t = torch.ones(1, device=device, dtype=dtype)
        t_expand = t.view(-1, 1, 1)
        attention_mask = torch.ones(1, chunk_frames, device=device, dtype=dtype)
        
        with torch.inference_mode():
            outputs = decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=t,
                attention_mask=attention_mask,
                encoder_hidden_states=enc_hs_curr,
                encoder_attention_mask=enc_mask_A,  # Assuming masks are identical
                context_latents=ctx_slice,
                use_cache=True,
                past_key_values=pkv,
            )
            v_pred, pkv = outputs[0], outputs[1]
            pred_x0 = xt - t_expand * v_pred
            
        # Truncate KV Cache
        pkv = truncate_kv_cache(pkv, MAX_KV_FRAMES)
            
        # Decode Latents
        # We need to decode this chunk into audio
        # Using dit_handler's wrapper for safety, though direct VAE call is also possible
        pred_wavs, _, _ = dit_handler._decode_generate_music_pred_latents(
            pred_latents=pred_x0,
            progress=None,
            use_tiled_decode=args.use_tiled_decode,
            time_costs={"total_time_cost": 0.0, "diffusion_per_step_time_cost": 0.0},
        )
        
        wav_chunk = pred_wavs[0].detach().cpu().numpy().T  # (samples, channels)
        
        # Push to audio queue (blocks if full)
        audio_queue.put(wav_chunk)
        chunk_idx += 1


# --- Audio Output ---

def audio_playback_loop(audio_queue: queue.Queue, sample_rate: int = 48000, channels: int = 2):
    """Pulls chunks from queue and plays them using sounddevice."""
    
    def callback(outdata, frames, time, status):
        if status:
            print(f"[Audio] Status: {status}")
        try:
            # We assume queue contains numpy arrays of shape (N, 2)
            # The callback might request `frames` which is not exactly the chunk size.
            # However, for simplicity in this demo, we can just write the chunk.
            # A more robust implementation requires a ring buffer.
            pass
        except queue.Empty:
            outdata.fill(0)
            
    # For a simple demo without complex ring buffering, we can just use `sd.OutputStream` 
    # and write to it directly in a thread instead of a callback.
    try:
        stream = sd.OutputStream(samplerate=sample_rate, channels=channels, dtype='float32')
        stream.start()
        print("[Audio] Playback started.")
        
        while True:
            chunk = audio_queue.get()
            stream.write(chunk)
            
    except sd.PortAudioError as e:
        print(f"[Audio] No audio device available ({e}). Falling back to dummy playback.")
        while True:
            # Just consume the queue so generation doesn't block
            chunk = audio_queue.get()
            # simulate playback time (approximate)
            time.sleep(chunk.shape[0] / sample_rate)
            
    except KeyboardInterrupt:
        if 'stream' in locals():
            stream.stop()
            stream.close()


# --- MIDI Worker ---

def midi_worker(state: ControlState):
    if not HAVE_MIDO:
        print("[MIDI] Mido not installed. MIDI control disabled.")
        return
        
    try:
        inputs = mido.get_input_names()
        if not inputs:
            print("[MIDI] No MIDI inputs found.")
            return
            
        inport_name = inputs[0]
        print(f"[MIDI] Listening on {inport_name}")
        
        with mido.open_input(inport_name) as inport:
            for msg in inport:
                if msg.type == 'control_change':
                    # Example mapping: CC 1 (Mod Wheel) -> Blend, CC 2 -> Temp
                    val = msg.value / 127.0
                    if msg.control == 1:
                        state.update_targets(blend=val)
                        # Sync GUI
                        dpg.set_value("blend_slider", val)
                    elif msg.control == 2:
                        temp_val = 0.5 + val  # map [0, 1] to [0.5, 1.5]
                        state.update_targets(temp=temp_val)
                        dpg.set_value("temp_slider", temp_val)
    except Exception as e:
        print(f"[MIDI] Error: {e}")


# --- Main & UI ---

def build_parser():
    parser = argparse.ArgumentParser(description="ACE-Step Real-time Interactive Demo")
    parser.add_argument("--prompt-a", type=str, required=True, help="First prompt style")
    parser.add_argument("--prompt-b", type=str, required=True, help="Second prompt style")
    parser.add_argument("--lyrics", type=str, default="[Instrumental]", help="Lyrics")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Base checkpoint root")
    parser.add_argument("--config-path", default="acestep-v15-xl-turbo", help="Model name")
    parser.add_argument("--lora-path", default="", help="LoRA adapter directory")
    parser.add_argument("--device", default="cuda", help="Device string")
    parser.add_argument("--use-tiled-decode", action="store_true", help="Use tiled VAE decode")
    parser.add_argument("--offload-to-cpu", action="store_true", help="Enable CPU offload")
    parser.add_argument("--headless", action="store_true", help="Run without GUI (for server testing)")
    return parser

def main():
    args = build_parser().parse_args()
    
    state = ControlState(initial_blend=0.5, initial_temp=1.0)
    audio_queue = queue.Queue(maxsize=3)
    
    # Start Inference Thread
    inf_thread = threading.Thread(target=inference_worker, args=(state, audio_queue, args), daemon=True)
    inf_thread.start()
    
    # Start Audio Thread
    audio_thread = threading.Thread(target=audio_playback_loop, args=(audio_queue, 48000, 2), daemon=True)
    audio_thread.start()
    
    # Start MIDI Thread
    midi_thread = threading.Thread(target=midi_worker, args=(state,), daemon=True)
    midi_thread.start()
    
    if args.headless:
        print("[System] Running in HEADLESS mode. GUI disabled.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[System] Exiting...")
            return
    
    # Start UI
    dpg.create_context()
    
    def blend_callback(sender, app_data):
        state.update_targets(blend=app_data)
        
    def temp_callback(sender, app_data):
        state.update_targets(temp=app_data)
        
    with dpg.window(label="ACE-Step Real-time Control", width=600, height=300):
        dpg.add_text("Prompt Blend")
        dpg.add_text(f"A: {args.prompt_a[:30]}...", color=[150, 150, 150])
        dpg.add_slider_float(tag="blend_slider", default_value=0.5, min_value=0.0, max_value=1.0, callback=blend_callback, width=500)
        dpg.add_text(f"B: {args.prompt_b[:30]}...", color=[150, 150, 150])
        
        dpg.add_separator()
        
        dpg.add_text("Temperature (Order <-> Chaos)")
        dpg.add_slider_float(tag="temp_slider", default_value=1.0, min_value=0.5, max_value=1.5, callback=temp_callback, width=500)
        
        dpg.add_separator()
        dpg.add_text("Note: Audio starts playing after initial warmup (~5-10s).")
        
    dpg.create_viewport(title='ACE-Step ISMIR Demo', width=650, height=350)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()

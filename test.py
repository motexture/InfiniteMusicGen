import torch
import sounddevice as sd
import numpy as np
from multiprocessing import Process, Queue
from time import sleep
from queue import Empty

# Assuming MusicGen and other dependencies are properly installed and imported
from audiocraft.models import MusicGen

def audio_generation_process(queue):
    device = 'cuda'
    sample_rate = 32000  # Define the sample rate
    buffer_duration = 30  # Duration of each audio buffer in seconds
    conditioning_seconds = 5

    # Initialize the MusicGen model
    model = MusicGen.get_pretrained('melody', device=device)
    model.set_generation_params(use_sampling=True, top_k=250, duration=buffer_duration)

    # Generate initial music chunk
    description = ['Tech House, Boiler Room Berlin, Deep, Medium, F Minor, 124 bpm, 44100 Hz']
    music = model.generate(descriptions=description, progress=True)
    queue.put(music.cpu().numpy())

    # Continuously generate and queue music chunks
    while True:
        try:
            music = model.generate_continuation(music[:, :, -sample_rate * conditioning_seconds:], sample_rate, description, progress=True)
            queue.put(music.cpu().numpy())
        except Exception as e:
            print(f"Error on {device}: {e}")
        sleep(0.1)  # Sleep briefly to yield control

# Initialize a buffer to hold incoming data
audio_buffer = np.array([], dtype=np.float32)

def audio_playback_process(queue):
    global audio_buffer
    crossfade_duration = 1000  # Crossfade duration in samples (adjust based on sample rate and desired effect)
    fade_out = np.linspace(1, 0, crossfade_duration)
    fade_in = np.linspace(0, 1, crossfade_duration)

    def callback(outdata, frames, time, status):
        global audio_buffer
        if status.output_underflow:
            print('Output underflow: Increase block_size or reduce sleep time')
            raise sd.CallbackAbort
        assert not status
        
        # Ensure the buffer has enough data, including crossfade buffer
        required_data = frames + crossfade_duration
        if len(audio_buffer) < required_data:
            try:
                data = queue.get_nowait()
                if data.ndim > 2:
                    data = np.squeeze(data)
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                    data /= 32768.0  # Normalize from int16 range
                if len(audio_buffer) >= crossfade_duration:
                    # Apply crossfade
                    audio_buffer[-crossfade_duration:] = (audio_buffer[-crossfade_duration:] * fade_out) + (data[:crossfade_duration] * fade_in)
                    audio_buffer = np.append(audio_buffer, data[crossfade_duration:])
                else:
                    audio_buffer = np.append(audio_buffer, data)
            except Empty:
                print('Buffer is empty: Increase queue size')
                outdata.fill(0)
                return

        # Fill the output buffer and update the main buffer
        outdata[:, 0] = audio_buffer[:frames]
        audio_buffer = audio_buffer[frames:]

    sample_rate = 32000  # This should match the generation sample rate
    channels = 1  # Mono audio

    # Open the audio stream
    with sd.OutputStream(channels=channels, callback=callback, samplerate=sample_rate):
        while True:
            sleep(0.01)  # Sleep to allow other processes/threads to run

if __name__ == '__main__':
    # Set the start method for multiprocessing
    from multiprocessing import set_start_method, Queue
    set_start_method('spawn')

    audio_queue = Queue(maxsize=10)  # Define a maximum size for the queue to prevent excessive memory usage
    processes = []

    # Start the audio generation process
    gen_process = Process(target=audio_generation_process, args=(audio_queue,))
    gen_process.start()
    processes.append(gen_process)

    # Start the audio playback process
    playback_process = Process(target=audio_playback_process, args=(audio_queue,))
    playback_process.start()
    processes.append(playback_process)

    # Wait for all processes to complete
    for process in processes:
        process.join()

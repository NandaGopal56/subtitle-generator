import whisper

# Load the Whisper Tiny model
model = whisper.load_model("tiny")


def format_timestamp(seconds):
    """Format seconds into SRT timestamp format with consistent zero-padding."""
    # Convert seconds to hours, minutes, seconds, and milliseconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    
    # Format as HH:MM:SS,mmm with zero-padding
    return f"{int(hours):02}:{int(minutes):02}:{seconds:02},{milliseconds:03}"

def generate_srt(transcription):
    srt_content = ""
    for i, segment in enumerate(transcription['segments'], start=1):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text'].strip()
        
        srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
    
    return srt_content

def format_transcription_result(result):
    """Extracts and formats segments from the Whisper result into a transcription dictionary."""
    transcription_result = {
        'segments': [
            {
                'id': segment['id'],
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            }
            for segment in result.get('segments', [])
        ]
    }
    return transcription_result


def transcribe_audio(model, filename):

    # Load the audio file and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(filename)
    # audio = whisper.pad_or_trim(audio)

    # Make a prediction
    result = model.transcribe(audio, word_timestamps=True)

    return result

def format_whisper_result_by_word_timestamps(result, words_per_segment=2):
    # Initialize the formatted result dictionary
    formatted_result = {
        'text': result.get('text', ''),
        'segments': [],
        'language': result.get('language', 'en')
    }

    # Process each segment in the original Whisper result
    for segment in result.get("segments", []):
        words = segment.get("words", [])
        
        # Loop through words in chunks of `words_per_segment`
        for i in range(0, len(words), words_per_segment):
            # Collect words and timestamps for each sub-segment
            chunk_words = words[i:i + words_per_segment]
            chunk_text = ' '.join(word['word'] for word in chunk_words)
            start_time = chunk_words[0]['start']
            end_time = chunk_words[-1]['end']

            # Add each new sub-segment to the formatted segments list
            formatted_result['segments'].append({
                'id': len(formatted_result['segments']),
                'seek': segment.get('seek', 0),
                'start': start_time,
                'end': end_time,
                'text': chunk_text
            })

    return formatted_result



whisper_result = transcribe_audio(model=model, filename="Audio 1 (enhanced).wav")

formatted_whisper_result_by_word = format_whisper_result_by_word_timestamps(whisper_result)

# Format the transcription result
transcription_result = format_transcription_result(formatted_whisper_result_by_word)

# Generate the SRT content
srt_content = generate_srt(transcription_result)


# Save it to an SRT file
with open("transcription.srt", "w") as file:
    file.write(srt_content)
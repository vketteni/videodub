import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import tempfile
import subprocess

@dataclass
class TranslationSegment:
    start_time: float
    end_time: float
    original_text: str
    translated_text: str
    translated_audio_path: Optional[str] = None

class AudioTranslator:
    def __init__(self, target_language: str = "es", tts_engine: str = "openai"):
        self.target_language = target_language
        self.tts_engine = tts_engine
        self.logger = logging.getLogger(__name__)
        
    def translate_transcript(self, transcript: List[Dict], target_language: str = None) -> List[TranslationSegment]:
        """Translate transcript text to target language"""
        if target_language:
            self.target_language = target_language
            
        translated_segments = []
        
        try:
            for segment in transcript:
                # For demo purposes, using a placeholder translation
                # In production, you'd use Google Translate API, OpenAI API, etc.
                translated_text = self._translate_text(segment["text"])
                
                translated_segments.append(TranslationSegment(
                    start_time=segment["start"],
                    end_time=segment.get("end", segment["start"] + 3),
                    original_text=segment["text"],
                    translated_text=translated_text
                ))
                
        except Exception as e:
            self.logger.error(f"Error translating transcript: {str(e)}")
            
        return translated_segments
    
    def _translate_text(self, text: str) -> str:
        """Translate text using OpenAI API"""
        try:
            from openai import OpenAI
            import os
            
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.error("OPENAI_API_KEY not found - using placeholder translation")
                return self._fallback_translation(text)
            
            client = OpenAI(api_key=api_key)
            
            # Language mapping
            language_names = {
                "de": "German",
                "es": "Spanish", 
                "fr": "French",
                "it": "Italian",
                "pt": "Portuguese",
                "ja": "Japanese",
                "ko": "Korean",
                "zh": "Chinese",
                "ru": "Russian",
                "ar": "Arabic",
                "hi": "Hindi"
            }
            
            target_lang_name = language_names.get(self.target_language, self.target_language)
            
            # Create translation request
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a professional translator. Translate the following text to {target_lang_name}. Only return the translated text, nothing else."},
                    {"role": "user", "content": text}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            translated = response.choices[0].message.content.strip()
            return translated
            
        except ImportError:
            self.logger.error("OpenAI package not installed. Using fallback translation.")
            return self._fallback_translation(text)
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}. Using fallback.")
            return self._fallback_translation(text)
    
    def _fallback_translation(self, text: str) -> str:
        """Fallback translation with language prefix"""
        language_prefixes = {
            "es": "[ES] ",
            "fr": "[FR] ",
            "de": "[DE] ",
            "it": "[IT] ",
            "pt": "[PT] ",
            "ja": "[JA] ",
            "ko": "[KO] ",
            "zh": "[ZH] "
        }
        
        prefix = language_prefixes.get(self.target_language, f"[{self.target_language.upper()}] ")
        return f"{prefix}{text}"
    
    def generate_translated_audio(self, segments: List[TranslationSegment], output_dir: str) -> str:
        """Generate audio for translated segments"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        audio_segments = []
        
        for i, segment in enumerate(segments):
            try:
                audio_file = self._text_to_speech(
                    segment.translated_text, 
                    f"segment_{i:04d}.wav",
                    str(output_path)
                )
                segment.translated_audio_path = audio_file
                audio_segments.append(audio_file)
                
            except Exception as e:
                self.logger.error(f"Error generating audio for segment {i}: {str(e)}")
        
        # Combine all audio segments
        final_audio_path = str(output_path / "translated_audio.wav")
        self._combine_audio_segments(audio_segments, final_audio_path)
        
        return final_audio_path
    
    def _text_to_speech(self, text: str, filename: str, output_dir: str) -> str:
        """Convert text to speech using available TTS engine"""
        output_path = Path(output_dir) / filename
        
        if self.tts_engine == "openai":
            return self._openai_tts(text, str(output_path))
        elif self.tts_engine == "google":
            return self._google_tts(text, str(output_path))
        elif self.tts_engine == "azure":
            return self._azure_tts(text, str(output_path))
        else:
            # Fallback to system TTS (Linux/macOS)
            return self._system_tts(text, str(output_path))
    
    def _openai_tts(self, text: str, output_path: str) -> str:
        """OpenAI TTS implementation"""
        try:
            from openai import OpenAI
            import os
            
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.error("OPENAI_API_KEY not found in environment")
                return self._system_tts(text, output_path)
            
            client = OpenAI(api_key=api_key)
            
            # Create TTS request
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",  # You can change to: echo, fable, onyx, nova, shimmer
                input=text,
                response_format="wav"
            )
            
            # Save to file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return output_path
            
        except ImportError:
            self.logger.error("OpenAI package not installed. Install with: pip install openai")
            return self._system_tts(text, output_path)
        except Exception as e:
            self.logger.error(f"OpenAI TTS error: {str(e)}")
            return self._system_tts(text, output_path)
    
    def _google_tts(self, text: str, output_path: str) -> str:
        """Google TTS implementation"""
        try:
            # Placeholder for Google TTS API
            # from google.cloud import texttospeech
            # client = texttospeech.TextToSpeechClient()
            
            with open(output_path, 'w') as f:
                f.write(f"# Google TTS placeholder for: {text[:50]}...")
                
            return output_path
            
        except Exception as e:
            self.logger.error(f"Google TTS error: {str(e)}")
            return self._system_tts(text, output_path)
    
    def _azure_tts(self, text: str, output_path: str) -> str:
        """Azure TTS implementation"""
        try:
            # Placeholder for Azure TTS
            with open(output_path, 'w') as f:
                f.write(f"# Azure TTS placeholder for: {text[:50]}...")
                
            return output_path
            
        except Exception as e:
            self.logger.error(f"Azure TTS error: {str(e)}")
            return self._system_tts(text, output_path)
    
    def _system_tts(self, text: str, output_path: str) -> str:
        """Fallback system TTS"""
        try:
            # Try using system espeak or say command
            if os.system("which espeak > /dev/null 2>&1") == 0:
                cmd = f'espeak -s 150 -w "{output_path}" "{text}"'
                os.system(cmd)
            elif os.system("which say > /dev/null 2>&1") == 0:
                # macOS
                cmd = f'say "{text}" -o "{output_path.replace(".wav", ".aiff")}"'
                os.system(cmd)
            else:
                # Create placeholder
                with open(output_path, 'w') as f:
                    f.write(f"# System TTS placeholder for: {text[:50]}...")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"System TTS error: {str(e)}")
            return output_path
    
    def _combine_audio_segments(self, audio_files: List[str], output_path: str):
        """Combine multiple audio files into single file"""
        try:
            # Using ffmpeg if available
            if os.system("which ffmpeg > /dev/null 2>&1") == 0:
                file_list = "|".join(audio_files)
                cmd = f'ffmpeg -i "concat:{file_list}" -c copy "{output_path}"'
                os.system(cmd)
            else:
                # Fallback: just copy first file
                if audio_files:
                    import shutil
                    shutil.copy2(audio_files[0], output_path)
                    
        except Exception as e:
            self.logger.error(f"Error combining audio segments: {str(e)}")
    
    def save_translation_data(self, segments: List[TranslationSegment], output_path: str):
        """Save translation data to JSON file"""
        data = {
            "target_language": self.target_language,
            "segments": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "original_text": seg.original_text,
                    "translated_text": seg.translated_text,
                    "audio_path": seg.translated_audio_path
                }
                for seg in segments
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Translation data saved to: {output_path}")
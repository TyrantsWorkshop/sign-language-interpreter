"""
Text-to-Speech Integration - ElevenLabs and Google TTS
"""

import aiohttp
import logging
from typing import Optional
import io

logger = logging.getLogger(__name__)


class TTSProcessor:
    """Convert text to speech"""
    
    def __init__(self, provider: str = "elevenlabs", api_key: str = "", 
                 voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        self.provider = provider
        self.api_key = api_key
        self.voice_id = voice_id
    
    async def synthesize_elevenlabs(self, text: str) -> Optional[bytes]:
        """
        Synthesize speech using ElevenLabs API
        
        Args:
            text: Text to convert to speech
        
        Returns:
            Audio bytes (MP3)
        """
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
            
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "text": text,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        logger.error(f"ElevenLabs API error: {response.status}")
                        return None
        
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            return None
    
    async def synthesize_google(self, text: str) -> Optional[bytes]:
        """
        Synthesize speech using Google Cloud Text-to-Speech
        
        Args:
            text: Text to convert to speech
        
        Returns:
            Audio bytes (MP3)
        """
        try:
            from google.cloud import texttospeech
            
            client = texttospeech.TextToSpeechClient()
            
            input_text = texttospeech.SynthesisInput(text=text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Neural2-C"
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            
            response = client.synthesize_speech(
                input=input_text,
                voice=voice,
                audio_config=audio_config
            )
            
            return response.audio_content
        
        except Exception as e:
            logger.error(f"Google TTS Error: {e}")
            return None
    
    async def synthesize(self, text: str) -> Optional[bytes]:
        """
        Synthesize text based on configured provider
        
        Args:
            text: Text to convert
        
        Returns:
            Audio bytes
        """
        if self.provider == "elevenlabs":
            return await self.synthesize_elevenlabs(text)
        elif self.provider == "google":
            return await self.synthesize_google(text)
        else:
            logger.error(f"Unknown TTS provider: {self.provider}")
            return None

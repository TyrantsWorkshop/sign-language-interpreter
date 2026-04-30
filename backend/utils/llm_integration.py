"""
Language Model Integration - ChatGPT API
"""

import openai
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LLMProcessor:
    """Process sign language text through ChatGPT for contextualization"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", 
                 max_tokens: int = 100, temperature: float = 0.7):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        openai.api_key = api_key
    
    async def contextualize_signs(self, sign_sequence: str) -> Optional[str]:
        """
        Generate meaningful sentence from recognized signs
        
        Args:
            sign_sequence: Space-separated signs/words (e.g., "HELLO WORLD")
        
        Returns:
            Contextual sentence from LLM
        """
        try:
            system_prompt = (
                "You are a helpful assistant interpreting sign language. "
                "The user has signed the following words or letters. "
                "Create a meaningful sentence or response based on what they signed. "
                "Keep responses concise and helpful. Response should be 1-2 sentences."
            )
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"The user signed: {sign_sequence}"}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response['choices'][0]['message']['content']
        
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return f"Recognized: {sign_sequence}"
    
    async def generate_response(self, user_input: str, context: str = "") -> Optional[str]:
        """
        Generate contextual response
        
        Args:
            user_input: User message
            context: Additional context
        
        Returns:
            Generated response
        """
        try:
            messages = [
                {"role": "system", "content": "You are a helpful sign language interpreter assistant."}
            ]
            
            if context:
                messages.append({"role": "assistant", "content": context})
            
            messages.append({"role": "user", "content": user_input})
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response['choices'][0]['message']['content']
        
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return None

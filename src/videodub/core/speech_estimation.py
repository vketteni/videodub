"""Speech duration estimation for natural timing alignment."""

import re
from typing import Optional, Dict, Any


class SpeechDurationEstimator:
    """Estimates natural speech duration from text content.
    
    Uses syllable-based estimation with adjustments for text complexity,
    punctuation, and content type to provide realistic speech timing.
    """
    
    # Base speech rate in syllables per second (average conversational rate)
    BASE_SYLLABLES_PER_SECOND = 4.5
    
    # Minimum and maximum duration bounds in milliseconds
    MIN_DURATION_MS = 300  # 0.3 seconds
    MAX_DURATION_MS = 10000  # 10.0 seconds
    
    # Punctuation pause adjustments in milliseconds
    PUNCTUATION_PAUSES = {
        '.': 200,
        '!': 200,
        '?': 200,
        ',': 100,
        ';': 150,
        ':': 150,
        '-': 50,
        '...': 300,
        '—': 100,
    }
    
    # Content type multipliers
    CONTENT_TYPE_MULTIPLIERS = {
        'music': 0.5,      # "[Music]" tags are shorter
        'sound': 0.5,      # "[Sound]" tags are shorter
        'applause': 0.3,   # "[Applause]" is brief
        'laughter': 0.4,   # "[Laughter]" is brief
        'numbers': 1.3,    # Numbers take longer to say
        'technical': 1.2,  # Technical terms are slower
    }
    
    def __init__(self, 
                 base_rate: float = BASE_SYLLABLES_PER_SECOND,
                 min_duration_ms: int = MIN_DURATION_MS,
                 max_duration_ms: int = MAX_DURATION_MS):
        """Initialize speech duration estimator.
        
        Args:
            base_rate: Base syllables per second for estimation
            min_duration_ms: Minimum duration in milliseconds
            max_duration_ms: Maximum duration in milliseconds
        """
        self.base_rate = base_rate
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
    
    def estimate_duration_ms(self, text: str) -> int:
        """Estimate speech duration for given text in milliseconds.
        
        Args:
            text: Text content to estimate duration for
            
        Returns:
            Estimated duration in milliseconds
        """
        if not text.strip():
            return self.min_duration_ms
        
        # Calculate base syllable count
        syllable_count = self._count_syllables(text)
        
        # Apply content type adjustments
        content_multiplier = self._get_content_multiplier(text)
        
        # Calculate punctuation pause time
        punctuation_pause_ms = self._calculate_punctuation_pauses(text)
        
        # Calculate base speech duration in milliseconds
        base_duration_ms = (syllable_count / self.base_rate) * 1000 * content_multiplier
        
        # Add punctuation pauses
        total_duration_ms = base_duration_ms + punctuation_pause_ms
        
        # Apply bounds
        return max(self.min_duration_ms, min(int(total_duration_ms), self.max_duration_ms))
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text using heuristic approach.
        
        Args:
            text: Text to count syllables for
            
        Returns:
            Estimated syllable count
        """
        # Remove punctuation and convert to lowercase
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Handle empty text
        if not clean_text.strip():
            return 1
        
        # Count vowel groups as syllables
        vowel_groups = re.findall(r'[aeiouy]+', clean_text)
        syllable_count = len(vowel_groups)
        
        # Adjust for common patterns
        # Silent 'e' at end of words
        silent_e = len(re.findall(r'\b\w*e\b', clean_text))
        syllable_count -= silent_e
        
        # Double vowels often count as one syllable
        double_vowels = len(re.findall(r'[aeiouy]{2,}', clean_text))
        syllable_count -= double_vowels * 0.5
        
        # Every word has at least one syllable
        word_count = len(clean_text.split())
        syllable_count = max(syllable_count, word_count)
        
        return max(1, int(syllable_count))
    
    def _get_content_multiplier(self, text: str) -> float:
        """Get content type multiplier based on text characteristics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Multiplier for speech rate adjustment
        """
        text_lower = text.lower()
        
        # Check for bracketed content (like [Music], [Applause])
        if text.startswith('[') and text.endswith(']'):
            content = text_lower[1:-1]
            for content_type, multiplier in self.CONTENT_TYPE_MULTIPLIERS.items():
                if content_type in content:
                    return multiplier
            return 0.5  # Default for bracketed content
        
        # Check for numbers
        if re.search(r'\d', text):
            return self.CONTENT_TYPE_MULTIPLIERS['numbers']
        
        # Check for technical content (lots of capital letters, technical terms)
        if len(re.findall(r'[A-Z]', text)) > len(text) * 0.3:
            return self.CONTENT_TYPE_MULTIPLIERS['technical']
        
        return 1.0  # Default multiplier
    
    def _calculate_punctuation_pauses(self, text: str) -> int:
        """Calculate total pause time for punctuation marks.
        
        Args:
            text: Text to analyze for punctuation
            
        Returns:
            Total pause time in milliseconds
        """
        total_pause_ms = 0
        
        for punct, pause_ms in self.PUNCTUATION_PAUSES.items():
            count = text.count(punct)
            total_pause_ms += count * pause_ms
        
        return total_pause_ms
    
    def estimate_with_metadata(self, text: str) -> Dict[str, Any]:
        """Estimate duration with detailed metadata.
        
        Args:
            text: Text content to estimate duration for
            
        Returns:
            Dictionary with duration and estimation metadata
        """
        syllable_count = self._count_syllables(text)
        content_multiplier = self._get_content_multiplier(text)
        punctuation_pause_ms = self._calculate_punctuation_pauses(text)
        
        base_duration_ms = (syllable_count / self.base_rate) * 1000 * content_multiplier
        total_duration_ms = base_duration_ms + punctuation_pause_ms
        final_duration_ms = max(self.min_duration_ms, min(int(total_duration_ms), self.max_duration_ms))
        
        return {
            'estimated_duration_ms': final_duration_ms,
            'syllable_count': syllable_count,
            'content_multiplier': content_multiplier,
            'punctuation_pause_ms': punctuation_pause_ms,
            'base_duration_ms': base_duration_ms,
            'was_clamped': final_duration_ms != int(total_duration_ms),
            'text_length': len(text),
            'word_count': len(text.split()),
        }
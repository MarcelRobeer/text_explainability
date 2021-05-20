import re
import string
from typing import Sequence, Iterable

def default_tokenizer(input: str) -> Sequence[str]:
    """Simple regex tokenizer."""
    return re.findall(r"\w+|[^\w\s]+", input)

def default_detokenizer(input: Iterable[str]) -> str:
    """Simple regex detokenizer, ideally resulting in `i = detokenizer(tokenizer(i))`."""
    out = " ".join(input).replace("`` ", '"') \
                         .replace(" ''", '"') \
                         .replace('. . .', '...') \
                         .replace(" ( ", " (") \
                         .replace(" ) ", ") ")
    out = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", out)
    out = re.sub(r' ([.,:;?!%]+)$', r"\1", out)
    out = re.sub(r'(\s+[0-9]+):\s+([0-9]+\s+)', r"\1:\2", out)
    return out.replace(" ` ", " '").strip()


PUNCTUATION = list(string.punctuation)

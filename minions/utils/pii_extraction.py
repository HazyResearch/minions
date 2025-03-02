import re
import spacy
from typing import Dict, List, Any


class PIIExtractor:
    """
    A class to extract personally identifiable information (PII) from text.
    """

    def __init__(self):
        """Initialize the PII extractor with regex patterns and NLP model."""
        try:
            self.nlp = spacy.load("en_core_web_md")
        except:
            # Fallback to smallest model as last resort
            self.nlp = spacy.load("en_core_web_sm")
            print(
                "Warning: Using smaller spaCy model. For better results, install larger models."
            )

        # Compile regex patterns for various PII types
        self.patterns = {
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(
                r"\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b"
            ),
            "ssn": re.compile(r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"),
            "credit_card": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
            "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
            "date_of_birth": re.compile(
                r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b",
                re.IGNORECASE,
            ),
            "url": re.compile(
                r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(/[-\w%!$&\'()*+,;=:]+)*"
            ),
            "zipcode": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
        }

    def extract_pii(self, text: str) -> Dict[str, List[str]]:
        """
        Extract PII from the given text and return as a dictionary.

        Args:
            text (str): The text to extract PII from

        Returns:
            Dict[str, List[str]]: Dictionary with PII types as keys and lists of found instances as values
        """
        if not text or not isinstance(text, str):
            return {"error": ["Invalid or empty input"]}

        # Initialize results dictionary
        pii_data = {
            "person_names": [],
            "organizations": [],
            "locations": [],
            "emails": [],
            "phone_numbers": [],
            "ssns": [],
            "credit_cards": [],
            "ip_addresses": [],
            "dates_of_birth": [],
            "urls": [],
            "zipcodes": [],
        }

        # Extract entities using spaCy
        doc = self.nlp(text)

        # Extract named entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if ent.text not in pii_data["person_names"]:
                    pii_data["person_names"].append(ent.text)
            elif ent.label_ == "ORG":
                if ent.text not in pii_data["organizations"]:
                    pii_data["organizations"].append(ent.text)
            elif ent.label_ in ["GPE", "LOC"]:
                if ent.text not in pii_data["locations"]:
                    pii_data["locations"].append(ent.text)

        # Extract regex-based patterns
        pii_data["emails"] = self._find_matches(self.patterns["email"], text)
        pii_data["phone_numbers"] = self._find_matches(self.patterns["phone"], text)
        pii_data["ssns"] = self._find_matches(self.patterns["ssn"], text)
        pii_data["credit_cards"] = self._find_matches(
            self.patterns["credit_card"], text
        )
        pii_data["ip_addresses"] = self._find_matches(self.patterns["ip_address"], text)
        pii_data["dates_of_birth"] = self._find_matches(
            self.patterns["date_of_birth"], text
        )
        pii_data["urls"] = self._find_matches(self.patterns["url"], text)
        pii_data["zipcodes"] = self._find_matches(self.patterns["zipcode"], text)

        # Remove empty lists from results
        return {k: v for k, v in pii_data.items() if v}

    def _find_matches(self, pattern: re.Pattern, text: str) -> List[str]:
        """Find all unique matches for a regex pattern in the text."""
        matches = pattern.findall(text)
        return list(set(matches))  # Remove duplicates

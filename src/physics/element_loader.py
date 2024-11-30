from core.element_data import ELEMENT_DATA

class ElementLoader:
    """Handles loading and caching of element data"""
    _element_cache = {}

    @classmethod
    def get_element(cls, element_type: str):
        """Get element data for a given element type"""
        if element_type not in cls._element_cache:
            if element_type not in ELEMENT_DATA:
                raise ValueError(f"Unknown element type: {element_type}")
            cls._element_cache[element_type] = ELEMENT_DATA[element_type]
        return cls._element_cache[element_type]

    @classmethod
    def clear_cache(cls):
        """Clear the element cache"""
        cls._element_cache.clear() 
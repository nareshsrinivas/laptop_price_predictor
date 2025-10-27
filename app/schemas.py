from pydantic import BaseModel

class LaptopFeatures(BaseModel):
    brand: str
    ram: int
    storage: int
    weight: float
    cpu_freq_ghz: float

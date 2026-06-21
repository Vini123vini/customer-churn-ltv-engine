from src.database import Base, engine
from src.models import CustomerPrediction

Base.metadata.create_all(bind=engine)
print("✅ Tables created successfully!")
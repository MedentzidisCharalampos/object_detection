from backend.db.models import Base
from backend.db.session import engine

Base.metadata.create_all(bind=engine)

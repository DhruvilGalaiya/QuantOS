from sqlalchemy import Column, Integer, String, Float, DateTime

from app.database import Base


class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)

    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(50), nullable=False, index=True)

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)

    adjusted_close = Column(Float)
    volume = Column(Float)

    source = Column(String(50))
from pydantic import BaseModel, Field

class CreditScoringRequest(BaseModel):
    Amount: float = Field(..., description="Current transaction amount", example=250.50)
    Value: float = Field(..., description="Transaction valuation index", example=250.50)
    Total_Amount: float = Field(..., description="Cumulative historical transaction sum", example=1500.00)
    Avg_Amount: float = Field(..., description="Historical transaction mean", example=300.00)
    Transaction_Count: int = Field(..., description="Total count of historical transactions", example=5)
    Std_Amount: float = Field(..., description="Standard deviation of transaction amounts", example=45.20)
    ProductCategory_Encoded: int = Field(..., description="Label or Hot encoded index value", example=1)
    ChannelId_Encoded: int = Field(..., description="Channel routing identification flag index", example=3)

class CreditScoringResponse(BaseModel):
    is_high_risk: int = Field(..., description="Binary prediction index where 1=High Risk, 0=Safe")
    risk_probability: float = Field(..., description="Calculated probability score of customer default risk")
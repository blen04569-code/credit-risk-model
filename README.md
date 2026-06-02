Credit Scoring Business Understanding
1. Basel II Framework & Model Interpretability

The Basel II Accord structures banking supervision into three core pillars: Minimum Capital Requirements, Supervisory Review, and Market Discipline. Under the Internal Ratings-Based (IRB) approach of Pillar 1, financial institutions are permitted to use their internal predictive models to estimate credit risk components, specifically the Probability of Default (PD).

Because these models directly dictate the amount of risk-weighted capital Bati Bank must hold on its balance sheet, regulators mandate absolute transparency and strict auditability:

    The Right to an Explanation: Regulators and consumers require clear visibility into why a credit decision was made. If an applicant is denied a buy-now-pay-later (BNPL) line of credit, the bank must be able to cite the exact risk drivers (e.g., high transaction volatility or low recency).

    Model Risk Management: A well-documented model ensures that the underlying statistical assumptions, data line lines, and mathematical transformations (such as Weight of Evidence) can be independently replicated and stress-tested by internal risk teams and external auditors.

    Prevention of Bias: Regulators require documentation proving that the model does not ingest or proxy prohibited discriminatory traits, ensuring ethical and fair lending practices.

2. The Necessity and Risks of Proxy Targets

In standard credit analytics, models are trained on historical loan portfolios with definitive "matured" outcomes—specifically, whether a borrower crossed a regulatory delinquency threshold (such as 90+ days past due). Because our eCommerce partner dataset only captures raw transactional histories and lacks credit performance history, a proxy variable (derived from RFM segmentation) is mathematically necessary to establish a supervised learning target.

While proxy engineering unlocks the ability to build predictive risk signals, it introduces distinct business risks:

    Target Misalignment (Basis Risk): A customer flagged as "Bad" due to low platform activity (low RFM score) might simply prefer using alternative cash payment channels, rather than actually being financially insolvent. Conversely, an active, high-volume transactor might be over-leveraged and close to bankruptcy, exposing the bank to unexpected defaults.

    Economic Regime Sensitivity: Transactional behaviors shift faster than core creditworthiness. A seasonal dip in eCommerce purchasing power could cause the model to systematically misclassify creditworthy applicants as high-risk, resulting in artificial credit tightening.

    Adverse Selection: If our synthetic target incorrectly opens credit lines to volatile users who mimic "Good" purchasing signals right before churning, the portfolio will accumulate hidden toxic debt.

3. Model Architecture Trade-Offs: Linear vs. Ensemble

In a regulated banking ecosystem like Bati Bank, selecting a model architecture requires a delicate balance between predictive discrimination and operational transparency:
Evaluation Dimension	

Simple Interpretable Model

(e.g., Logistic Regression + WoE)
	

High-Performance Ensemble

(e.g., Gradient Boosting / XGBoost)
Predictive Power (AUC-ROC)	Moderate. Captures monotonic, linear relationships between binned risk features and the target.	High. Optimally captures complex, non-linear feature interactions and subtle transactional variations.
Interpretability	Extremely High. Yields a deterministic scorecard where individual attribute bins map directly to credit score points.	Low (Black Box). Requires secondary framework overlays (like SHAP or LIME) to approximate explanation vectors.
Regulatory Approval Path	Streamlined. Fully compliant with traditional Basel II scorecards; easily understood and trusted by risk committees.	Challenging. Demands comprehensive algorithmic auditing, drift monitoring, and robust documentation of edge-case behaviors.
Operational Stability	High. Highly resistant to overfitting; minor data anomalies or missing categorical fields do not degrade stability.	Medium. Prone to overfitting on minor behavioral noise if regularization parameters are not explicitly controlled.
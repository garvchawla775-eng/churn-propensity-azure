# Deploying to Azure

This folder is deployment-ready but **not deployed** — deploying requires
your own Azure subscription and credentials, which I can't act on for you.
Here's how to actually deploy it once you have an Azure account:

## Option A — Azure ML Managed Online Endpoint (closer to how TD's Analytics team would run this)

```bash
az ml model create --name churn-propensity-model --path ../outputs/model.pkl
az ml online-endpoint create --name churn-endpoint --auth-mode key
az ml online-deployment create --name churn-deploy \
    --endpoint churn-endpoint \
    --model churn-propensity-model:1 \
    --code-configuration code=. scoring-script=score.py \
    --instance-type Standard_DS2_v2 --instance-count 1
```

## Option B — Azure Function (lighter weight, good for a portfolio demo)

```bash
func init churn-function --python
# copy function_app.py, score.py, and outputs/model.pkl into the function folder
func azure functionapp publish <your-function-app-name>
```

## Testing locally first (no Azure needed)

```bash
pip install azure-functions
func start
curl -X POST http://localhost:7071/api/predict_churn \
  -H "Content-Type: application/json" \
  -d '{"data": [{"tenure_months": 5, "monthly_charge": 80.0, "contract_type": "month-to-month", "num_support_calls": 3, "has_autopay": 0, "num_products": 1, "satisfaction_score": 2}]}'
```

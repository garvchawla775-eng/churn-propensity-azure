"""
HTTP-triggered Azure Function that wraps score.py, for a lightweight
serverless deployment alternative to a full Azure ML managed endpoint.

Deploy with the Azure Functions Core Tools:
    func azure functionapp publish <your-function-app-name>
"""
import json
import azure.functions as func
import score

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
score.init()


@app.route(route="predict_churn", methods=["POST"])
def predict_churn(req: func.HttpRequest) -> func.HttpResponse:
    try:
        result = score.run(req.get_body())
        return func.HttpResponse(json.dumps(result), mimetype="application/json", status_code=200)
    except Exception as e:
        return func.HttpResponse(json.dumps({"error": str(e)}), mimetype="application/json", status_code=400)
